"""PaddleOCR MCP server (FastMCP).

Exposes two tools:
  - recognize_text: run OCR over an image (path / URL / base64 / data-URI)
  - list_supported_languages: report the supported language codes

Transport is selected via the MCP_TRANSPORT env var:
  - "stdio" (default): for local MCP clients such as Claude Desktop
  - "streamable-http": for running as a network/Docker service
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any

import numpy as np

import anyio

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .ocr_engine import (
    LANG_DESCRIPTIONS,
    SUPPORTED_LANGS,
    OCREngine,
    OCRLoaderError,
    load_image,
    release_input,
    resolve_lang,
    run_ocr,
)
from .layout_engine import (
    LayoutEngine,
    LayoutEngineError,
    run_layout,
)

logger = logging.getLogger("ocr_mcp.server")


def _setup_logging() -> None:
    """Configure root logging once at startup.

    PaddleOCR/PaddleX dump a lot at INFO; we only promote our own package to
    the configured level and keep noisy third-party libs at WARNING so the
    server log stays readable. Override the level with OCR_LOG_LEVEL.
    """
    level = os.getenv("OCR_LOG_LEVEL", "INFO").strip().upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quieten the very chatty paddle/paddlex loggers unless explicitly lowered.
    for noisy in ("paddle", "paddlex", "paddleocr"):
        if logging.getLogger(noisy).level == logging.NOTSET:
            logging.getLogger(noisy).setLevel(logging.WARNING)


mcp = FastMCP(
    "PaddleOCR",
    instructions=(
        "OCR service backed by PaddleOCR (PP-OCRv5). Supports Chinese, English, "
        "Japanese and Korean. Accepts images, PDFs and office documents "
        "(doc/docx/ppt/pptx/xls/xlsx) via local path, http(s) URL, or base64; "
        "office docs are converted to PDF first. Use list_supported_languages "
        "to see language codes, "
        "then recognize_text to extract text (with bounding boxes); "
        "recognize_layout to recover page structure and tables (HTML + Markdown)."
    ),
)

# A single engine registry is shared across all tool calls; models load lazily.
_engine = OCREngine()
_layout_engine = LayoutEngine()

def _build_transport_security() -> TransportSecuritySettings:
    """Extend the DNS-rebinding allowlist so network clients can reach us.

    FastMCP's default allowlist only permits loopback hosts. When the service
    is reached by LAN IP or hostname, requests are rejected with
    "Invalid Host header", so MCP clients like Codex never connect. Extra
    hosts/origins come from MCP_ALLOWED_HOSTS / MCP_ALLOWED_ORIGINS (csv).
    """
    extra_hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    extra_origins = [o.strip() for o in os.getenv("MCP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"] + extra_hosts,
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ] + extra_origins,
    )

def _blocking_recognize(engine: Any, image_input: Any) -> list[dict[str, Any]]:
    """Thin sync wrapper run on a worker thread (PaddleOCR is CPU-bound)."""
    return run_ocr(engine, image_input)


def _describe_input(image_input: Any) -> str:
    """Human-readable one-liner describing a loaded input for log messages."""
    if isinstance(image_input, str):
        tag = "pdf-path" if image_input.lower().endswith(".pdf") else "file-path"
        return f"{tag}: {image_input}"
    if isinstance(image_input, np.ndarray):
        return f"ndarray: {image_input.shape}"
    return f"{type(image_input).__name__}: {image_input!r}"


@mcp.tool()
async def recognize_text(
    image: str,
    language: str = "ch",
    detail: bool = False,
    min_confidence: float = 0.0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Recognize text in an image using PaddleOCR (PP-OCRv5).

    Args:
        image: Local file path (image, PDF, or office doc), an http(s) URL, a
            data URI ("data:image/png;base64,..."), or a raw base64 string.
            PDFs are read natively by PaddleOCR page by page; office documents
            (doc/docx/ppt/pptx/xls/xlsx/...) are first converted to PDF via
            LibreOffice. Office docs via URL use the file extension; base64
            office docs are not supported.
        language: OCR language - "ch" (Chinese+English), "en" (English),
            "japan" (Japanese), "korean" (Korean). Default "ch".
        detail: When true, also include per-line confidence scores. Bounding
            boxes are always returned (one per line) so the coordinates are
            available without enabling detail.
        min_confidence: Drop lines below this confidence (0.0-1.0). 0 keeps all.

        Returns:
            A dict with "language", "count", "recognized_text" (newline-joined),
            and a "lines" list of {text, box} (plus "confidence" when
            detail=true).
            Note: the top-level key is "recognized_text" rather than "text"
            because Dify reserves "text" as a workflow variable name.
        """
    logger.info("recognize_text: start (lang=%s, detail=%s, min_confidence=%s)", language, detail, min_confidence)
    try:
        lang = resolve_lang(language)
        logger.info("recognize_text: resolved language '%s' -> '%s'", language, lang)
        engine = _engine.get_engine(lang)
        logger.info("recognize_text: engine ready for lang '%s'", lang)
        image_input = load_image(image)
        logger.info("recognize_text: input loaded -> %s", _describe_input(image_input))
    except (ValueError, OCRLoaderError) as exc:
        logger.error("recognize_text: input/engine failed: %s", exc, exc_info=True)
        return {"error": str(exc), "language": None, "count": 0, "recognized_text": ""}

    if ctx is not None:
        await ctx.info(f"Running OCR ({lang}) ...")

    try:
        logger.info("recognize_text: running PaddleOCR.predict (offloaded to worker thread)")
        # Offload the blocking inference so the event loop stays responsive.
        items = await anyio.to_thread.run_sync(_blocking_recognize, engine, image_input)
        logger.info("recognize_text: predict done, %d lines", len(items))

        if min_confidence and min_confidence > 0.0:
            before = len(items)
            items = [it for it in items if (it["confidence"] or 0.0) >= min_confidence]
            logger.info("recognize_text: min_confidence filter %d -> %d lines", before, len(items))

        lines = [it["text"] for it in items]
        result: dict[str, Any] = {
            "language": lang,
            "count": len(lines),
            "recognized_text": "\n".join(lines),
            # Each line always carries its bounding box (and, for multi-page
            # PDFs, its 0-based page index) so callers can place text on the
            # page; confidence is opt-in via ``detail`` to keep the default
            # payload compact.
            "lines": [
                {
                    "text": it["text"],
                    "box": it["box"],
                    "page": it["page"],
                    **({"confidence": it["confidence"]} if detail else {}),
                }
                for it in items
            ],
        }
        logger.info("recognize_text: success, %d lines returned", len(lines))
        return result
    except Exception as exc:
        # PaddleOCR inference (predict) can fail at runtime for many reasons
        # (OOM, corrupt image, model load failure, native segfault-adjacent
        # crashes, ...). Catch them here so the client gets a structured error
        # instead of a raw traceback, and the server log records the full stack.
        logger.error("recognize_text: inference failed: %s", exc, exc_info=True)
        return {
            "error": f"OCR inference failed: {exc}",
            "language": lang,
            "count": 0,
            "recognized_text": "",
        }
    finally:
        # Remove temp PDFs created for URL/base64 sources; no-op for images.
        release_input(image_input)


@mcp.tool()
def list_supported_languages() -> dict[str, Any]:
    """List the OCR languages this server supports, with descriptions."""
    return {
        "languages": SUPPORTED_LANGS,
        "descriptions": {lang: LANG_DESCRIPTIONS[lang] for lang in SUPPORTED_LANGS},
    }


@mcp.tool()
async def recognize_layout(
    image: str,
    language: str = "ch",
    output: str = "markdown",
    flat: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Recover page structure (layout + tables) using PP-Structure.

    Unlike recognize_text (flat text lines), this runs full layout analysis:
    it segments the page into regions (title / text / table / figure /
    formula / ...), recognizes tables and converts them to both HTML and
    Markdown, and reconstructs the reading order.

    Args:
        image: Local file path (image, PDF, or office doc), an http(s) URL, a
            data URI ("data:image/png;base64,..."), or a raw base64 string.
            PDFs are read natively by PaddleOCR page by page; office documents
            (doc/docx/ppt/pptx/xls/xlsx/...) are first converted to PDF via
            LibreOffice. Office docs via URL use the file extension; base64
            office docs are not supported.
        language: OCR language - "ch" / "en" / "japan" / "korean". Default "ch".
        output: Top-level convenience content - "markdown" (the whole page as
            Markdown, tables preserved) or "text" (flattened text only).
           Structured data (regions/tables) is always returned regardless.
        flat: When True (default), serialize regions/tables to JSON strings so
            every field is a basic type (str/int). Required for Dify, whose
            workflow variable system rejects nested dicts with
            "Only basic types and lists are allowed". Pass False for nested
            dicts (Codex / Claude / raw MCP).

    Returns:
        A dict with "language", "regions" (each {type, text, box, and for
        tables html/markdown}), "tables" (each {html, markdown, box,
        cell_count}), and "markdown" (the full page rendered as Markdown with
        table layout preserved). On error returns {"error": ...}.
        """
    logger.info("recognize_layout: start (lang=%s, output=%s, flat=%s)", language, output, flat)
    try:
        lang = resolve_lang(language)
        logger.info("recognize_layout: resolved language '%s' -> '%s'", language, lang)
        engine = _layout_engine.get_engine(lang)
        logger.info("recognize_layout: layout engine ready for lang '%s'", lang)
        image_input = load_image(image)
        logger.info("recognize_layout: input loaded -> %s", _describe_input(image_input))
    except (ValueError, OCRLoaderError, LayoutEngineError) as exc:
        logger.error("recognize_layout: input/engine failed: %s", exc, exc_info=True)
        return {
            "error": str(exc),
            "language": None,
            "region_count": 0,
            "table_count": 0,
            "markdown": "",
            "regions_json": "[]",
            "tables_json": "[]",
        }

    if ctx is not None:
        await ctx.info(f"Running layout analysis ({lang}) ...")

    try:
        logger.info("recognize_layout: running PP-Structure.predict (offloaded to worker thread)")
        structure, md = await anyio.to_thread.run_sync(run_layout, engine, image_input)
        logger.info(
            "recognize_layout: predict done, %d pages, %d regions, %d tables",
            structure.get("page_count"), len(structure.get("regions", [])), len(structure.get("tables", [])),
        )

        regions = structure.get("regions", [])
        tables = structure.get("tables", [])

        if flat:
            # Dify (and similarly strict workflow variable systems) only accept
            # basic types; the per-region/per-table records are dicts, which Dify
            # rejects with "Only basic types and lists are allowed". Serialize them
            # to JSON strings so every field is a basic type, while keeping the
            # Markdown (the main deliverable) as a real string.
            result: dict[str, Any] = {
                "language": lang,
                "region_count": len(regions),
                "table_count": len(tables),
                "markdown": md,
                "regions_json": json.dumps(regions, ensure_ascii=False),
                "tables_json": json.dumps(tables, ensure_ascii=False),
            }
        else:
            result = {
                "language": lang,
                "region_count": len(regions),
                "table_count": len(tables),
                "regions": regions,
                "tables": tables,
                "markdown": md,
            }
        if structure.get("width") is not None:
            result["width"] = structure["width"]
        if structure.get("height") is not None:
            result["height"] = structure["height"]
        if structure.get("page_count") is not None:
            result["page_count"] = structure["page_count"]
        if output == "text":
            result["text"] = "\n\n".join(r.get("text", "") for r in regions)
        logger.info(
            "recognize_layout: success, %d regions, %d tables, markdown=%d chars",
            len(regions), len(tables), len(md or ""),
        )
        return result
    except Exception as exc:
        # PP-Structure inference (predict) can fail at runtime for many reasons
        # (OOM, corrupt image, model load failure, dependency errors, ...).
        # Catch them here so the client gets a structured error instead of a raw
        # traceback, and the server log records the full stack.
        logger.error("recognize_layout: inference failed: %s", exc, exc_info=True)
        return {
            "error": f"Layout analysis failed: {exc}",
            "language": lang,
            "region_count": 0,
            "table_count": 0,
            "markdown": "",
            "regions_json": "[]",
            "tables_json": "[]",
        }
    finally:
        # Remove temp PDFs created for URL/base64 sources; no-op for images.
        release_input(image_input)


def _select_transport() -> str:
    raw = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if raw in ("http", "streamable-http", "streamable_http", "http-streamable"):
        return "streamable-http"
    return "stdio"


def main() -> None:
    """Entry point: run the server with the transport from the environment."""
    _setup_logging()
    logger.info("Starting PaddleOCR MCP server")
    mcp.settings.transport_security = _build_transport_security()
    transport = _select_transport()
    if transport == "streamable-http":
        mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
        print(
            f"PaddleOCR MCP server listening on "
            f"http://{mcp.settings.host}:{mcp.settings.port}/mcp "
            f"(streamable-http)",
            flush=True,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

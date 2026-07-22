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
from typing import Any

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

mcp = FastMCP(
    "PaddleOCR",
    instructions=(
        "OCR service backed by PaddleOCR (PP-OCRv5). Supports Chinese, English, "
        "Japanese and Korean. Accepts images and PDFs (local path, http(s) URL, "
        "or base64). Use list_supported_languages to see language codes, "
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
        image: Local file path (image or PDF), an http(s) URL, a data URI
            ("data:image/png;base64,..."), or a raw base64 string. PDF files
            are read natively by PaddleOCR and processed page by page.
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
    try:
        lang = resolve_lang(language)
        engine = _engine.get_engine(lang)
        image_input = load_image(image)
    except (ValueError, OCRLoaderError) as exc:
        return {"error": str(exc), "language": None, "count": 0, "recognized_text": ""}

    if ctx is not None:
        await ctx.info(f"Running OCR ({lang}) ...")

    try:
        # Offload the blocking inference so the event loop stays responsive.
        items = await anyio.to_thread.run_sync(_blocking_recognize, engine, image_input)

        if min_confidence and min_confidence > 0.0:
            items = [it for it in items if (it["confidence"] or 0.0) >= min_confidence]

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
        return result
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
        image: Local file path (image or PDF), an http(s) URL, a data URI
            ("data:image/png;base64,..."), or a raw base64 string. PDF files
            are read natively by PaddleOCR and processed page by page.
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
    try:
        lang = resolve_lang(language)
        engine = _layout_engine.get_engine(lang)
        image_input = load_image(image)
    except (ValueError, OCRLoaderError, LayoutEngineError) as exc:
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
        structure, md = await anyio.to_thread.run_sync(run_layout, engine, image_input)

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
        return result
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

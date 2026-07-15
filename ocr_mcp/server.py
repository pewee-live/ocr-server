"""PaddleOCR MCP server (FastMCP).

Exposes two tools:
  - recognize_text: run OCR over an image (path / URL / base64 / data-URI)
  - list_supported_languages: report the supported language codes

Transport is selected via the MCP_TRANSPORT env var:
  - "stdio" (default): for local MCP clients such as Claude Desktop
  - "streamable-http": for running as a network/Docker service
"""

from __future__ import annotations

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
    resolve_lang,
    run_ocr,
)

mcp = FastMCP(
    "PaddleOCR",
    instructions=(
        "OCR service backed by PaddleOCR (PP-OCRv5). Supports Chinese, English, "
 "Japanese and Korean. Use list_supported_languages to see language codes, "
        "then recognize_text to extract text from an image."
    ),
)

# A single engine registry is shared across all tool calls; models load lazily.
_engine = OCREngine()

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
        image: Local file path, an http(s) URL, a data URI
            ("data:image/png;base64,..."), or a raw base64 string.
        language: OCR language - "ch" (Chinese+English), "en" (English),
            "japan" (Japanese), "korean" (Korean). Default "ch".
        detail: When true, include per-line bounding boxes and confidences.
        min_confidence: Drop lines below this confidence (0.0-1.0). 0 keeps all.

        Returns:
            A dict with "language", "count", "recognized_text" (newline-joined),
            and when detail=true a "lines" list of {text, confidence, box}.
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

    # Offload the blocking inference so the event loop stays responsive.
    items = await anyio.to_thread.run_sync(_blocking_recognize, engine, image_input)

    if min_confidence and min_confidence > 0.0:
        items = [it for it in items if (it["confidence"] or 0.0) >= min_confidence]

    lines = [it["text"] for it in items]
    result: dict[str, Any] = {
        "language": lang,
        "count": len(lines),
        "recognized_text": "\n".join(lines),
    }
    if detail:
        result["lines"] = items
    return result


@mcp.tool()
def list_supported_languages() -> dict[str, Any]:
    """List the OCR languages this server supports, with descriptions."""
    return {
        "languages": SUPPORTED_LANGS,
        "descriptions": {lang: LANG_DESCRIPTIONS[lang] for lang in SUPPORTED_LANGS},
    }


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

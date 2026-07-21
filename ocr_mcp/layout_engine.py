"""Layout / table analysis engine (PP-Structure).

Wraps PaddleOCR's layout-analysis pipeline so the MCP server can do more than
plain text recognition: it recovers the *structure* of a page -- titles,
paragraphs, figures and especially *tables* -- and returns it as structured
data plus a Markdown rendering that keeps table layout intact.

paddleocr >=3.0 dropped the legacy ``PPStructure`` class. The current entry
point is ``PPStructureV3`` (a thin wrapper over the PaddleX ``PP-StructureV3``
pipeline), whose ``predict`` returns PaddleX result objects exposing ``.json``
and ``.markdown``. This module targets that 3.x API (matching the paddleocr
pin in requirements.txt); a defensive import shim falls back to the legacy
``PPStructure`` when an older paddleocr is detected so the server still boots.
"""

from __future__ import annotations

import html as _html
import os
import re
import threading
from typing import Any

# The shared image-loading helpers (path / URL / base64 / data-URI) live in the
# OCR module; layout analysis needs exactly the same input parsing.
from .ocr_engine import OCRLoaderError, _resize_for_ocr, load_image


class LayoutEngineError(RuntimeError):
    """Raised when layout analysis cannot be performed."""


# --------------------------------------------------------------------------- #
# PP-StructureV3 instance management
# --------------------------------------------------------------------------- #
def _create_pp_structure(lang: str) -> Any:
    """Build a configured PP-StructureV3 (paddleocr 3.x) engine for ``lang``.

    Disables the optional doc-orientation/unwarping sub-pipelines by default to
    keep CPU inference fast; table recognition is always on, since that is the
    whole point of this module. Honors the same OCR_ENGINE / OCR_DEVICE env
   knobs as the text engine so deployments stay consistent.
    Under the onnxruntime engine (auto-selected on aarch64, or forced via
    OCR_ENGINE) the formula / chart / seal sub-pipelines are disabled too,
    because those models have no onnx package in paddlex's official model source
    and would crash the pipeline at load time. Set OCR_LAYOUT_FULL=1 to attempt
    loading them anyway (only meaningful with the native paddle engine).
    """
    from .ocr_engine import _select_inference_engine

    init_kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
       "device": os.getenv("OCR_DEVICE", "cpu"),
   }
    engine_kind = _select_inference_engine()
    if engine_kind == "onnxruntime":
        init_kwargs["engine"] = "onnxruntime"
        # These three sub-pipelines ship NO onnx model package (see paddlex's
        # ONNX_SUPPORTED_MODELS whitelist: PP-FormulaNet, seal and chart models
        # are absent), so under the onnxruntime engine they fail at load with
        # "does not provide a 'onnx' package for model 'PP-FormulaNet_plus-L'".
        # The onnx engine is auto-selected on aarch64 (and forced by OCR_ENGINE)
        # to dodge native paddle segfaults, so disable them there. On x86_64
        # with the native paddle engine they stay on (OCR_LAYOUT_FULL=1 forces
        # them on even under onnx, at the cost of a guaranteed load failure).
        if os.getenv("OCR_LAYOUT_FULL", "0").strip() not in ("1", "true", "yes"):
            init_kwargs["use_formula_recognition"] = False
            init_kwargs["use_chart_recognition"] = False
            init_kwargs["use_seal_recognition"] = False
   # On native paddle we leave enable_mkldnn at its default; PP-StructureV3's
    # config does not expose it as a ctor kwarg the way PaddleOCR does.
    return _PPStructureCls()(lang=lang, **init_kwargs)


def _PPStructureCls() -> Any:
    """Return the best available layout-analysis class for the installed paddleocr.

    paddleocr 3.x exposes ``PPStructureV3``; older 2.x wheels expose the legacy
    ``PPStructure``. Importing is deferred so the (heavy) paddle stack only
    loads when layout analysis is actually requested.
    """
    try:
        from paddleocr import PPStructureV3  # paddleocr >=3.0
    except ImportError:
        try:
            from paddleocr import PPStructure  # legacy 2.x fallback
        except Exception as exc:  # pragma: no cover - defensive
            raise LayoutEngineError(
                "paddleocr is not installed or does not expose a layout-analysis "
                "pipeline (PPStructureV3 / PPStructure)."
            ) from exc
        return PPStructure
    return PPStructureV3


class LayoutEngine:
    """Manages lazily-initialized, cached, thread-safe PP-Structure instances."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_engine(self, lang: str) -> Any:
        engine = self._engines.get(lang)
        if engine is not None:
            return engine
        with self._lock:
            engine = self._engines.get(lang)
            if engine is not None:
                return engine
            try:
                engine = _create_pp_structure(lang)
            except Exception as exc:
                # paddlex wraps a missing optional dependency (premailer, shapely,
                # openpyxl, ...) into a generic "A dependency error occurred
                # during pipeline creation" RuntimeError, hiding which package is
                # actually missing. Surface the original cause + an install hint.
                msg = str(exc)
                if "dependency" in msg.lower() or "No module" in msg:
                    msg += (
                        " | Layout analysis (PP-StructureV3) needs extra packages "
                        "not pulled in by plain paddleocr. Install them with "
                        "`pip install paddleocr[doc-parser]` (or paddlex[ocr])."
                    )
                raise LayoutEngineError(msg) from exc
            self._engines[lang] = engine
            return engine


# --------------------------------------------------------------------------- #
# HTML table -> Markdown conversion
# --------------------------------------------------------------------------- #
def _clean_cell_text(raw: str) -> str:
    """Collapse whitespace and unescape entities inside a table cell."""
    if not raw:
        return ""
    text = _html.unescape(raw)
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    # Any remaining inline tags (e.g. <b>) are dropped; Markdown tables can't
    # carry arbitrary HTML reliably across renderers.
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Escape pipes so they don't break the Markdown table column count.
    return text.replace("|", "\\|")


def html_table_to_markdown(html: str) -> str:
    """Convert a simple HTML ``<table>`` into a GitHub-Flavored Markdown table.

    Handles ``colspan``/``rowspan`` by *expanding* merged cells: the cell's
    text is repeated across the spanned cells (Markdown tables have no span
    syntax), which preserves the visual grid better than dropping it. Nested
    tables are not supported (PaddleOCR's table model is strictly 2D).
    """
    if not html or "<table" not in html.lower():
        return ""

    rows_raw = re.findall(r"<tr.*?>(.*?)</tr>", html, re.S | re.I)
    if not rows_raw:
        return ""

    # First pass: parse every row into a list of (text, colspan, rowspan).
    parsed: list[list[tuple[str, int, int]]] = []
    for tr in rows_raw:
        cells: list[tuple[str, int, int]] = []
        for m in re.finditer(r"<(td|th)([^>]*)>(.*?)</\1>", tr, re.S | re.I):
            tag_attrs = m.group(2) or ""
            body = m.group(3) or ""
            colspan = 1
            rowspan = 1
            cs = re.search(r"colspan\s*=\s*['\"]?(\d+)", tag_attrs, re.I)
            rs = re.search(r"rowspan\s*=\s*['\"]?(\d+)", tag_attrs, re.I)
            if cs:
                colspan = max(1, int(cs.group(1)))
            if rs:
                rowspan = max(1, int(rs.group(1)))
            cells.append((_clean_cell_text(body), colspan, rowspan))
        if cells:
            parsed.append(cells)

    if not parsed:
        return ""

    # Second pass: expand spans into a uniform grid. ``pending`` maps a column
    # index to (text, rows_remaining) for cells carried down by a rowspan.
    grid: list[list[str | None]] = []
    pending: dict[int, tuple[str, int]] = {}

    def _ensure(row_list: list[str | None], idx: int) -> None:
        while len(row_list) <= idx:
            row_list.append(None)

    for cells in parsed:
        row: list[str | None] = []
        col = 0
        for text, colspan, rowspan in cells:
            # Skip columns still occupied by a rowspan carried from above.
            while pending.get(col) is not None:
                _ensure(row, col)
                row[col] = pending[col][0]
                remaining = pending[col][1] - 1
                if remaining <= 0:
                    pending.pop(col, None)
                else:
                    pending[col] = (pending[col][0], remaining)
                col += 1
            # Place this cell, expanding colspan across ``colspan`` columns.
            for _ in range(colspan):
                _ensure(row, col)
                row[col] = text
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1
        # Drain any trailing rowspan shadows beyond the last real cell.
        while col in pending:
            _ensure(row, col)
            row[col] = pending[col][0]
            remaining = pending[col][1] - 1
            if remaining <= 0:
                pending.pop(col, None)
            else:
                pending[col] = (pending[col][0], remaining)
            col += 1
        grid.append([c if c is not None else "" for c in row])

    width = max((len(r) for r in grid), default=0)
    if width == 0:
        return ""

    # Normalize every row to the grid width.
    grid = [r + [""] * (width - len(r)) for r in grid]

    header = grid[0]
    body = grid[1:] if len(grid) > 1 else []

    def _fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    lines = [_fmt_row(header), separator]
    for r in body:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Result normalization
# --------------------------------------------------------------------------- #
def _as_builtin(value: Any) -> Any:
    """Recursively convert numpy scalars / AttrDicts into JSON-safe builtins."""
    try:
        import numpy as np  # local import; only needed when numpy is around
    except Exception:  # pragma: no cover - numpy is a hard dep, but be safe
        np = None

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(v) for v in value]
    return value


def _bbox_to_list(bbox: Any) -> list[int] | None:
    """Coerce a PaddleOCR/PaddleX bbox (``[x1,y1,x2,y2]``) into a clean list."""
    try:
        return [int(round(float(p))) for p in bbox]
    except (TypeError, ValueError):
        return None


def _result_json(res: Any) -> dict[str, Any]:
    """Best-effort extraction of the JSON dict from a PaddleX result object."""
    json_attr = getattr(res, "json", None)
    if callable(json_attr):
        try:
            data = json_attr()
        except Exception:  # pragma: no cover - defensive
            data = None
    elif isinstance(json_attr, dict):
        data = json_attr
    else:
        data = None
    if isinstance(data, dict):
        # PP-StructureV3 wraps the payload under a "res" key (see
        # LayoutParsingResultV2); unwrap when present.
        inner = data.get("res")
        if isinstance(inner, dict) and (
            "parsing_res_list" in inner or "table_res_list" in inner
        ):
            return inner
        if "parsing_res_list" in data or "table_res_list" in data:
            return data
        return data
    # Fallback: treat the object itself as a mapping (PaddleX results support
    # __getitem__).
    try:
        probe = res.get("parsing_res_list") if hasattr(res, "get") else None
    except Exception:  # pragma: no cover - defensive
        probe = None
    if probe is not None:
        return dict(res)  # type: ignore[arg-type]
    return {}


def _result_markdown(res: Any) -> str:
    """Extract the plain Markdown text (tables as Markdown) from a result.

    PP-StructureV3's ``.markdown`` property yields a dict shaped like
    ``{"markdown_texts": "...", "markdown_images": {...}}``. We prefer the
    non-pretty ("plain") rendering so tables come back as real Markdown tables
    rather than embedded HTML, then post-process any residual HTML tables.
    """
    md_attr = getattr(res, "markdown", None)
    data: Any = None
    if callable(md_attr):
        try:
            data = md_attr()
        except TypeError:
            # Some versions let you toggle pretty via args; fall back.
            try:
                data = md_attr(pretty=False)  # type: ignore[misc]
            except Exception:  # pragma: no cover - defensive
                data = None
        except Exception:  # pragma: no cover - defensive
            data = None
    elif isinstance(md_attr, dict):
        data = md_attr

    if isinstance(data, dict):
        text = data.get("markdown_texts") or data.get("markdown") or ""
    elif isinstance(data, str):
        text = data
    else:
        text = ""
    return _html_tables_to_markdown(text)


def _html_tables_to_markdown(markdown_text: str) -> str:
    """Replace any ``<table>...</table>`` blocks in Markdown with MD tables.

    PP-StructureV3 sometimes emits tables as HTML even in Markdown mode (e.g.
    when ``format_block_content`` is on or spans are involved). Converting them
    here keeps the promise that the returned Markdown uses real table syntax.
    """

    def _repl(match: re.Match[str]) -> str:
        return html_table_to_markdown(match.group(0))

    return re.sub(r"<table.*?</table>", _repl, markdown_text, flags=re.S | re.I)


def _normalize_layout_result(res: Any) -> dict[str, Any]:
    """Turn one PaddleX layout result into the MCP-facing structure dict."""
    data = _result_json(res)

    parsing = (
        data.get("parsing_res_list") if isinstance(data, dict) else None
    ) or []
    table_res_list = (
        data.get("table_res_list") if isinstance(data, dict) else None
    ) or []

    regions: list[dict[str, Any]] = []
    for blk in parsing:
        if not isinstance(blk, dict):
            continue
        label = blk.get("block_label") or blk.get("label")
        content = blk.get("block_content")
        if content is None:
            content = blk.get("content", "")
        region: dict[str, Any] = {
            "type": str(label) if label is not None else "unknown",
            "text": str(content) if content is not None else "",
            "box": _bbox_to_list(blk.get("block_bbox") or blk.get("bbox")),
        }
        order = blk.get("block_order")
        if order is not None:
            region["order"] = order
        # For table blocks, expose both the raw HTML and a Markdown rendering.
        if label == "table" and content:
            region["html"] = str(content)
            md = html_table_to_markdown(str(content))
            if md:
                region["markdown"] = md
        regions.append(region)

    tables: list[dict[str, Any]] = []
    for t in table_res_list:
        if not isinstance(t, dict):
            continue
        html = t.get("pred_html") or t.get("html") or ""
        region_id = t.get("table_region_id")
        cell_boxes = t.get("cell_box_list") or []
        entry: dict[str, Any] = {
            "html": str(html),
            "markdown": html_table_to_markdown(str(html)) if html else "",
            "box": _bbox_to_list(t.get("table_bbox") or t.get("table_region")),
            "cell_count": len(cell_boxes),
        }
        if region_id is not None:
            entry["region_id"] = region_id
        tables.append(entry)

    return {
        "regions": regions,
        "tables": tables,
        "width": _as_builtin(data.get("width")) if isinstance(data, dict) else None,
        "height": _as_builtin(data.get("height")) if isinstance(data, dict) else None,
    }


def run_layout(engine: Any, image_input: Any) -> tuple[dict[str, Any], str]:
    """Run layout analysis and return ``(structure_dict, markdown_text)``.

    ``structure_dict`` has ``regions`` (each with ``type``/``text``/``box`` and,
    for tables, ``html``/``markdown``), ``tables``, and page ``width``/``height``.
    ``markdown_text`` is the whole page rendered as Markdown with tables kept
    as Markdown tables.
    """
    image_input = _resize_for_ocr(image_input)
    results = engine.predict(image_input)
    if results is None:
        return {"regions": [], "tables": []}, ""

    # Predict may yield a list (3.x) or already a single result (legacy 2.x).
    if isinstance(results, list):
        items = results
    else:
        items = [results]

    merged_regions: list[dict[str, Any]] = []
    merged_tables: list[dict[str, Any]] = []
    md_pages: list[str] = []
    width = height = None
    for res in items:
        norm = _normalize_layout_result(res)
        merged_regions.extend(norm["regions"])
        merged_tables.extend(norm["tables"])
        width = norm.get("width") or width
        height = norm.get("height") or height
        md_pages.append(_result_markdown(res))

    structure = {
        "regions": merged_regions,
        "tables": merged_tables,
        "width": width,
        "height": height,
    }
    markdown = "\n\n".join(p for p in md_pages if p and p.strip())
    return structure, markdown


__all__ = [
    "LayoutEngine",
    "LayoutEngineError",
    "html_table_to_markdown",
    "load_image",
    "run_layout",
]

"""PaddleOCR engine wrapper.

Keeps one lazily-initialized, cached PaddleOCR (PP-OCRv5) instance per language,
parses heterogeneous image inputs (local path / http(s) URL / base64 data URI),
and normalizes OCR results into plain dicts.
"""

from __future__ import annotations

import base64
import io
import os
import logging
import platform
import re
import sysconfig
import threading
import tempfile
import shutil
import subprocess
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ocr_mcp.engine")

# paddlepaddle 3.x can segfault on the CPU PIR execution path
# (ConvertPirAttribute2RuntimeAttribute) regardless of `enable_mkldnn`.
# These C++ flags must be set via the environment *before* the paddle runtime
# is imported, so we apply them at module import time. Override with
# OCR_PIR=1 if you ever need the new IR executor back.
def _apply_paddle_flags() -> None:
    if os.getenv("OCR_PIR", "0").strip() in ("1", "true", "yes"):
        return
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")


_apply_paddle_flags()

def _select_inference_engine() -> str | None:
    """Pick the PaddleOCR inference `engine`.

    PaddlePaddle 3.x pre-built aarch64 wheels crash with SIGSEGV at two sites:
      1. model loading (PIR executor corrupts std::filesystem::path), fixed by
         the PIR flags above;
      2. inference (null-pointer deref in native kernels) -- no env flag can
         fix this; the only workaround is ONNX Runtime, which bypasses the
         broken native kernels entirely.

    So on Linux aarch64 (Raspberry Pi, Graviton, Apple Silicon Docker, ...) we
    auto-select `onnxruntime`. Needs the `onnxruntime` + `paddle2onnx` packages.
    Override with OCR_ENGINE (e.g. `paddle`, `onnxruntime`).
    """
    forced = os.getenv("OCR_ENGINE", "").strip().lower()
    if forced:
        return forced
    if sysconfig.get_platform().startswith("linux-aarch64") or platform.machine().lower() in ("aarch64", "arm64"):
        return "onnxruntime"
    return None


# Map human-friendly language names to the lang codes PaddleOCR expects.
LANG_ALIASES: dict[str, str] = {
    "ch": "ch",
    "chs": "ch",
    "chinese": "ch",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "cn": "ch",
    "en": "en",
    "english": "en",
    "japan": "japan",
    "ja": "japan",
    "japanese": "japan",
    "korean": "korean",
    "ko": "korean",
    "kr": "korean",
    "hangul": "korean",
}

# The four languages this service targets, in display order.
SUPPORTED_LANGS: list[str] = ["ch", "en", "japan", "korean"]

LANG_DESCRIPTIONS: dict[str, str] = {
    "ch": "Chinese + English (PP-OCRv5 default, also good for mixed CJK/Latin text)",
    "en": "English (Latin script)",
    "japan": "Japanese",
    "korean": "Korean",
}

_DATA_URI_RE = re.compile(r"^data:[^;]*;base64,(.*)$", re.DOTALL)
_BASE64_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/\s\r\n]+=*$")


class OCRLoaderError(RuntimeError):
    """Raised when an image source cannot be interpreted or decoded."""


def resolve_lang(language: str | None) -> str:
    """Normalize a user-supplied language string to a PaddleOCR lang code."""
    key = (language or "ch").strip().lower()
    code = LANG_ALIASES.get(key)
    if code is None:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: "
            + ", ".join(
                f"'{c}' ({LANG_DESCRIPTIONS[c].split(' (')[0]})"
                for c in SUPPORTED_LANGS
            )
        )
    return code


def _create_paddleocr(
    lang: str,
    det_model: str | None = None,
    rec_model: str | None = None,
) -> Any:
    """Build a configured PaddleOCR instance for ``lang``.

    ``det_model`` / ``rec_model`` default to the OCR_DET_MODEL / OCR_REC_MODEL
    env vars; pass explicit names to override (used by warmup to pre-download
    multiple model variants into the disk cache).
    """
    from paddleocr import PaddleOCR

    # Pin to PP-OCRv5. paddleocr >=3.7 defaults to PP-OCRv6 when `lang` is
    # given but `ocr_version` is omitted; PP-OCRv6 segfaults on CPU under
    # paddlepaddle 3.x even with mkldnn disabled.
    ocr_version = os.getenv("OCR_VERSION", "PP-OCRv5")
    inference_engine = _select_inference_engine()
    init_kwargs: dict[str, Any] = {
        "lang": lang,
        "ocr_version": ocr_version,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        # "cpu" / "gpu" / "gpu:0". Needs a GPU build of paddlepaddle
        # (paddlepaddle-gpu) + NVIDIA driver; see README GPU section.
        "device": os.getenv("OCR_DEVICE", "cpu"),
    }
    det = (det_model or os.getenv("OCR_DET_MODEL", "")).strip()
    rec = (rec_model or os.getenv("OCR_REC_MODEL", "")).strip()
    if det:
        init_kwargs["text_detection_model_name"] = det
    if rec:
        init_kwargs["text_recognition_model_name"] = rec
    if inference_engine == "onnxruntime":
        init_kwargs["engine"] = "onnxruntime"
    else:
        init_kwargs["enable_mkldnn"] = False
    return PaddleOCR(**init_kwargs)


class OCREngine:
    """Manages lazily-initialized, cached, thread-safe PaddleOCR instances."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_engine(self, lang: str) -> Any:
        """Return the cached engine for ``lang``, creating it on first use."""
        engine = self._engines.get(lang)
        if engine is not None:
            return engine

        with self._lock:
            # Re-check inside the lock; another thread may have built it already.
            engine = self._engines.get(lang)
            if engine is not None:
                return engine

            # Imported lazily so the module loads fast and code paths that don't
            # need PaddleOCR aren't forced to import the (heavy) paddle stack.
            from paddleocr import PaddleOCR

            logger.info("[engine] first use for lang '%s', creating PaddleOCR instance (may download models)...", lang)
            import time as _time
            _t0 = _time.monotonic()
            engine = _create_paddleocr(lang)
            logger.info("[engine] PaddleOCR instance for '%s' ready in %.1fs", lang, _time.monotonic() - _t0)
            self._engines[lang] = engine
            return engine

    def warmup(self, langs: list[str] | None = None) -> None:
        """Pre-initialize engines so model downloads happen up front."""
        for lang in langs or SUPPORTED_LANGS:
            self.get_engine(lang)


# --------------------------------------------------------------------------- #
# Image input parsing
# --------------------------------------------------------------------------- #
def _bytes_to_ndarray(data: bytes) -> np.ndarray:
    from PIL import Image

    logger.info("[decode] decoding %d image bytes with Pillow", len(data))
    try:
        img = Image.open(io.BytesIO(data))
        arr = np.array(img.convert("RGB"))
        logger.info("[decode] decoded to ndarray %s", arr.shape)
        return arr
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[decode] Pillow failed to decode %d bytes: %s", len(data), exc, exc_info=True)
        raise OCRLoaderError("Could not decode image bytes into an array.") from exc


def _decode_to_bytes(raw: str) -> bytes:
    """Decode a plain or data-URI base64 string into raw bytes."""
    is_data_uri = raw.strip().lower().startswith("data:")
    logger.info("[base64] decoding %s (%d chars)", "data-URI" if is_data_uri else "raw base64", len(raw))
    match = _DATA_URI_RE.match(raw.strip())
    payload = match.group(1) if match else raw.strip()
    try:
        data = base64.b64decode(payload)
        logger.info("[base64] decoded %d bytes", len(data))
        return data
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[base64] decode failed: %s", exc, exc_info=True)
        raise OCRLoaderError("Image source looked like base64 but could not be decoded.") from exc


def _download_bytes(url: str) -> bytes:
    """Download a URL's raw bytes (works for both images and PDFs)."""
    logger.info("[download] fetching %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ocr-mcp/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - user URL
            data = resp.read()
        logger.info("[download] got %d bytes from %s", len(data), url)
        return data
    except Exception as exc:
        logger.error("[download] failed to fetch %s: %s", url, exc, exc_info=True)
        raise OCRLoaderError(f"Failed to download image from URL: {url}") from exc


# PDF inputs are materialized to temp files because PaddleOCR reads PDFs by path
# (it cannot consume raw PDF bytes / ndarrays). Temp paths are tracked here so
# callers can clean them up after inference via release_input().
_TEMP_PATHS: set[str] = set()

# Magic bytes that identify a PDF stream (all PDF versions start with "%PDF").
_PDF_MAGIC = b"%PDF"


def _is_pdf_bytes(data: bytes) -> bool:
    return data[: len(_PDF_MAGIC)] == _PDF_MAGIC


def _materialize_pdf(data: bytes) -> str:
    """Write PDF bytes to a temp file and return its path for native PaddleOCR use."""
    logger.info("[pdf] writing %d PDF bytes to temp file", len(data))
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="ocr_mcp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception as exc:
        logger.error("[pdf] failed to write temp PDF: %s", exc, exc_info=True)
        try:
            os.remove(path)
        except OSError:
            pass
        raise OCRLoaderError("Could not write PDF to a temporary file.") from exc
    logger.info("[pdf] temp PDF ready: %s", path)
    _TEMP_PATHS.add(path)
    return path


def release_input(image_input: Any) -> None:
    """Remove a temp PDF created by ``load_image``; no-op for real paths / arrays."""
    if isinstance(image_input, str) and image_input in _TEMP_PATHS:
        _TEMP_PATHS.discard(image_input)
        try:
            os.remove(image_input)
        except OSError:
            pass


# Office document extensions convertible to PDF via LibreOffice headless.
# doc/docx/ppt/pptx/xls/xlsx (and ODF equivalents) have no text layer PaddleOCR
# can read directly, so they are rasterized to PDF first, then flow through the
# existing native-PDF path (page-by-page inference).
_OFFICE_EXTS = {
    ".doc", ".docx", ".rtf", ".odt",
    ".ppt", ".pptx", ".odp",
    ".xls", ".xlsx", ".ods",
}


def _office_to_pdf(src_path: str) -> str:
    """Convert an office document to a temp PDF via LibreOffice headless.

    Returns the path of the produced PDF (tracked in ``_TEMP_PATHS`` for cleanup
    via release_input()). Uses a unique ``-env:UserInstallation`` profile per call
    so concurrent soffice processes don't contend on a shared profile lock.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        logger.error("[office] soffice not found on PATH")
        raise OCRLoaderError(
            "LibreOffice (soffice) is not installed; cannot convert office "
            "documents. Install it in the server environment to enable "
            "doc/ppt/xls support."
        )
    logger.info("[office] found soffice: %s", soffice)
    src = Path(src_path)
    outdir = tempfile.mkdtemp(prefix="lo_out_")
    profile = tempfile.mkdtemp(prefix="lo_prof_")
    logger.info("[office] converting '%s' (%s) -> PDF (outdir=%s)", src.name, src.suffix, outdir)
    import time as _time
    _t0 = _time.monotonic()
    try:
        subprocess.run(
            [
                soffice,
                "--headless",
                "--norestore",
                "--nofirststartwizard",
                f"-env:UserInstallation=file:///{profile}",
                "--convert-to", "pdf",
                "--outdir", outdir,
                str(src),
            ],
            timeout=120,
            check=True,
            capture_output=True,
        )
        logger.info("[office] soffice exited in %.1fs", _time.monotonic() - _t0)
        pdf = Path(outdir) / f"{src.stem}.pdf"
        if not pdf.exists():
            logger.error("[office] no output PDF produced for '%s'", src.name)
            raise OCRLoaderError(
                f"LibreOffice produced no PDF for '{src.name}'. The file may be "
                "corrupted or in an unsupported format."
            )
        # Move the PDF out of the temp outdir into its own tracked temp file so
        # the outdir can be cleaned up immediately.
        fd, final = tempfile.mkstemp(suffix=".pdf", prefix="ocr_mcp_")
        os.close(fd)
        shutil.move(str(pdf), final)
        _TEMP_PATHS.add(final)
        logger.info("[office] '%s' -> temp PDF %s (%.1fs)", src.name, final, _time.monotonic() - _t0)
        return final
    except subprocess.TimeoutExpired as exc:
        logger.error("[office] conversion timed out for '%s' (>120s)", src.name)
        raise OCRLoaderError("LibreOffice conversion timed out (>120s).") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.error("[office] soffice failed (exit %s) for '%s': %s", exc.returncode, src.name, detail)
        raise OCRLoaderError(
            f"LibreOffice failed to convert '{src.name}'"
            + (f": {detail}" if detail else "")
       ) from exc
    finally:
        shutil.rmtree(outdir, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)


def _office_to_pdf_bytes(data: bytes, ext: str) -> str:
    """Convert raw office-document bytes to a temp PDF.

    Writes the bytes to a temp file (so LibreOffice gets a real path + extension
    to infer the format from), converts it, then removes the temp source. The
    produced PDF is tracked in ``_TEMP_PATHS`` for release_input() cleanup.
    Office formats can't be sniffed from magic bytes (all are ZIP containers),
    so the caller must supply the extension (e.g. taken from the URL path).
    """
    logger.info("[office] %d office bytes (ext=%s) -> temp file for conversion", len(data), ext)
    fd, tmp = tempfile.mkstemp(suffix=ext, prefix="ocr_mcp_dl_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return _office_to_pdf(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _looks_like_base64(src: str) -> bool:
    compact = re.sub(r"\s+", "", src)
    # A real encoded image is far longer than this threshold; this avoids
    # misclassifying short non-path tokens as base64.
    if len(compact) < 48:
        return False
    return bool(_BASE64_ALPHABET_RE.match(compact))


def load_image(source: str) -> Any:
    """Resolve an image/PDF/office-doc source to something PaddleOCR accepts.

    Accepts a local file path (image, PDF, or office doc), an http(s) URL, a data
    URI, or a raw base64 string. Images become an ``np.ndarray`` (RGB) or are
    returned as a path for PaddleOCR to read directly; PDFs are always returned
    as a file path so PaddleOCR can read them natively (page by page). Office
    documents (doc/docx/ppt/pptx/xls/xlsx/...) are first converted to PDF via
    LibreOffice, then treated as PDFs. Temp files created for URL/base64/office
    sources are tracked for cleanup via release_input().
    """
    if not isinstance(source, str) or not source.strip():
        raise OCRLoaderError("Image source is empty.")

    logger.info("[load] resolving input source (%d chars)", len(source))
    src = source.strip()

    # 1) Existing local file -> office docs are converted to PDF first; images
    #    and PDFs are passed through for PaddleOCR to read directly (it reads
    #    PDFs natively via PyMuPDF, one result object per page).
    candidate = Path(src)
    try:
        if candidate.exists() and candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix in _OFFICE_EXTS:
                logger.info("[load] local office doc (%s) -> convert to PDF", suffix)
                return _office_to_pdf(str(candidate))
            logger.info("[load] local %s file -> pass-through path", suffix or "(no ext)")
            return str(candidate.resolve())
    except (OSError, ValueError):
        # Some strings (e.g. long base64 blobs) are not valid path encodings on
        # every platform; fall through to the URL/base64 branches below.
        pass

    lowered = src.lower()
    # 2) http(s) URL -> raw bytes; dispatch by content (PDF) or URL extension
    #    (office docs are ZIP containers, so they can't be sniffed from bytes).
    if lowered.startswith("http://") or lowered.startswith("https://"):
        logger.info("[load] source is http(s) URL")
        data = _download_bytes(src)
        if _is_pdf_bytes(data):
            logger.info("[load] downloaded content is PDF -> materialize")
            return _materialize_pdf(data)
        url_ext = Path(urlparse(src).path).suffix.lower()
        if url_ext in _OFFICE_EXTS:
            logger.info("[load] downloaded content is office doc (%s) -> convert", url_ext)
            return _office_to_pdf_bytes(data, url_ext)
        logger.info("[load] downloaded content is image -> decode")
        return _bytes_to_ndarray(data)
    # 3) data-URI / base64 -> raw bytes; same PDF/image dispatch.
    if lowered.startswith("data:") or _looks_like_base64(src):
        logger.info("[load] source is %s", "data-URI" if lowered.startswith("data:") else "base64")
        data = _decode_to_bytes(src)
        if _is_pdf_bytes(data):
            logger.info("[load] decoded content is PDF -> materialize")
            return _materialize_pdf(data)
        logger.info("[load] decoded content is image -> decode")
        return _bytes_to_ndarray(data)

    logger.warning("[load] unrecognized input source type")
    raise OCRLoaderError(
        "Image source is not a recognizable local path, http(s) URL, "
        "or base64/data-URI string."
    )


# --------------------------------------------------------------------------- #
# OCR execution + result normalization
# --------------------------------------------------------------------------- #
def _extract_result_dict(res: Any) -> dict[str, Any]:
    """Pull the flat dict of detection/recognition fields out of a result.

    PaddleOCR 3.x wraps fields under a nested "res" key inside ``result.json``:
    ``{"res": {"rec_texts": [...], "dt_polys": [...], ...}}``. Older or
    alternative result objects may expose the fields directly, so we handle both.
    """
    data: Any = None
    json_attr = getattr(res, "json", None)
    if callable(json_attr):
        try:
            data = json_attr()
        except Exception:  # pragma: no cover - defensive
            data = None
    elif json_attr is not None:
        data = json_attr

    if isinstance(data, dict):
        # Unwrap the nested envelope if present.
        for key in ("res", "result", "json"):
            inner = data.get(key)
            if isinstance(inner, dict) and (
                "rec_texts" in inner or "dt_polys" in inner
            ):
                data = inner
                break
        if isinstance(data, dict):
            return data
    return {}


def _normalize_result(res: Any) -> list[dict[str, Any]]:
    """Turn one PaddleOCR result object into a list of per-line dicts."""
    data = _extract_result_dict(res)

    texts = (
        data.get("rec_texts")
        or getattr(res, "rec_texts", None)
        or []
    )
    scores = (
        data.get("rec_scores")
        or getattr(res, "rec_scores", None)
        or []
    )
    polys = (
        data.get("dt_polys")
        or data.get("rec_polys")
        or getattr(res, "dt_polys", None)
        or []
    )

    lines: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        confidence = float(scores[i]) if i < len(scores) else None
        box = None
        if i < len(polys):
            try:
                box = [[float(pt[0]), float(pt[1])] for pt in polys[i]]
            except (IndexError, TypeError, ValueError):  # pragma: no cover - defensive
                box = None
        lines.append({"text": str(text), "confidence": confidence, "box": box})
    return lines


def run_ocr(engine: Any, image_input: Any) -> list[dict[str, Any]]:
    """Run OCR for an image/PDF and return normalized per-line results.

    For multi-page PDFs, PaddleOCR returns one result object per page; each
    emitted line is tagged with its 0-based ``page`` index, and bounding boxes
    are expressed in each page's own coordinate space.
    """
    image_input = _resize_for_ocr(image_input)
    logger.info("[ocr] calling engine.predict()")
    import time as _time
    _t0 = _time.monotonic()
    results = engine.predict(image_input)
    logger.info("[ocr] engine.predict returned in %.1fs", _time.monotonic() - _t0)
    items = results if isinstance(results, list) else [results]
    logger.info("[ocr] %d page(s) to normalize", len(items))
    lines: list[dict[str, Any]] = []
    for page_idx, res in enumerate(items):
        page_lines = _normalize_result(res)
        logger.info("[ocr] page %d: %d lines", page_idx, len(page_lines))
        for line in page_lines:
            line["page"] = page_idx
            lines.append(line)
    logger.info("[ocr] done, %d total lines", len(lines))
    return lines


# Cap the longest image edge before OCR so PaddleOCR doesn't OOM on huge
# photos (a multi-thousand-pixel image can trigger ~10GB+ allocations in the
# detection operator). PaddleOCR resizes internally anyway, so this only
# bounds peak memory without hurting accuracy for documents. Override with
# OCR_MAX_IMAGE_SIDE.
def _resize_for_ocr(image_input: Any) -> Any:
    """Downscale the image so its longest edge <= OCR_MAX_IMAGE_SIDE pixels."""
    from PIL import Image

    max_side = int(os.getenv("OCR_MAX_IMAGE_SIDE", "2880"))
    if max_side <= 0:
        return image_input

    if isinstance(image_input, str):
        # PDFs are handed to PaddleOCR natively (read page by page with
        # PyMuPDF inside the engine), so never try to open them with Pillow.
        if image_input.lower().endswith(".pdf"):
            logger.info("[resize] PDF input, skipping (max_side=%d)", max_side)
            return image_input
        try:
            img = Image.open(image_input)
        except Exception:
            logger.warning("[resize] could not open %s with Pillow, skipping", image_input)
            return image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        return image_input

    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        logger.info("[resize] downscaling %s -> %s (max_side=%d)", img.size, new_size, max_side)
        img = img.resize(new_size)
    else:
        logger.info("[resize] no resize needed (%s <= %d)", img.size, max_side)

    return np.array(img.convert("RGB"))

"""PaddleOCR engine wrapper.

Keeps one lazily-initialized, cached PaddleOCR (PP-OCRv5) instance per language,
parses heterogeneous image inputs (local path / http(s) URL / base64 data URI),
and normalizes OCR results into plain dicts.
"""

from __future__ import annotations

import base64
import io
import os
import platform
import re
import sysconfig
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

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

            # Pin to PP-OCRv5. paddleocr >=3.7 defaults to PP-OCRv6 when `lang`
            # is given but `ocr_version` is omitted; PP-OCRv6 segfaults on CPU
            # under paddlepaddle 3.x even with mkldnn disabled. PP-OCRv5 is the
            # version this project was built and verified against. Override with
            # OCR_VERSION (e.g. PP-OCRv4 / PP-OCRv6) if you know your stack works.
            ocr_version = os.getenv("OCR_VERSION", "PP-OCRv5")
            # Select the inference engine. On aarch64 the paddle native engine
            # segfaults at inference time, so we route through onnxruntime.
            inference_engine = _select_inference_engine()
            init_kwargs: dict[str, Any] = {
                "lang": lang,
                "ocr_version": ocr_version,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "device": "cpu",
            }
            # `enable_mkldnn` only applies to the native paddle engine; it is
            # meaningless (and may error) under onnxruntime.
            if inference_engine == "onnxruntime":
                init_kwargs["engine"] = "onnxruntime"
            else:
                init_kwargs["enable_mkldnn"] = False
            engine = PaddleOCR(**init_kwargs)
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

    try:
        img = Image.open(io.BytesIO(data))
        return np.array(img.convert("RGB"))
    except Exception as exc:  # pragma: no cover - defensive
        raise OCRLoaderError("Could not decode image bytes into an array.") from exc


def _decode_base64_image(raw: str) -> np.ndarray:
    """Decode a plain or data-URI base64 string into an RGB numpy array."""
    match = _DATA_URI_RE.match(raw.strip())
    payload = match.group(1) if match else raw.strip()
    try:
        data = base64.b64decode(payload)
    except Exception as exc:  # pragma: no cover - defensive
        raise OCRLoaderError("Image source looked like base64 but could not be decoded.") from exc
    return _bytes_to_ndarray(data)


def _download(url: str) -> np.ndarray:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ocr-mcp/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - user URL
            data = resp.read()
    except Exception as exc:
        raise OCRLoaderError(f"Failed to download image from URL: {url}") from exc
    return _bytes_to_ndarray(data)


def _looks_like_base64(src: str) -> bool:
    compact = re.sub(r"\s+", "", src)
    # A real encoded image is far longer than this threshold; this avoids
    # misclassifying short non-path tokens as base64.
    if len(compact) < 48:
        return False
    return bool(_BASE64_ALPHABET_RE.match(compact))


def load_image(source: str) -> Any:
    """Resolve an image source to something ``PaddleOCR.predict`` accepts.

    Returns either an absolute file path (str) or an ``np.ndarray`` (RGB).
    """
    if not isinstance(source, str) or not source.strip():
        raise OCRLoaderError("Image source is empty.")

    src = source.strip()

    # 1) Existing local file -> let PaddleOCR read the path directly.
    candidate = Path(src)
    try:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    except (OSError, ValueError):
        # Some strings (e.g. long base64 blobs) are not valid path encodings on
        # every platform; fall through to the URL/base64 branches below.
        pass

    lowered = src.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return _download(src)
    if lowered.startswith("data:") or _looks_like_base64(src):
        return _decode_base64_image(src)

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
    """Run OCR for a single image and return normalized per-line results."""
    image_input = _resize_for_ocr(image_input)
    results = engine.predict(image_input)
    lines: list[dict[str, Any]] = []
    for res in results:
        lines.extend(_normalize_result(res))
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
        try:
            img = Image.open(image_input)
        except Exception:
            return image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        return image_input

    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)

    return np.array(img.convert("RGB"))

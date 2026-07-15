"""Pre-download and initialize PaddleOCR models.

Run as ``python -m ocr_mcp.warmup`` so that model weights are cached ahead of
time (e.g. during a Docker build), letting the image run fully offline.
"""

from __future__ import annotations

import sys

from .ocr_engine import SUPPORTED_LANGS, OCREngine


def main() -> int:
    engine = OCREngine()
    for lang in SUPPORTED_LANGS:
        print(f"[warmup] initializing PaddleOCR engine: {lang}", flush=True)
        engine.get_engine(lang)
    print("[warmup] all models ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

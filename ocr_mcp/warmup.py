"""Pre-download and initialize PaddleOCR models.

Run as ``python -m ocr_mcp.warmup`` so that model weights are cached ahead of
time (e.g. during a Docker build), letting the image run fully offline.
"""

from __future__ import annotations

import sys

from .ocr_engine import SUPPORTED_LANGS, _create_paddleocr

# Model variants to pre-download. The "server" set is PaddleOCR's default
# (high accuracy, slow); the "mobile" set is ~3-5x faster for clean documents
# like shipping labels. Pre-downloading both lets the operator switch at
# runtime via OCR_DET_MODEL / OCR_REC_MODEL without needing network access.
MODEL_VARIANTS = [
    ("server", "PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    ("mobile", "PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
]


def main() -> int:
    for label, det_model, rec_model in MODEL_VARIANTS:
        for lang in SUPPORTED_LANGS:
            print(
                f"[warmup] {label}: {lang} (det={det_model}, rec={rec_model})",
                flush=True,
            )
            _create_paddleocr(lang, det_model, rec_model)
    print("[warmup] all models ready (server + mobile)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

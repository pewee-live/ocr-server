"""Pre-download and initialize PaddleOCR models.

Run as ``python -m ocr_mcp.warmup`` so that model weights are cached ahead of
time (e.g. during a Docker build), letting the image run fully offline.
"""

from __future__ import annotations

import sys

from .ocr_engine import SUPPORTED_LANGS, _create_paddleocr
from .layout_engine import _create_pp_structure

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
    # Pre-download the PP-StructureV3 model set. _create_pp_structure itself
    # decides which sub-pipelines to enable based on the inference engine:
    # under onnxruntime (aarch64 default, or OCR_ENGINE=onnxruntime) the
    # formula/chart/seal models are skipped because they have no onnx package
    # and would crash the build; under the native paddle engine (x86_64) they
    # are included so invoice seal/chart/formula recognition works. Without
    # this warmup the first recognize_layout call would hit the network for
    # several multi-hundred-MB downloads (and the formula model is slow enough
    # to cause runtime timeouts). One language is enough: layout/table/formula
    # models are language-agnostic except for the OCR sub-model already cached.
    print("[warmup] PP-Structure layout/table (+formula/chart/seal if paddle)", flush=True)
    _create_pp_structure("ch")
    print("[warmup] all models ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

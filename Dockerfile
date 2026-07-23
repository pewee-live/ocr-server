# syntax=docker/dockerfile:1.7
# PaddleOCR MCP server image.
# Runs a streamable-http MCP server (PP-OCRv5: zh / en / ja / ko) with models
# pre-downloaded at build time so the container can run fully offline.

FROM python:3.11-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    OCR_ENGINE=onnxruntime \
    OCR_DET_MODEL=PP-OCRv5_mobile_det \
    OCR_REC_MODEL=PP-OCRv5_mobile_rec \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Runtime libraries required by OpenCV / paddlepaddle (libgomp, GL, etc.).
# LibreOffice (headless) converts office docs (doc/docx/ppt/pptx/xls/xlsx/...)
# to PDF so they can flow through the existing PDF->OCR pipeline. fonts-noto-cjk
# is REQUIRED: without CJK glyphs LibreOffice renders Chinese as tofu boxes and
# the resulting PDF OCRs as garbage.
# An apt cache mount keeps downloaded .deb packages across builds so only the
# first build downloads the ~300-500MB LibreOffice stack. Requires BuildKit.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgeos-dev \
        ca-certificates \
        libreoffice-core \
        libreoffice-writer \
        libreoffice-impress \
        libreoffice-calc \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer-cache strategy ────────────────────────────────────────────────────
# Build steps are ordered so the slow, stable steps (apt, pip deps, model
# download) are cached and do NOT re-run when application code (ocr_mcp/*.py)
# changes. Three BuildKit cache mounts persist downloads on the host across
# builds: apt debs, pip wheels, and PaddleX model weights. Combined with layer
# ordering, a code-only change rebuilds only the tiny package-install layer.
# ────────────────────────────────────────────────────────────────────────────

# 1) Python dependencies. Keys off requirements.txt only; stays cached across
#    code changes. pip cache mount reuses wheels on the host.
COPY requirements.txt ./
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# 2) Create the runtime user up front so model prefetch can chown its output.
RUN useradd --create-home --uid 1000 app

# 3) Pre-download OCR/layout models. DELIBERATELY placed BEFORE the application
#    code copy: model download is the slowest step (multi-GB) and depends only
#    on the engine version (requirements.txt) + the ENV knobs below, NOT on app
#    code. A standalone prefetch script (inline heredoc) is used instead of
#    `python -m ocr_mcp.warmup` so this layer does NOT import ocr_mcp and thus
#    is not invalidated by code edits. The cache mount holds the weights across
#    builds; they are then copied into the image layer for offline runtime use
#    (cache mounts themselves do not ship with the image).
#    The heredoc mirrors ocr_mcp/warmup.py + the engine factories in
#    ocr_engine.py / layout_engine.py; it reads the same OCR_* env vars so it
#    stays aligned without importing them.
RUN --mount=type=cache,id=paddlex-models,target=/mnt/pdx,uid=1000,gid=1000 <<'EOF'
set -e
PADDLEX_HOME=/mnt/pdx python - <<'PY'
import os
from paddleocr import PaddleOCR
try:
    from paddleocr import PPStructureV3
except ImportError:  # legacy paddleocr 2.x
    from paddleocr import PPStructure as PPStructureV3

ver = os.getenv("OCR_VERSION", "PP-OCRv5")
onnx = os.getenv("OCR_ENGINE", "").strip().lower() == "onnxruntime"


def base_kw():
    kw = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=os.getenv("OCR_DEVICE", "cpu"),
    )
    if onnx:
        kw["engine"] = "onnxruntime"
    else:
        kw["enable_mkldnn"] = False
    return kw


# Pre-download both server (accurate) and mobile (fast) det/rec model pairs for
# every supported language, so OCR_DET_MODEL/OCR_REC_MODEL can switch at runtime
# without network access.
for det, rec in [
    ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
]:
    for lang in ["ch", "en", "japan", "korean"]:
        print(f"[warmup] {lang} det={det} rec={rec}", flush=True)
        PaddleOCR(
            lang=lang,
            ocr_version=ver,
            text_detection_model_name=det,
            text_recognition_model_name=rec,
            **base_kw(),
        )

# PP-Structure layout/table models. Under onnxruntime the formula/chart/seal
# sub-pipelines have no onnx package and would crash load, so disable them
# (matching layout_engine._create_pp_structure) unless OCR_LAYOUT_FULL=1.
lay = base_kw()
if onnx and os.getenv("OCR_LAYOUT_FULL", "0").strip() not in ("1", "true", "yes"):
    lay["use_formula_recognition"] = False
    lay["use_chart_recognition"] = False
    lay["use_seal_recognition"] = False
print("[warmup] PP-Structure layout/table", flush=True)
PPStructureV3(lang="ch", **lay)
print("[warmup] all models ready", flush=True)
PY
mkdir -p /home/app/.paddlex
cp -a /mnt/pdx/. /home/app/.paddlex/
chown -R app:app /home/app/.paddlex
EOF

# 4) Application code. This layer (and only this) rebuilds on code changes; it
#    is fast (no deps, no models). `--no-deps` because deps are already installed
#    in step 1.
COPY pyproject.toml ./
COPY ocr_mcp ./ocr_mcp
RUN pip install --no-deps . && chown -R app:app /app

USER app

EXPOSE 8000

# Healthcheck: verify the MCP port is accepting TCP connections. The streamable
# HTTP endpoint rejects non-MCP payloads with 400, so a raw TCP check is the
# most reliable liveness signal.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(5); \
        sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"

CMD ["ocr-mcp"]

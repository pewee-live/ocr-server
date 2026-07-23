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

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
# The cache mount persists pip's HTTP/wheel cache on the host across builds:
# the first build downloads everything, subsequent builds reuse the local cache
# and only fetch newly added packages. Requires BuildKit, already enabled by
# the `# syntax=docker/dockerfile:1.7` directive on line 1.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# Install the package itself.
COPY pyproject.toml ./
COPY ocr_mcp ./ocr_mcp
RUN pip install --no-deps .

# Create a non-root user for runtime.
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

# Pre-download OCR models for all supported languages. A cache mount holds the
# multi-GB model weights across builds so the download happens only once; when
# the cache is warm, warmup just re-verifies the files. The weights are then
# copied from the cache into the image layer so the final image is self-contained
# for offline use (cache mounts are NOT persisted to the image layer themselves).
# PADDLEX_HOME is set only for warmup so models download into the mount; at
# runtime it is unset and PaddleX reads from the default ~/.paddlex location.
RUN --mount=type=cache,id=paddlex-models,target=/mnt/pdx,uid=1000,gid=1000 \
    PADDLEX_HOME=/mnt/pdx python -m ocr_mcp.warmup && \
    mkdir -p /home/app/.paddlex && \
    cp -a /mnt/pdx/. /home/app/.paddlex/

EXPOSE 8000

# Healthcheck: verify the MCP port is accepting TCP connections. The streamable
# HTTP endpoint rejects non-MCP payloads with 400, so a raw TCP check is the
# most reliable liveness signal.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(5); \
        sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"

CMD ["ocr-mcp"]

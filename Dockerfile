# syntax=docker/dockerfile:1.7
# PaddleOCR MCP server image.
# Runs a streamable-http MCP server (PP-OCRv5: zh / en / ja / ko) with models
# pre-downloaded at build time so the container can run fully offline.

FROM python:3.11-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    OCR_ENGINE=onnxruntime \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Runtime libraries required by OpenCV / paddlepaddle (libgomp, GL, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt

# Install the package itself.
COPY pyproject.toml ./
COPY ocr_mcp ./ocr_mcp
RUN pip install --no-deps .

# Create a non-root user for runtime.
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

# Pre-download OCR models for all supported languages so the image is
# self-contained. This is the slowest build step.
RUN python -m ocr_mcp.warmup

EXPOSE 8000

# Healthcheck: verify the MCP port is accepting TCP connections. The streamable
# HTTP endpoint rejects non-MCP payloads with 400, so a raw TCP check is the
# most reliable liveness signal.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(5); \
        sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"

CMD ["ocr-mcp"]

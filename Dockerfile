# syntax=docker/dockerfile:1
# ---- Swarm-SAR reproducible environment (CPU by default) ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# Core deps first (cached layer); RL/viz extras are optional at build time.
# Install only the core package deps, not the full requirements.txt (which
# includes heavy optional dependencies like torch, ray, etc.)
COPY pyproject.toml setup.cfg* ./
COPY src/ src/
RUN pip install --upgrade pip && pip install -e . 2>/dev/null || true
COPY requirements.txt requirements-dev.txt ./

COPY . .
RUN pip install -e ".[rl,viz]" || pip install -e .

# Default: run the self-contained demo and export figures to results/figures
CMD ["python", "scripts/run_demo.py", "--episodes", "1", "--figures"]

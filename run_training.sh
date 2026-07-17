#!/usr/bin/env bash
# Swarm-SAR - one-click training on a Python 3.11 CUDA setup.
set -euo pipefail
cd "$(dirname "$0")"

ALLOW_CPU=0
RECREATE=0
for arg in "$@"; do
  [[ "$arg" == "--allow-cpu" ]] && ALLOW_CPU=1
  [[ "$arg" == "--recreate" ]] && RECREATE=1
done

echo "=== Swarm-SAR GPU training ==="
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it or set PYTHON_BIN=/path/to/python3.11"
  exit 1
fi
"$PYTHON_BIN" --version

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif [[ "$ALLOW_CPU" != "1" ]]; then
  echo "nvidia-smi was not found. Re-run with --allow-cpu to permit CPU training."
  exit 1
else
  echo "No NVIDIA GPU detected - CPU training explicitly allowed."
fi

echo "[1/5] Creating Python 3.11 virtual environment"
if [[ "$RECREATE" == "1" && -d .venv ]]; then
  echo "Removing the existing .venv as requested..."
  rm -rf .venv
fi
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
if ! python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
then
  echo ".venv is not Python 3.11. Re-run with --recreate to rebuild it."
  exit 1
fi

echo "[2/5] Upgrading pip"
python -m pip install --upgrade pip

echo "[3/5] Installing CUDA PyTorch and project dependencies"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e ".[rl,viz]"

echo "[4/5] Verifying GPU runtime"
if [[ "$ALLOW_CPU" == "1" ]]; then
  python scripts/verify_gpu.py
else
  python scripts/verify_gpu.py --require-cuda
fi

echo "[5/5] Smoke test, then laptop-sized training"
python scripts/train_and_report.py --smoke
python scripts/train_and_report.py --config configs/training/mappo_easy.yaml \
  --archs transformer_gnn --seeds 0 --timesteps 300000

echo "DONE. Figures: results/trained/figures ; report: results/trained/report.json"
echo "TensorBoard: tensorboard --logdir results/trained"

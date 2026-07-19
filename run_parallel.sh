#!/bin/bash
# Launch the science campaign's seeds in parallel (one process per seed).
# Assumes the environment is already set up (see run_training.sh / TRAIN_ON_YOUR_GPU.md);
# this script deliberately does NOT trigger the full setup-and-train launcher.
set -e

ARCH="${ARCH:-transformer_gnn}"
TIMESTEPS="${TIMESTEPS:-3000000}"
SEEDS=(${SEEDS:-0 1 2 3 4})

source .venv/bin/activate 2>/dev/null || true

echo "=== Parallel training: arch=$ARCH timesteps=$TIMESTEPS seeds=${SEEDS[*]} ==="
PIDS=()
for s in "${SEEDS[@]}"; do
    python scripts/train_and_report.py --archs "$ARCH" --seeds "$s" \
        --timesteps "$TIMESTEPS" > "seed${s}.log" 2>&1 &
    PIDS+=($!)
    echo "seed $s -> PID ${PIDS[-1]} (seed${s}.log)"
done

echo "Waiting for ${#PIDS[@]} runs..."
wait "${PIDS[@]}"
echo "=== All runs completed — see results/trained ==="

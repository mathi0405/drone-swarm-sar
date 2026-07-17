#!/bin/bash
set -e

echo "=== Setting up environment ==="
# Run the setup script once sequentially so pip doesn't crash on race conditions
./run_training.sh --timesteps 1

echo "=== Starting Parallel Training (3 Seeds) ==="
source .venv/bin/activate

# Launch seeds 0, 1, and 2 in the background
python scripts/train_and_report.py --config configs/training/mappo_easy.yaml --archs transformer_gnn --seeds 0 --timesteps 3000000 > seed0.log 2>&1 &
PID0=$!

python scripts/train_and_report.py --config configs/training/mappo_easy.yaml --archs transformer_gnn --seeds 1 --timesteps 3000000 > seed1.log 2>&1 &
PID1=$!

python scripts/train_and_report.py --config configs/training/mappo_easy.yaml --archs transformer_gnn --seeds 2 --timesteps 3000000 > seed2.log 2>&1 &
PID2=$!

echo "All 3 seeds launched in background!"
echo "Seed 0 PID: $PID0 -> logging to seed0.log"
echo "Seed 1 PID: $PID1 -> logging to seed1.log"
echo "Seed 2 PID: $PID2 -> logging to seed2.log"

echo "Waiting for all runs to finish..."
wait $PID0 $PID1 $PID2

echo -e "\e[32m=== All 3 parallel runs completed! ===\e[0m"
echo "Check results/trained for the outputs."

# Training Swarm-SAR on your own GPU

The neural policies (MAPPO with MLP / GNN / Transformer / Transformer+GNN) need
PyTorch + a CUDA GPU. This repository ships everything wired up so you get
**measured** results with a single command — the run trains the policies,
evaluates them, and regenerates the training/comparison figures from the real
logs (replacing the illustrative placeholders).

## One click

| OS | Do this |
|----|---------|
| **Windows** | double-click `run_training.bat` (or `run_training.ps1`) |
| **Linux / macOS / WSL** | `bash run_training.sh` |

Each launcher: creates a virtualenv, installs the correct CUDA build of PyTorch,
installs Swarm-SAR + RL extras, runs a fast smoke test, then the full run.

Python **3.11** is required. If an older `.venv` was created with Python 3.14
or another version, rebuild it explicitly (this deletes only `.venv`):

```powershell
.\run_training.ps1 -Recreate
# or
.\run_training.bat --recreate
```

## Manual (equivalent)

```bash
# from the repo root (Windows)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 build
pip install -r requirements.txt && pip install -e ".[rl]"

# validate the whole pipeline in ~1-2 min (tiny run)
python scripts/train_and_report.py --smoke

# the real thing (edit to taste)
python scripts/train_and_report.py \
    --archs mlp gnn transformer transformer_gnn \
    --seeds 0 1 2 \
    --timesteps 1000000
```

## What you get (all measured, no placeholders)

- `results/trained/<arch>_s<seed>/` — per-run TensorBoard logs, `scalars.csv`, config, `checkpoints/final.pt`
- `results/trained/figures/fig08_reward_curves_REAL.png` — learning curves from real logs
- `results/trained/figures/fig20_algo_comparison_REAL.png` — architecture comparison from real eval metrics
- `results/trained/figures/fig11_success_rate_REAL.png`
- `results/trained/report.json` — per-architecture SIS with **IQM + 95% bootstrap CIs**
- `tensorboard --logdir results/trained` to watch training live

## Choosing scale

| GPU | Suggested |
|-----|-----------|
| Laptop (≤6 GB) | `--archs transformer_gnn --seeds 0 --timesteps 300000` |
| RTX 3080/4080 (10–16 GB) | run `--archs mlp gnn transformer transformer_gnn --seeds 0 1 2 --timesteps 1000000` |
| A100 / multi-GPU | add more seeds; use the RLlib path (`scripts/train.py --rllib`) for 10-drone / large maps |

## Notes

- `torch-geometric` is optional; if its wheels don't match your CUDA/torch, the GNN
  automatically falls back to a built-in attention layer — training still runs.
- The run is fully seeded (cuDNN determinism enabled) for reproducibility.
- To fold the measured figures into the paper, point `paper/main.tex`'s
  `\includegraphics` for Figs 8/11/20 at the `*_REAL.png` files.

## Known scope of the built-in trainer

The self-contained `training/mappo.py` trains per-agent encoders (the Transformer
does intra-agent set reasoning). True *inter-agent* GNN message passing over the
live communication graph during training is exercised in the RLlib path and is the
extension point addressed by the learned-communication module
(`policies/comm_head.py`). All four architectures train and evaluate; expect the
largest gains from GNN/Transformer once graph batching is enabled in your run.

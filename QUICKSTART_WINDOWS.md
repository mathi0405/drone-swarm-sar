# Quickstart — Windows + NVIDIA GPU

Every command below is copy‑paste ready. Run them in **PowerShell** from this folder.

## 0. Prerequisites (one time)

- **Python 3.11** — NOT 3.13/3.14 (PyTorch and Ray have no wheels for those yet).
  Get it from https://www.python.org/downloads/release/python-3119/ and tick
  "Add python.exe to PATH". Verify:
  ```powershell
  py -3.11 --version      # -> Python 3.11.x
  ```
- **NVIDIA driver** installed. Verify:
  ```powershell
  nvidia-smi
  ```

## 1. Set up the environment (one time, ~10 min)

```powershell
cd "C:\path\to\drone-swarm-sar"
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version                       # MUST say 3.11.x
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e ".[rl]"
```

## 2. Validate the whole pipeline (~2 min)

```powershell
python scripts\train_and_report.py --smoke
```
Success = it ends with a line like `done ... report.json`.

## 3. Train (easy curriculum — best for a laptop GPU, ~2–2.5 hr for 1M steps)

```powershell
python scripts\train_and_report.py --config configs\training\mappo_easy.yaml --archs transformer_gnn --seeds 0 --timesteps 1000000
```
Faster first look (~1.2 hr): change `1000000` to `500000`.

## 4. Watch it learn (second PowerShell, venv activated)

```powershell
tensorboard --logdir results\trained
```
Open the printed URL and watch **`ep_return`** — rising = learning.

## 5. Benchmark the trained policy vs classical baselines

```powershell
python scripts\run_benchmark.py --n-maps 8 --checkpoint results\trained\transformer_gnn_s0_<timestamp>\checkpoints\final.pt
```
(Use the newest folder name under `results\trained\`.)

## Where results land

| Path | What |
|------|------|
| `results\trained\figures\*_REAL.png` | measured learning + comparison figures |
| `results\trained\report.json` | SIS with IQM + 95% bootstrap CIs |
| `results\trained\<run>\checkpoints\final.pt` | trained model |

## No‑GPU things you can run anytime

```powershell
python scripts\run_demo.py --episodes 3 --figures          # heuristic demo + figures
python scripts\generate_figures.py                          # all 20 publication figures
python scripts\run_experiments.py --config configs\experiments\ablation_comm.yaml
python scripts\comm_study.py                                # communication degradation curve
streamlit run src\swarm_sar\dashboard\app.py                # live dashboard
```

## Notes

- `run_training.bat` is a one‑click option, but it uses the **full** 64×64 task with
  4 architectures × 3 seeds — that's an overnight run on a laptop. Prefer the easy
  curriculum above for a first result.
- If `pip install torch` fails, check the CUDA version in the top‑right of `nvidia-smi`
  and swap the index URL (e.g. `cu118` instead of `cu121`).
- This build already includes the critic‑execution fix and a BOM‑proof config loader,
  so no manual patching is needed.

<div align="center">

# 🚁 Swarm-SAR

### Decentralized Multi-Agent Reinforcement Learning for Cooperative Search & Rescue Drone Swarms

*Centralized Training, Decentralized Execution (CTDE) · Multi-Agent PPO · Transformer + Graph Neural Networks*

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Research question

> **How can decentralized Multi-Agent Reinforcement Learning improve cooperative search-and-rescue efficiency under realistic communication, sensing, energy, and environmental constraints?**

Swarm-SAR studies *cooperative intelligence*, not just flight control. A swarm of 3–10 autonomous drones must explore an unknown disaster site, detect and rescue victims, share discoveries over an unreliable radio link, avoid collisions, manage battery, and gracefully tolerate hardware faults — all with **decentralized** policies that see only local observations and neighbour messages.

<div align="center">
<img src="results/figures/fig01_architecture.png" width="720"/>
</div>

## Highlights

- **Realistic procedural environments** — every episode generates a new disaster site (buildings, roads, trees, rubble, fire, smoke, dynamic/static obstacles, charging stations, victims, no-fly zones) to prevent memorization.
- **Full sensing & actuation stack** — camera, IMU, GPS (noise + drift + dropout), LiDAR, comms; a 10-action discrete controller; point-mass dynamics with wind and motion noise.
- **CTDE MAPPO** with five interchangeable encoders: **MLP · GRU · GNN · Transformer · Transformer+GNN**.
- **Decentralized comms** with limited range, latency and packet loss — and a study of how comm quality shapes swarm intelligence.
- **Six task-allocation strategies**: random, nearest, Hungarian (provably optimal, SciPy-free fallback), auction (Bertsekas), consensus-based auction (CBAA), and RL-based.
- **Fault injection & robustness**: drone loss, GPS/camera failure, comm loss, motor degradation, with recovery.
- **A novel metric — the Swarm Intelligence Score (SIS)** — a weighted *geometric* mean of coverage, rescue, energy, communication and safety that rewards *balanced* competence.
- **20 publication-quality figures**, an animated replay (GIF/MP4), and a live **Streamlit dashboard**.
- **Runs anywhere**: the self-contained NumPy/Matplotlib backend needs no GPU or AirSim; an **AirSim/Unreal** adapter provides a high-fidelity backend, and a **ROS2** node gives a deployment skeleton (obs assembly + policy + cmd_vel/broadcast publishing) for hardware integration.

## Selected results (measured on the self-contained backend)

These come directly from the shipped code (`scripts/run_demo.py`, `scripts/run_experiments.py`) — **no training required** — using the cooperative *heuristic* controller as a strong baseline. Learning curves for the neural policies are marked *illustrative* until a GPU training run replaces them.

| Communication regime | Coverage | Victims rescued | Collisions/step | **SIS** |
|---|---:|---:|---:|---:|
| **No comms** (range 0, 100% loss) | 84% | low | high | **25.2 ± 12.2** |
| **Lossy comms** (30 m, 30% loss) | 80% | med | med | **69.5 ± 19.1** |
| **Full comms** (30 m, 5% loss) | 80% | high | low | **71.8 ± 20.3** |

> **Finding (updated after the SIS communication term was redefined as outcome-linked *utility* rather than message activity).** For the scripted controller, communication is worth **+5.0 SIS and +0.8 victims** (71.6 → 76.6, no-comm → full-comm, 5 seeds) — real but modest, because a silent swarm that partitions its search well now earns the coordination credit it deserves. The earlier "communication triples SIS" headline was largely a metric artifact: the activity-based term zeroed silent swarms. For the **learned** policies the channel is causally inert: SIS is flat across no/lossy/full comm and the assisted-rescue rate (rescues where the rescuer's first knowledge of the victim arrived by radio) is ≈0 — the policies communicate but never learned to convert messages into rescues. Closing that gap is the clearest open problem this benchmark poses.

Single best demo episode (heuristic controller): **8/8 victims rescued, 72% coverage, SIS 76.7**; benchmark IQM across the 20 frozen maps: heuristic **74.5 [67.2, 76.7]**, learned MAPPO at 3M steps (best checkpoint per seed, 5 seeds): **GRU 64.6 ± 3.4**, **Transformer-GNN 62.3 ± 3.0**. Measured decision latency for the *neural* policy: **≈7 ms/step (~145 Hz) on laptop CPU** (`scripts/evaluate.py`, `inference` block — hardware-dependent; re-measure on your target platform).



## Research-grade additions (v0.2)

Following an in-depth review (see [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)):

- **Turnkey GPU training** — one click (`run_training.bat`/`.sh`) trains, evaluates, and regenerates figures from **real** logs. See [`TRAIN_ON_YOUR_GPU.md`](TRAIN_ON_YOUR_GPU.md).
- **Decentralization fixed** — observations use each agent's own belief (no global-truth leak); peers visible only within (noisy) comm range.
- **Classical baselines** — frontier, Lloyd/Voronoi coverage control, greedy-TSP, lawnmower, and a privileged oracle ceiling (`scripts/run_benchmark.py`).
- **Reliable stats** — IQM + stratified bootstrap CIs (Agarwal et al. 2021), performance profiles.
- **Frozen benchmark** — versioned held-out maps; in-distribution, OOD, and 10-drone scale regimes.
- **Learned bandwidth-constrained communication** masked by the *physical* channel (range **and** per-edge packet loss) + a graceful-degradation curve vs packet loss.
- **Vectorized exact-LOS sensing** — per-radius precomputed Bresenham ray tables turn the sensing sweep into NumPy fancy indexing (~1.7× faster env steps despite a much richer observation).
- **Hyperparameter search** — `scripts/tune_hparams.py` (Optuna TPE) emits a study report and a ready-to-train best-config YAML (`pip install -e ".[tune]"`).
- **Audited, bounded reward** — every dense term normalized to O(weight) per step with a documented before/after audit ([`docs/reward_design.md`](docs/reward_design.md)); **triage-style time-critical rescue** (severity × time-decay) turns rescue ordering into a real scheduling problem.
- **Performance-gated curriculum** — stages advance when the rolling training-window rescue rate at the current stage clears its threshold (linear schedule as fallback floor), a fraction of envs stays pinned to the benchmark to soften distribution shocks, and every transition is logged.
- **MAPPO done properly** — running value normalization with clipped Huber loss, a compact ~50-dim *privileged* critic state (true victim/drone/mission summary) instead of concatenated observation stacks, episode clock in the observation, truncation bootstrapping, LR + imitation-coefficient annealing, orthogonal initialization, per-minibatch advantage normalization, and a masked return-to-base macro-action so autopilot steps never poison the PPO buffer.
- **SIS validated** — ranking stable under weight perturbation (ρ=0.96), Pareto analysis, geometric-mean justification.
- **Sharper detection** — false positives + confirmation; principled path-efficiency & documented safety scaling.

## Quickstart

```bash
# 1) core install (NumPy/Matplotlib only — runs the full simulation & figures)
pip install -r requirements.txt && pip install -e .

# 2) run a decentralized SAR episode and export figures
python scripts/run_demo.py --episodes 3 --figures

# 3) regenerate all 20 publication figures + animation
python scripts/generate_figures.py --out results/figures

# 4) reproduce the communication ablation (multi-seed, mean ± std)
python scripts/run_experiments.py --config configs/experiments/ablation_comm.yaml

# 5) launch the live dashboard
streamlit run src/swarm_sar/dashboard/app.py

# 6) (optional) train neural MAPPO — needs the RL extras & a GPU
pip install -e ".[rl]"
python scripts/train_and_report.py --config configs/training/mappo_transformer_gnn.yaml --archs transformer_gnn --seeds 0
```

Docker:

```bash
docker compose up swarm-sar     # runs the demo, writes results/
docker compose up dashboard     # serves the dashboard on :8501
```

## Benchmark, model zoo & deployment

```bash
# Evaluate any checkpoint on the frozen 20-map benchmark (utility-based SIS)
python scripts/evaluate.py --ckpt path/to/best.pt --out results/eval.json

# Regenerate the public leaderboard from result JSON (PR-based submissions)
python scripts/update_leaderboard.py results/*.json      # -> docs/leaderboard.md

# Export a policy for edge inference (TorchScript + ONNX + I/O contract)
python scripts/export_model.py --ckpt path/to/best.pt

# Dump an episode and open the zero-dependency browser replay viewer
python scripts/export_replay.py --out website/replay_data.json
#   then open website/replay.html
```

- **[Benchmark & leaderboard](docs/leaderboard.md)** — SwarmSAR-Bench v1 (20 frozen
  maps), ranked, with a PR submission protocol and fair-comparison rules.
- **[Model zoo](docs/model_zoo.md)** — the trained checkpoints with per-model cards.
- **Docs site:** <https://mathi0405.github.io/drone-swarm-sar> · **Live demo:** deploy
  `spaces/` to Hugging Face Spaces (see [PUBLISHING.md](PUBLISHING.md)).

## Key finding: does learned communication help?

The SIS communication term measures outcome-linked **utility** (comm-assisted
rescues + coverage uniqueness), not message volume. Under it, communication buys
the scripted coordinator **+5.0 SIS / +0.8 victims** (71.6 → 76.6). The central
**negative result**: our *learned* policies leave the channel **causally inert** —
they transmit, but the assisted-rescue rate is ≈ 0, and a matched reward for
acting on messages does not fix it (it's a credit-assignment problem). GRU and
Transformer+GNN finish **statistically tied** at 3M steps (64.6 ± 3.4 vs
62.3 ± 3.0; 95% CI on the difference straddles zero).

## Figure gallery

| | | |
|---|---|---|
| ![](results/figures/fig02_environment.png) | ![](results/figures/fig03_trajectory_3d.png) | ![](results/figures/fig05_coverage_heatmap.png) |
| Procedural environment | 3-D trajectories | Coverage heatmap |
| ![](results/figures/fig06_dashboard.png) | ![](results/figures/fig07_comm_graph.png) | ![](assets/fig04_replay.gif) |
| Swarm dashboard | Communication graph | Animated replay |

See [`results/figures/`](results/figures) for all 20 figures.

## Architecture

```
src/swarm_sar/
├── config.py            # typed, hierarchical YAML config (everything is configurable)
├── environment/         # procedural world, PettingZoo-parallel env, AirSim adapter
├── drone/               # state + point-mass dynamics (wind, motion noise)
├── sensors/             # camera / IMU / GPS / LiDAR with noise, drift, dropout
├── battery/             # energy model, charging, ageing/degradation
├── communication/       # range / latency / packet-loss message bus
├── mission/             # planner FSM + 5 task-allocation strategies
├── faults/              # fault injection & recovery
├── policies/            # MLP · GNN · Transformer · Transformer+GNN (torch-optional)
├── training/            # MAPPO (CTDE) + Ray RLlib config + rollout worker
├── evaluation/          # 12 metrics + Swarm Intelligence Score
├── visualization/       # 20 publication figures + animation
├── dashboard/           # Streamlit app
└── logging_utils/       # CSV / JSON / TensorBoard logging
```

Full details in [`docs/architecture.md`](docs/architecture.md). Other docs: [installation](docs/installation.md) · [configuration](docs/configuration.md) · [experiments](docs/experiments.md) · [metrics & SIS](docs/metrics.md).

## Observation / Action / Reward

- **Observation (per drone, partial/decentralized):** an 8-frame temporal stack of — own pose (noisy GPS), velocity, altitude and battery; a victim-belief summary (count, density, two nearest believed victims); locally sensed fire/smoke distances and densities (within camera radius only); an egocentric 7×7 map patch (occupancy · own explored mask · own victim belief); an 8-ray LiDAR scan; the k-nearest in-range peers with their intent (relative position, rescue target, mode); and a comm-link summary (messages received, age, signal strength, loss rate).
- **Actions (10 discrete, 7 enabled by default):** hover, N/S/E/W move, broadcast, return-to-base; ascend/descend/rotate exist but are masked out by default because altitude/yaw only affect energy in the 2.5-D simulation.
- **Reward (see [`docs/reward_design.md`](docs/reward_design.md)):** every dense term is footprint-normalized and bounded per step, so mission events dominate by construction; rescues are **time-critical** — scaled by victim severity and decaying over the episode (triage). `+` new/frontier/team-new cells, victim detected/classified/rescued(×urgency), mission complete, novel broadcasts, safe separation; `−` collision, near-miss, hazard, battery depletion, duplicate exploration, hover/idle, excessive comms, time pressure. Sparse mode keeps only mission/safety events for reward ablations.

## Reproducibility

Fixed seeds throughout; every metric reported as mean ± std across seeds; each run serializes its resolved config. `make test` runs the suite; CI runs lint + tests + a smoke demo on Python 3.11 (matching `requires-python`).

## Citation

```bibtex
@software{swarmsar2026,
  title  = {Swarm-SAR: Decentralized Multi-Agent RL for Cooperative Search-and-Rescue Drone Swarms},
  author = {Manichandan, Mathi},
  year   = {2026},
  url    = {https://github.com/mathi0405/drone-swarm-sar}
}
```

## License

MIT — see [LICENSE](LICENSE).

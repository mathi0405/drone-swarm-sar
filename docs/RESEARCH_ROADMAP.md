# Reviewer report & research roadmap

A candid, professor-style review of Swarm-SAR and the concrete plan to take it from
a strong systems scaffold to a top-venue (ICRA/IROS/CoRL/NeurIPS) result. Items
marked ✅ are already implemented in this repository; ⏳ require your GPU/compute.

## Verdict

The engineering is top-decile; the science is gated on one thing: **train the
policies and let real results replace the heuristic/illustrative numbers.** The
apparatus is now complete and instrumented for exactly that.

## Tier 1 — Close the scientific gap
- ⏳ Train MAPPO to convergence (4 archs × ≥8 seeds) — one command: `run_training.bat` / `scripts/train_and_report.py`.
- ✅ Turnkey GPU pipeline that trains → evaluates → regenerates **real** figures from logs (`*_REAL.png`) and reports IQM + bootstrap CIs.

## Tier 2 — Corrections & alignments (done)
- ✅ Removed the decentralization leak: observation now uses each agent's **own belief** (coverage/victim map), not global truth; peers are visible **only within comm range** and with **noisy** relative position. (New tests assert this.)
- ✅ Detection now has **false positives** + a **confirmation** step (precision < 1); belief **decays** (uncertainty grows).
- ✅ Principled `path_efficiency` (vs a lawnmower lower bound) and a **documented** safety constant (`COLLISION_RATE_CAP`).
- ✅ **Sparse-reward mode** + reward-ablation support (`reward.mode: shaped|sparse`).

## Tier 3 — Evaluation rigor (done)
- ✅ Classical baselines: **frontier**, **Lloyd/Voronoi coverage control**, **greedy-TSP**, **lawnmower**, and a privileged **oracle** upper bound.
- ✅ **rliable-style statistics**: IQM, stratified bootstrap CIs, performance profiles, probability-of-improvement.
- ✅ **Frozen benchmark suite** (versioned held-out maps) + `run_benchmark.py`; in-distribution, **OOD**, and **scale** (10-drone) regimes.
- ⏳ Run the full 20-map benchmark with your trained checkpoint (`--checkpoint`).

## Tier 4 — Sharpen the contribution (done / ⏳ to train)
- ✅ **Learned bandwidth-constrained communication** head (gate + top-k budget + TarMAC-style attention + comm-cost regularizer) — the paper's sharpened thesis.
- ✅ **Graceful-degradation curve** vs packet loss (a curve, not 3 points).
- ⏳ Train the comm head and compare emergent protocols; add MAT as a learned baseline.

## Tier 5 — Simulation fidelity
- ✅ Shared reward module used by both backends; AirSim adapter now scores with the same objective (no longer a stub).
- ⏳ Port to Isaac Lab / Flightmare with real quadrotor dynamics for the fidelity result; validate on the ROS2 nodes.

## Tier 6 — Validate the SIS (done)
- ✅ Weight-sensitivity: ranking is stable (mean Spearman **ρ = 0.96**, top-1 retained **81%** over 2000 reweightings).
- ✅ Geometric-vs-linear justification (imbalance penalty) and **Pareto-front** analysis across the five objectives.

## Tier 7 — Engineering to lab standard (done)
- ✅ Full determinism (`set_deterministic`), optional **W&B** logging, **RLlib new API stack** (RLModule/Learner), pinned `constraints.txt` (+ pip-tools guidance), **checkpoint/resume**, mypy in CI.
- ⏳ Cluster sweeps (Hydra + Ray Tune/submitit) for the large studies.

## Tier 8 — Positioning, theory, ethics (done)
- ✅ Deepened related work (coverage control, CommNet/DIAL/TarMAC/IC3Net, **MAT**), a light **Proposition** on when communication helps, and a **Broader Impact** statement. All 23 citations resolve.

## Suggested timeline
1. Weeks 1–6: Tier-1 training + full benchmark with trained checkpoints.
2. Weeks 6–14: learned comms as the centerpiece + MAT baseline + degradation curves.
3. Weeks 14–22: Isaac/Flightmare fidelity + scaling/generalization + write-up.

Targets: workshop after phase 1; ICRA/IROS after phase 2; CoRL/RSS or NeurIPS D&B after phase 3.

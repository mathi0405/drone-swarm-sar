# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-21

First public release.

### Highlights
- Decentralized MARL search-and-rescue environment (PettingZoo-style, procedural
  disaster worlds, occlusion/fire/smoke, faults, battery, limited comms).
- CTDE MAPPO trainer with shared/independent policies, a centralized privileged
  critic, value normalization, LR + imitation annealing (with warm-up), and a
  performance-gated curriculum.
- Policy architectures: MLP, GRU, GNN, Transformer (entity-tokenized), and the
  proposed Transformer+GNN.
- Frozen 20-map benchmark (seeds 200-219) with IQM + bootstrap CIs, and a
  utility-based Swarm Intelligence Score whose communication term measures
  outcome-linked utility (comm-assisted rescues + coverage uniqueness) rather
  than message activity.
- A 3M-step × 5-seed campaign for GRU and Transformer+GNN; the two are
  statistically indistinguishable (P(GRU>T-GNN)=0.76, mean SIS diff +2.24, 95%
  CI straddling 0). Trained-policy communication ablation shows the learned
  channel is causally inert (assisted-rescue rate ≈ 0).
- Reproducibility: fixed seeds, serialized configs, CI (lint + tests + smoke
  demo), Docker/compose, MkDocs documentation site, and a public benchmark
  leaderboard + model zoo generated from result JSON.

[Unreleased]: https://github.com/mathi0405/drone-swarm-sar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mathi0405/drone-swarm-sar/releases/tag/v0.1.0

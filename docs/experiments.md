# Experiments

## Protocol

- **Multi-seed.** Every result is averaged over ≥5 seeds; we report mean ± std.
- **Fresh maps.** With `randomize_each_episode: true`, train and test never share a
  map, so success measures *generalization*, not memorization.
- **Ablations.** Toggle one factor at a time (comms, GNN, Transformer, battery,
  dynamics) and measure the change in the Swarm Intelligence Score.

## Built-in ablations

```bash
# communication quality (no / lossy / full)
python scripts/run_experiments.py --config configs/experiments/ablation_comm.yaml

# architecture comparison (PPO / +GNN / +Transformer / +Transformer+GNN)
python scripts/run_experiments.py --config configs/experiments/ablation_arch.yaml
```

Add your own by dropping a YAML in `configs/experiments/` with a `grid:` of named
variants and a list of `seeds:`.

## Reproducing the figures

```bash
python scripts/generate_figures.py --out results/figures --assets assets
```

Real, rollout-derived figures: environment, 3-D trajectories, animation, coverage
heatmap, dashboard, communication graph, battery, rescue/detection/failure
timelines, exploration entropy, task allocation. Figures that require a trained
network (reward/learning curves, attention, the 4-architecture bar chart) use
documented *illustrative* data (`visualization/synthetic.py`) and are labelled as
such — replace them with your TensorBoard/CSV logs after `scripts/train.py`.

## Suggested study matrix

| Factor | Levels |
|--------|--------|
| Communication | none · lossy · full |
| Architecture | MLP · GNN · Transformer · Transformer+GNN |
| Swarm size | 3 · 5 · 10 drones |
| Environment | static · dynamic obstacles |
| Energy | unconstrained · battery-aware |
| Faults | off · on (robustness) |

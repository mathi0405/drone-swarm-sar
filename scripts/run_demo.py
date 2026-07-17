#!/usr/bin/env python3
"""Run a self-contained heuristic SAR episode and (optionally) export quick figures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.training.rollout import run_episode, HeuristicActor
from swarm_sar.evaluation.metrics import episode_metrics, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--strategy", default="auction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures", action="store_true", help="export a few quick figures")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    mets = []
    last = None
    for e in range(args.episodes):
        cfg = EnvConfig(num_drones=args.drones, max_steps=args.steps,
                        world=WorldConfig(size=args.size, n_victims=8))
        env = SARSwarmEnv(cfg, seed=args.seed + e)
        log = run_episode(env, HeuristicActor(env, args.strategy, args.seed + e), seed=args.seed + e)
        m = episode_metrics(log, grid_size=args.size); mets.append(m); last = log
        print(f"[ep {e}] " + " ".join(f"{k}={m[k]:.2f}" for k in
              ["coverage", "victims_rescued", "collision_rate",
               "communication_efficiency", "swarm_intelligence_score"]))

    agg = aggregate(mets)
    print("\n=== aggregate (mean ± std) ===")
    for k, v in agg.items():
        print(f"  {k:28s} {v['mean']:.3f} ± {v['std']:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "demo_metrics.json", "w") as f:
        json.dump({"aggregate": agg}, f, indent=2)

    if args.figures and last is not None:
        from swarm_sar.visualization import plots
        fdir = Path(args.out) / "figures"; fdir.mkdir(parents=True, exist_ok=True)
        plots.fig_world_map(last, fdir / "demo_environment.png")
        plots.fig_trajectory_3d(last, fdir / "demo_trajectory_3d.png")
        plots.fig_coverage_heatmap(last, args.size, fdir / "demo_coverage_heatmap.png")
        plots.fig_dashboard_snapshot(last, args.size, fdir / "demo_dashboard.png")
        print(f"\n[OK] figures written to {fdir}")


if __name__ == "__main__":
    main()

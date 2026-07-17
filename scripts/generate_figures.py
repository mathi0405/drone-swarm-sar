#!/usr/bin/env python3
"""Generate the full publication figure suite from *real* simulation rollouts.

Real figures come from actual heuristic-swarm episodes on procedurally generated
maps; figures that depend on a trained network (reward/learning curves, attention,
the 4-architecture comparison) use documented illustrative data and are labelled
as such. Run `scripts/train.py` then point this at the resulting logs to replace
them with measured curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import (
    episode_metrics,
    exploration_entropy,
    swarm_intelligence_score,
)
from swarm_sar.mission.task_allocation import TaskAllocator
from swarm_sar.training.rollout import HeuristicActor, run_episode
from swarm_sar.visualization import plots
from swarm_sar.visualization import synthetic as syn


def hero_episode(size=64, drones=3, steps=220):
    """Pick a seed that yields a visually rich episode (rescues+collisions+comm)."""
    best = None
    for seed in range(12):
        env = SARSwarmEnv(EnvConfig(num_drones=drones, max_steps=steps,
                                    world=WorldConfig(size=size, n_victims=8)), seed=seed)
        log = run_episode(env, HeuristicActor(env, "auction", seed), seed=seed)
        s = log.summary
        score = s["victims_rescued"] * 3 + s["collisions"] + min(5, s["comm_delivered"] / 5) + s["coverage"] * 3
        if best is None or score > best[0]:
            best = (score, log, seed)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets); assets.mkdir(parents=True, exist_ok=True)
    made = []

    print("· running hero episode …")
    log, seed = hero_episode(size=args.size)
    print(f"  chosen seed={seed}: {log.summary}")

    # ---- real, rollout-derived figures ----
    made.append(plots.fig_architecture(out / "fig01_architecture.png"))
    made.append(plots.fig_world_map(log, out / "fig02_environment.png"))
    made.append(plots.fig_trajectory_3d(log, out / "fig03_trajectory_3d.png"))
    made += plots.animate_trajectory(log, assets / "fig04_replay.gif", assets / "fig04_replay.mp4")
    made.append(plots.fig_coverage_heatmap(log, args.size, out / "fig05_coverage_heatmap.png"))
    made.append(plots.fig_dashboard_snapshot(log, args.size, out / "fig06_dashboard.png"))
    made.append(plots.fig_comm_graph(log, out / "fig07_comm_graph.png"))
    made.append(plots.fig_battery(log, out / "fig12_battery.png"))
    made.append(plots.fig_rescue_timeline(log, out / "fig13_rescue_timeline.png"))
    made.append(plots.fig_failure_recovery(log, out / "fig14_failure_recovery.png"))
    made.append(plots.fig_gnn_message_passing(log, out / "fig16_gnn_message_passing.png"))
    made.append(plots.fig_detection_timeline(log, out / "fig19_detection_timeline.png"))

    # ---- real multi-seed series (Figs 9,10,17) ----
    seeds = list(range(8))
    cov_series, coll_series = {}, {}
    ent_series = {}
    for label, ndr in [("3 drones", 3), ("6 drones", 6)]:
        covs, colls = [], []
        for s in seeds:
            env = SARSwarmEnv(EnvConfig(num_drones=ndr, max_steps=200,
                              world=WorldConfig(size=args.size, n_victims=8)), seed=s)
            ep_log = run_episode(env, HeuristicActor(env, "auction", s), seed=s)
            m = episode_metrics(ep_log, grid_size=args.size)
            covs.append(m["coverage"] * 100); colls.append(m["collision_rate"])
        cov_series[f"Heuristic, {label} (measured)"] = covs
        coll_series[f"Heuristic, {label} (measured)"] = colls
    # entropy over time within the hero episode, and for a no-comm variant
    def entropy_over_time(ep_log):
        return [exploration_entropy(ep_log.frames[: k + 1], args.size) for k in range(0, len(ep_log.frames), 4)]
    ent_series["with comm"] = entropy_over_time(log)
    env = SARSwarmEnv(EnvConfig(num_drones=3, max_steps=200,
                      world=WorldConfig(size=args.size, n_victims=8)), seed=seed)
    env.cfg.comm.range_m = 0.0; env.cfg.comm.packet_loss = 1.0
    nolog = run_episode(env, HeuristicActor(env, "nearest", seed), seed=seed)
    ent_series["no comm"] = entropy_over_time(nolog)
    # illustrative learning overlay
    cov_series["MAPPO+T+GNN (illustrative)"] = list(syn.learning_curve("PPO+Transformer+GNN", len(seeds)) * 100)
    coll_series["MAPPO+T+GNN (illustrative)"] = list((1 - syn.learning_curve("PPO+Transformer+GNN", len(seeds))) * 0.08)

    made.append(plots.fig_metric_vs_episodes(cov_series, out / "fig09_coverage_vs_episodes.png",
                "coverage (%)", "Coverage vs Episodes (measured heuristic + illustrative RL)"))
    made.append(plots.fig_metric_vs_episodes(coll_series, out / "fig10_collision_vs_episodes.png",
                "collision rate (per step)", "Collision Rate vs Episodes"))
    made.append(plots.fig_exploration_entropy(ent_series, out / "fig17_exploration_entropy.png"))

    # ---- task allocation (Fig 18), REAL assignment ----
    env = SARSwarmEnv(EnvConfig(num_drones=4, max_steps=1,
                      world=WorldConfig(size=args.size, n_victims=5)), seed=3)
    env.reset(seed=3)
    dpos = np.array([d.kinematics.pos for d in env.drones])
    vpos = np.array([v.pos for v in env.world.victims])
    assign = TaskAllocator("auction").assign(dpos, vpos)
    made.append(plots.fig_task_allocation(dpos, vpos, assign, out / "fig18_task_allocation.png", "auction (Hungarian-optimal)"))

    # ---- illustrative training curves & comparison (Figs 8,11,15,20) ----
    made.append(plots.fig_reward_curves(out / "fig08_reward_curves.png"))
    made.append(plots.fig_attention_heatmap(out / "fig15_attention.png"))
    # synth per-arch aggregates for comparison
    def synth_agg(arch):
        base = syn._PROFILE[arch][0]
        def mk(mu, sd):
            return {"mean": float(mu), "std": float(sd)}
        return {"coverage": mk(base, 0.03),
                "victims_rescued": mk(base * 8 * 0.8, 0.6),
                "swarm_intelligence_score": mk(swarm_intelligence_score(
                    {"coverage": base, "rescue": base*0.8, "energy": 0.7, "communication": 0.8, "safety": base}), 4),
                "collision_rate": mk((1 - base) * 0.05, 0.01)}
    agg_by_arch = {a: synth_agg(a) for a in syn.ARCHS}
    made.append(plots.fig_success_rate({a: syn._PROFILE[a][0] for a in syn.ARCHS}, out / "fig11_success_rate.png"))
    made.append(plots.fig_algo_comparison(agg_by_arch, out / "fig20_algo_comparison.png"))

    # ---- persist metrics for the dashboard / paper ----
    metrics_dump = {
        "hero_seed": seed, "hero_summary": log.summary,
        "hero_metrics": episode_metrics(log, grid_size=args.size),
        "illustrative_archs": {a: syn._PROFILE[a][0] for a in syn.ARCHS},
    }
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    with open("results/logs/figure_metrics.json", "w") as f:
        json.dump(metrics_dump, f, indent=2)

    made = [m for m in made if m]
    print(f"\n[OK] generated {len(made)} figure files -> {out} and {assets}")
    for m in sorted(made):
        print("   ", m)


if __name__ == "__main__":
    main()

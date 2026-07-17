#!/usr/bin/env python3
"""Run all methods on the FROZEN benchmark suite; report IQM + 95% bootstrap CIs.

Every method sees identical maps (in-distribution or OOD), so the comparison is
fair and reproducible. Add --checkpoint to include a trained MAPPO policy.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.training.rollout import run_episode, HeuristicActor
from swarm_sar.mission.baselines import BASELINES, ControllerActor
from swarm_sar.evaluation.metrics import episode_metrics
from swarm_sar.evaluation.stats import iqm, bootstrap_ci
from swarm_sar.evaluation.benchmark import suite, BENCHMARK_VERSION
from swarm_sar.visualization import plots


def build_methods(checkpoint=None):
    m = {name: (lambda env, C=C: ControllerActor(C(env, seed=0))) for name, C in BASELINES.items()}
    m["heuristic_auction"] = lambda env: HeuristicActor(env, "auction", 0)
    if checkpoint:
        from swarm_sar.training.rollout import load_neural_actor
        actor = load_neural_actor(checkpoint)
        m["mappo_transformer_gnn"] = lambda env: actor
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="in_dist", choices=["in_dist", "ood", "scale"])
    ap.add_argument("--n-maps", type=int, default=None, help="subset frozen maps (quick runs)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    frozen = suite(args.regime)
    if args.n_maps:
        frozen = frozen[: args.n_maps]
    methods = build_methods(args.checkpoint)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark {BENCHMARK_VERSION} | regime={args.regime} | {len(frozen)} frozen maps\n")
    results, table = {}, []
    for name, factory in methods.items():
        mets = []
        for seed, cfg in frozen:
            env = SARSwarmEnv(cfg, seed=seed)
            mets.append(episode_metrics(run_episode(env, factory(env), seed=seed),
                                        grid_size=cfg.world.size))
        sis = np.array([m["swarm_intelligence_score"] for m in mets])
        lo, hi = bootstrap_ci(sis)
        row = {"method": name, "sis_iqm": iqm(sis), "sis_ci_lo": lo, "sis_ci_hi": hi,
               "coverage": float(np.mean([m["coverage"] for m in mets])),
               "victims_rescued": float(np.mean([m["victims_rescued"] for m in mets])),
               "collision_rate": float(np.mean([m["collision_rate"] for m in mets])),
               "success_rate": float(np.mean([m["mission_success"] for m in mets]))}
        results[name] = {"summary": {"iqm": row["sis_iqm"], "ci95_low": lo, "ci95_high": hi}}
        table.append(row)

    table.sort(key=lambda r: r["sis_iqm"], reverse=True)
    print(f"{'method':24s} {'SIS (IQM [95% CI])':24s} {'cov%':>5s} {'resc':>5s} {'succ%':>6s}")
    for r in table:
        ci = f"{r['sis_iqm']:.1f} [{r['sis_ci_lo']:.1f},{r['sis_ci_hi']:.1f}]"
        print(f"{r['method']:24s} {ci:24s} {r['coverage']*100:5.0f} "
              f"{r['victims_rescued']:5.1f} {r['success_rate']*100:6.0f}")

    with open(out / f"benchmark_{args.regime}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with open(out / f"benchmark_{args.regime}.json", "w") as f:
        json.dump(table, f, indent=2)
    plots.fig_method_comparison(results, out / f"benchmark_{args.regime}.png",
                                title=f"Benchmark ({args.regime}): SIS by method (IQM ± 95% CI)")
    print(f"\n[OK] saved {out}/benchmark_{args.regime}.[csv|json|png]")


if __name__ == "__main__":
    main()

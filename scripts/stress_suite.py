#!/usr/bin/env python3
"""Stress-test heuristic vs learned control where a scripted controller should break.

Runs every method on the FROZEN benchmark maps under escalating stressors —
amplified fault injection, a packet-loss degradation sweep, OOD maps, and
10-drone scaling — and reports SIS IQM + 95% bootstrap CIs per condition.
This is the measurement behind the sharpened claim: if the learned swarm only
ties the heuristic on nominal missions, the publishable story is how each
degrades under faults, comm loss, distribution shift and scale.

Usage:
  python scripts/stress_suite.py                          # heuristic only
  python scripts/stress_suite.py --checkpoint best.pt     # + trained policy
  python scripts/stress_suite.py --all-baselines          # + scripted baselines
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.benchmark import BENCHMARK_VERSION, suite
from swarm_sar.evaluation.metrics import episode_metrics
from swarm_sar.evaluation.stats import bootstrap_ci, iqm
from swarm_sar.mission.baselines import BASELINES, ControllerActor
from swarm_sar.training.rollout import HeuristicActor, run_episode
from swarm_sar.visualization import plots

PACKET_LOSS_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]


def build_methods(checkpoint: str | None, all_baselines: bool) -> dict:
    methods = {"heuristic_auction": lambda env: HeuristicActor(env, "auction", 0)}
    if all_baselines:
        for name, C in BASELINES.items():
            methods[name] = (lambda env, C=C: ControllerActor(C(env, seed=0)))
    if checkpoint:
        from swarm_sar.training.rollout import load_neural_actor
        methods["mappo_learned"] = lambda env, _c=checkpoint: load_neural_actor(_c)
    return methods


def stressed_suite(condition: str, fault_scale: float,
                   packet_loss: float | None = None) -> list[tuple[int, object]]:
    """Frozen (seed, EnvConfig) pairs for one stress condition.

    Conditions reuse the frozen map seeds, so per-map pairing across
    conditions (and across methods) stays intact.
    """
    if condition == "ood":
        return suite("ood")
    if condition == "scale":
        return suite("scale")
    pairs = []
    for seed, cfg in suite("in_dist"):
        cfg = cfg.model_copy(deep=True)
        if condition == "faults":
            f = cfg.faults
            f.enable = True
            f.drone_failure_prob = min(0.25, f.drone_failure_prob * fault_scale)
            f.gps_failure_prob = min(0.25, f.gps_failure_prob * fault_scale)
            f.camera_failure_prob = min(0.25, f.camera_failure_prob * fault_scale)
            f.comm_loss_prob = min(0.5, f.comm_loss_prob * fault_scale)
            f.motor_degradation_prob = min(0.25, f.motor_degradation_prob * fault_scale)
        elif condition == "packet_loss":
            cfg.comm.packet_loss = float(packet_loss)
        elif condition != "nominal":
            raise ValueError(condition)
        pairs.append((seed, cfg))
    return pairs


def run_condition(methods: dict, pairs, n_maps: int | None) -> dict[str, dict]:
    if n_maps:
        pairs = pairs[:n_maps]
    out = {}
    for name, factory in methods.items():
        mets, rescue_ratios = [], []
        for seed, cfg in pairs:
            env = SARSwarmEnv(cfg, seed=seed)
            log = run_episode(env, factory(env), seed=seed)
            m = episode_metrics(log, grid_size=cfg.world.size)
            mets.append(m)
            rescue_ratios.append(m["victims_rescued"] / max(1, cfg.world.n_victims))
        sis = np.array([m["swarm_intelligence_score"] for m in mets])
        lo, hi = bootstrap_ci(sis)
        out[name] = {
            "n_maps": len(pairs),
            "sis_iqm": float(iqm(sis)),
            "sis_ci95": [float(lo), float(hi)],
            "sis_mean": float(sis.mean()),
            "sis_std": float(sis.std()),
            "rescue_rate": float(np.mean(rescue_ratios)),
            "victims_rescued": float(np.mean([m["victims_rescued"] for m in mets])),
            "coverage": float(np.mean([m["coverage"] for m in mets])),
            "collision_rate": float(np.mean([m["collision_rate"] for m in mets])),
            "mission_success": float(np.mean([m["mission_success"] for m in mets])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="trained policy .pt to include")
    ap.add_argument("--all-baselines", action="store_true",
                    help="also run lawnmower/frontier/greedy_tsp/lloyd/oracle")
    ap.add_argument("--conditions", nargs="+",
                    default=["nominal", "faults", "packet_loss", "ood", "scale"],
                    choices=["nominal", "faults", "packet_loss", "ood", "scale"])
    ap.add_argument("--packet-loss", type=float, nargs="+", default=PACKET_LOSS_SWEEP)
    ap.add_argument("--fault-scale", type=float, default=10.0,
                    help="multiplier on nominal per-step fault probabilities")
    def _positive(v):
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError("--n-maps must be >= 1")
        return n
    ap.add_argument("--n-maps", type=_positive, default=None, help="subset maps (quick runs)")
    ap.add_argument("--out", default="results/stress")
    args = ap.parse_args()

    methods = build_methods(args.checkpoint, args.all_baselines)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stress suite | benchmark {BENCHMARK_VERSION} | methods: {', '.join(methods)}")

    report: dict[str, dict] = {}
    rows = []
    for condition in args.conditions:
        if condition == "packet_loss":
            for p in args.packet_loss:
                label = f"packet_loss_{p:.1f}"
                print(f"\n== {label} ==")
                res = run_condition(methods, stressed_suite("packet_loss", args.fault_scale, p),
                                    args.n_maps)
                report[label] = res
                for m, r in res.items():
                    print(f"  {m:24s} SIS {r['sis_iqm']:6.1f} "
                          f"[{r['sis_ci95'][0]:.1f},{r['sis_ci95'][1]:.1f}] "
                          f"resc {r['victims_rescued']:.1f}")
                    rows.append({"condition": label, "method": m, **{
                        k: v for k, v in r.items() if not isinstance(v, list)}})
        else:
            print(f"\n== {condition} (fault_scale={args.fault_scale if condition == 'faults' else '-'}) ==")
            res = run_condition(methods, stressed_suite(condition, args.fault_scale),
                                args.n_maps)
            report[condition] = res
            for m, r in res.items():
                print(f"  {m:24s} SIS {r['sis_iqm']:6.1f} "
                      f"[{r['sis_ci95'][0]:.1f},{r['sis_ci95'][1]:.1f}] "
                      f"resc {r['victims_rescued']:.1f}")
                rows.append({"condition": condition, "method": m, **{
                    k: v for k, v in r.items() if not isinstance(v, list)}})

    # Per-method packet-loss degradation curves (the "graceful degradation" figure).
    loss_labels = sorted(k for k in report if k.startswith("packet_loss_"))
    if loss_labels:
        losses = [float(k.rsplit("_", 1)[1]) for k in loss_labels]
        for method in methods:
            sis_mean = [report[k][method]["sis_mean"] for k in loss_labels]
            sis_std = [report[k][method]["sis_std"] for k in loss_labels]
            rescued = [report[k][method]["victims_rescued"] for k in loss_labels]
            plots.fig_comm_degradation(losses, sis_mean, sis_std, rescued,
                                       out_dir / f"degradation_{method}.png")

    with open(out_dir / "stress_suite.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if rows:
        with open(out_dir / "stress_suite.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[OK] saved {out_dir}/stress_suite.[json|csv] and degradation figures")


if __name__ == "__main__":
    main()

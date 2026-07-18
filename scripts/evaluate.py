#!/usr/bin/env python3
"""Evaluate trained neural policies across multiple seeds; report metrics + SIS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.config import load_config
from swarm_sar.evaluation.evaluator import Evaluator
from swarm_sar.evaluation.stats import bootstrap_ci, iqm
from swarm_sar.training.rollout import load_neural_actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/mappo_transformer_gnn.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to trained model checkpoint(s) (.pt)", nargs="+")
    # Canonical protocol: the frozen benchmark's 20 held-out map seeds, so
    # these numbers are directly comparable with scripts/run_benchmark.py.
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 220)))
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = {}

    for ckpt in args.ckpt:
        print(f"\n=== Evaluating {ckpt} ===")
        def actor_factory(env, _ckpt=ckpt):
            return load_neural_actor(_ckpt, deterministic=True)
        ev = Evaluator(cfg.env, actor_factory=actor_factory)

        res = ev.evaluate(args.seeds)
        # >= 10 seeds: below that the nominal/faulted SIS ratio is noise.
        rob = ev.robustness(args.seeds[:10])
        gen = ev.generalization(args.seeds[: max(1, len(args.seeds) // 2)], args.seeds[len(args.seeds) // 2:])
        inf = ev.inference_time()

        print("=== Evaluation (mean +/- std over seeds) ===")
        for k, v in res["aggregate"].items():
            print(f"  {k:28s} {v['mean']:.3f} +/- {v['std']:.3f}")
        print("robustness:", {k: round(v, 3) for k, v in rob.items()})
        print("generalization:", {k: round(v, 3) for k, v in gen.items()})
        print("inference:", {k: round(v, 3) for k, v in inf.items()})

        sis = np.array([m["swarm_intelligence_score"] for m in res["per_seed"]])
        report[ckpt] = {
            "aggregate": res["aggregate"],
            "robustness": rob,
            "generalization": gen,
            "inference": inf,
            "sis_iqm": float(iqm(sis)),
            "sis_ci95": [float(x) for x in bootstrap_ci(sis)],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] saved {args.out}")


if __name__ == "__main__":
    main()

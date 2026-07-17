#!/usr/bin/env python3
"""Run an ablation grid (from configs/experiments/*.yaml) across multiple seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import aggregate, episode_metrics
from swarm_sar.training.rollout import HeuristicActor, run_episode


def _apply(env_cfg: EnvConfig, overrides: dict):
    # EnvConfig is a pydantic model, not a dataclass — mutate a deep copy.
    # (dataclasses.replace raised TypeError here and broke the ablation grid.)
    env_cfg = env_cfg.model_copy(deep=True)
    e = overrides.get("env", {})
    for k, v in e.get("comm", {}).items():
        setattr(env_cfg.comm, k, v)
    for k, v in e.items():
        if k != "comm" and hasattr(env_cfg, k):
            setattr(env_cfg, k, v)
    return env_cfg, overrides.get("model", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/experiments.json")
    args = ap.parse_args()
    with open(args.config) as f:
        spec = yaml.safe_load(f)
    seeds = spec.get("seeds", [0, 1, 2])
    results = {}
    for name, ov in spec["grid"].items():
        base = EnvConfig(num_drones=spec.get("base", {}).get("env", {}).get("num_drones", 3),
                         max_steps=200, world=WorldConfig(size=64, n_victims=8))
        cfg, model_ov = _apply(base, ov)
        mets = []
        for s in seeds:
            env = SARSwarmEnv(cfg, seed=s)
            log = run_episode(env, HeuristicActor(env, "auction", s), seed=s)
            mets.append(episode_metrics(log, grid_size=cfg.world.size))
        results[name] = aggregate(mets)
        sis = results[name]["swarm_intelligence_score"]
        cov = results[name]["coverage"]
        print(f"{name:14s} SIS={sis['mean']:5.1f}±{sis['std']:4.1f}  coverage={cov['mean']*100:4.0f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] saved {args.out}")


if __name__ == "__main__":
    main()

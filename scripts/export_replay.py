#!/usr/bin/env python3
"""Export one episode to a compact JSON the browser replay viewer can animate.

Runs a heuristic (or, with --checkpoint, a trained) episode and serializes the
world grid, victims, charging stations, and per-frame drone state + comm edges.
The output loads in website/replay.html — no server, no Python needed to view.

Usage:
  python scripts/export_replay.py --seed 3 --out website/replay_data.json
  python scripts/export_replay.py --checkpoint best.pt --out website/replay_data.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.training.rollout import HeuristicActor, run_episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--victims", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--checkpoint", default=None, help="trained .pt (default: heuristic)")
    ap.add_argument("--out", default="website/replay_data.json")
    args = ap.parse_args()

    cfg = EnvConfig(num_drones=args.drones, max_steps=args.steps,
                    world=WorldConfig(size=args.size, n_victims=args.victims))
    env = SARSwarmEnv(cfg, seed=args.seed)
    if args.checkpoint:
        from swarm_sar.training.rollout import load_neural_actor
        actor = load_neural_actor(args.checkpoint)
        method = f"MAPPO ({Path(args.checkpoint).stem})"
    else:
        actor = HeuristicActor(env, "auction", args.seed)
        method = "Heuristic auction"
    log = run_episode(env, actor, seed=args.seed)

    grid = log.world_grid.astype(int).tolist()
    frames = [{
        "t": int(f["t"]),
        "pos": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in f["pos"]],
        "soc": [round(float(s), 3) for s in f["soc"]],
        "alive": [bool(a) for a in f["alive"]],
        "edges": [[int(a), int(b)] for a, b in f.get("comm_edges", [])],
        "coverage": round(float(f["coverage"]), 3),
        "found": int(f["found"]),
        "rescued": int(f["rescued"]),
    } for f in log.frames]

    data = {
        "method": method, "seed": args.seed, "size": args.size,
        "grid": grid,
        "victims": [{"pos": [float(v["pos"][0]), float(v["pos"][1])],
                     "severity": round(float(v.get("severity", 0.5)), 2),
                     "rescued_step": int(v.get("rescued_step", -1))}
                    for v in log.victims],
        "charging": [[float(c[0]), float(c[1])] for c in log.charging],
        "frames": frames,
        "summary": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in log.summary.items()
                    if k in ("victims_rescued", "victims_total", "coverage",
                             "collisions", "steps", "swarm_intelligence_score")},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[OK] {method}: {len(frames)} frames, {len(data['victims'])} victims "
          f"-> {out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()

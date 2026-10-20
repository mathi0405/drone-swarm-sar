#!/usr/bin/env python3
"""Communication-quality study: SIS & rescues vs packet loss (graceful degradation)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import episode_metrics
from swarm_sar.training.rollout import HeuristicActor, run_episode
from swarm_sar.visualization import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--checkpoint", default=None,
                    help="trained policy .pt; omitted = scripted heuristic "
                         "(the output is labeled with the actor either way)")
    ap.add_argument("--out", default="results/figures/fig23_comm_degradation.png")
    args = ap.parse_args()
    if args.steps < 1 or args.drones < 1 or any(not 0 <= x <= 1 for x in args.losses):
        ap.error("steps/drones must be positive and packet losses must be in [0, 1]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
            text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None

    def make_actor(env, s):
        if args.checkpoint:
            from swarm_sar.training.rollout import load_neural_actor
            actor = load_neural_actor(args.checkpoint)
            actor.bind_env(env)
            return actor
        return HeuristicActor(env, "auction", s)

    actor_label = f"mappo({args.checkpoint})" if args.checkpoint else "heuristic_auction"
    pl, sm, ss, rm = [], [], [], []
    for loss in args.losses:
        sis, resc = [], []
        for s in args.seeds:
            cfg = EnvConfig(num_drones=args.drones, max_steps=args.steps)
            cfg.comm.packet_loss = loss
            if loss >= 1.0:
                cfg.comm.range_m = 0.0
            env = SARSwarmEnv(cfg, seed=s)
            m = episode_metrics(run_episode(env, make_actor(env, s), seed=s), grid_size=64)
            records.append({"seed": s, "packet_loss": loss,
                            "config": cfg.model_dump(mode="json"), "metrics": m})
            sis.append(m["swarm_intelligence_score"]); resc.append(m["victims_rescued"])
        pl.append(loss); sm.append(np.mean(sis)); ss.append(np.std(sis)); rm.append(np.mean(resc))
        print(f"packet_loss={loss:.1f}  SIS={np.mean(sis):5.1f}±{np.std(sis):4.1f}  rescued={np.mean(resc):.1f}")
    plots.fig_comm_degradation(pl, sm, ss, rm, args.out)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump({"actor": actor_label, "packet_loss": pl, "sis_mean": sm,
                   "sis_std": ss, "rescued": rm, "episodes": records,
                   "provenance": {
                       "git_commit": revision, "working_tree_dirty": dirty,
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "python": sys.version, "platform": platform.platform(),
                       "command": [sys.executable, *sys.argv],
                       "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                       if args.checkpoint else None,
                       "metric": "utility-based SIS; implementation at recorded git revision",
                       "uncertainty": "population std across evaluation map seeds; not independent training seeds",
                       "intervention": "packet loss; 100% loss also sets radio range to zero, changing peer visibility"
                   }}, f, indent=2)
    print(f"[OK] saved {args.out}  (actor: {actor_label})")


if __name__ == "__main__":
    main()

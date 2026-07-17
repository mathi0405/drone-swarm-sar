#!/usr/bin/env python3
"""Train MAPPO (CTDE). Uses the lightweight built-in trainer; for large-scale
distributed runs pass --rllib to use Ray RLlib instead."""
from __future__ import annotations
import argparse
import _bootstrap  # noqa: F401

from swarm_sar.config import load_config
from swarm_sar.utils.seeding import set_global_seed
from swarm_sar.policies.base import HAS_TORCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/mappo_transformer_gnn.yaml")
    ap.add_argument("--rllib", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end validation run")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    if not HAS_TORCH:
        raise SystemExit(
            "PyTorch is not installed, so neural training is unavailable in this "
            "environment.\nInstall RL extras:  pip install -e '.[rl]'\n"
            "You can still run the full heuristic pipeline:  python scripts/run_demo.py --figures")

    cfg = load_config(args.config)
    if args.smoke:
        # A genuine end-to-end check: one small update, no lengthy behaviour
        # cloning or curriculum phase.  This is intentionally independent of
        # the selected production config.
        cfg.train.total_timesteps = 256
        cfg.train.rollout_len = 32
        cfg.train.num_envs = 2
        cfg.train.epochs = 1
        cfg.train.minibatches = 1
        cfg.train.bc_warmstart_episodes = 0
        cfg.train.curriculum = False
        cfg.env.max_steps = 32
        cfg.env.world.size = 24
        cfg.env.world.n_victims = 1
        cfg.env.faults.enable = False
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps
    set_global_seed(cfg.train.seed)

    if args.rllib:
        from swarm_sar.training.rllib_trainer import train as rllib_train
        rllib_train(cfg, iterations=cfg.train.total_timesteps // (cfg.train.rollout_len * cfg.train.num_envs))
        return

    from swarm_sar.training.mappo import MAPPOTrainer
    from swarm_sar.logging_utils import RunManager, ExperimentLogger
    rm = RunManager(cfg.log.out_dir, cfg.log.experiment_name)
    logger = ExperimentLogger(rm.dir, use_tensorboard=cfg.log.log_tensorboard)
    logger.save_config(cfg)
    trainer = MAPPOTrainer(cfg)
    trainer.train(logger=logger)
    trainer.save(rm.path("checkpoints", "final.pt"))
    logger.close()
    print(f"[OK] training complete -> {rm.dir}")


if __name__ == "__main__":
    main()

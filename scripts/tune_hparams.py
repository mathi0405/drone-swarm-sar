#!/usr/bin/env python3
"""Hyperparameter search for MAPPO with Optuna.

Each trial runs a short (budget-capped) training and scores the resulting
policy with the trainer's deterministic evaluation on the base environment.
Results land in a study JSON plus a ready-to-train YAML with the best
parameters merged in.

    pip install -e ".[tune]"
    python scripts/tune_hparams.py --config configs/training/mappo_transformer_gnn.yaml \
        --trials 20 --timesteps 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from swarm_sar.config import load_config
from swarm_sar.utils.seeding import set_global_seed


def build_objective(args):
    from swarm_sar.training.mappo import MAPPOTrainer

    def objective(trial):
        cfg = load_config(args.config)
        cfg.train.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        cfg.train.entropy_coef = trial.suggest_float("entropy_coef", 1e-3, 3e-2, log=True)
        cfg.train.clip = trial.suggest_float("clip", 0.1, 0.3)
        cfg.train.minibatches = trial.suggest_categorical("minibatches", [2, 4, 8])
        cfg.train.epochs = trial.suggest_int("epochs", 2, 6)
        cfg.model.hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        cfg.train.total_timesteps = args.timesteps
        cfg.train.num_envs = args.num_envs
        cfg.train.curriculum = False           # fixed difficulty keeps trials comparable
        cfg.train.bc_warmstart_episodes = 0
        cfg.train.seed = args.seed
        set_global_seed(args.seed)

        trainer = MAPPOTrainer(cfg)
        try:
            trainer.train(logger=None)
            stats = trainer.evaluate_policy(episodes=args.eval_episodes,
                                            seed=args.seed + 50_000)
        finally:
            if trainer.vec_env is not None:
                trainer.vec_env.close()
        trial.set_user_attr("eval_coverage", stats["eval_coverage"])
        trial.set_user_attr("eval_return", stats["eval_return"])
        # Rescue rate is the mission objective; coverage breaks ties early in
        # training when nothing gets rescued yet.
        return stats["eval_rescue_rate"] + 0.2 * stats["eval_coverage"]

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search over MAPPO hyperparameters.")
    parser.add_argument("--config", default="configs/training/mappo_transformer_gnn.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="training budget per trial")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/tuning")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        raise SystemExit("Optuna is not installed. Install the tuning extras: "
                         'pip install -e ".[tune]"')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(build_objective(args), n_trials=args.trials)

    report = {
        "config": args.config,
        "timesteps_per_trial": args.timesteps,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params,
             "user_attrs": t.user_attrs, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(out / "study.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Ready-to-train YAML: the source config with the best params merged in.
    best = study.best_params
    overrides = {
        "model": {"hidden_dim": best["hidden_dim"]},
        "train": {k: best[k] for k in ("lr", "entropy_coef", "clip",
                                        "minibatches", "epochs")},
    }
    with open(args.config, encoding="utf-8-sig") as f:
        merged = yaml.safe_load(f) or {}
    for section, values in overrides.items():
        merged.setdefault(section, {}).update(values)
    with open(out / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    print(f"\n[OK] best value {study.best_value:.3f} with {study.best_params}")
    print(f"[OK] study -> {out / 'study.json'}")
    print(f"[OK] train with -> {out / 'best_config.yaml'}")


if __name__ == "__main__":
    main()

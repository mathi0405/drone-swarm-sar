#!/usr/bin/env python3
"""Train MAPPO experiments and produce reproducible evaluation artefacts.

This is the end-to-end entry point used by the Windows, PowerShell and shell
launchers.  It intentionally keeps every run self-contained: resolved config,
TensorBoard/CSV training logs, final checkpoint, held-out evaluation metrics and
the combined SIS report all land under one output directory.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from swarm_sar.config import load_config
from swarm_sar.evaluation.evaluator import Evaluator
from swarm_sar.evaluation.metrics import aggregate
from swarm_sar.evaluation.stats import bootstrap_ci, iqm
from swarm_sar.logging_utils import ExperimentLogger, RunManager
from swarm_sar.policies.base import HAS_TORCH
from swarm_sar.training.mappo import MAPPOTrainer
from swarm_sar.training.rollout import load_neural_actor
from swarm_sar.utils.seeding import set_global_seed
from swarm_sar.visualization import plots


def _configure_smoke(cfg) -> None:
    """Make a fast, real update suitable for dependency and GPU validation."""
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
    cfg.env.world.n_buildings = 2
    cfg.env.world.n_trees = 4
    cfg.env.world.n_rubble = 2
    cfg.env.world.n_dynamic_obstacles = 0
    cfg.env.world.n_no_fly_zones = 0
    cfg.env.faults.enable = False


def _safe_close(trainer: MAPPOTrainer) -> None:
    if getattr(trainer, "vec_env", None) is not None:
        trainer.vec_env.close()


def _run_one(base_cfg, arch: str, seed: int, timesteps: int | None, out_dir: Path,
             eval_seeds: list[int], smoke: bool, skip_eval: bool) -> dict:
    cfg = base_cfg.model_copy(deep=True)
    cfg.model.arch = arch
    cfg.train.seed = seed
    if timesteps is not None:
        cfg.train.total_timesteps = timesteps
    if smoke:
        _configure_smoke(cfg)
    cfg.log.out_dir = str(out_dir)
    cfg.log.experiment_name = f"{arch}_s{seed}"
    set_global_seed(seed)

    manager = RunManager(cfg.log.out_dir, cfg.log.experiment_name)
    logger = ExperimentLogger(manager.dir, use_tensorboard=cfg.log.log_tensorboard)
    logger.save_config(cfg)
    trainer = MAPPOTrainer(cfg)
    checkpoint = manager.path("checkpoints", "final.pt")
    try:
        trainer.train(logger=logger)
        trainer.save(checkpoint)
    finally:
        _safe_close(trainer)
        logger.close()

    record = {
        "architecture": arch,
        "seed": seed,
        "run_dir": str(manager.dir),
        "checkpoint": str(checkpoint),
        "scalars": str(manager.path("scalars.csv")),
    }
    if skip_eval:
        return record

    # Preserve the benchmark configuration rather than evaluating only the
    # final curriculum stage mutated inside the trainer.
    def actor_factory(env, _ckpt=str(checkpoint)):
        return load_neural_actor(_ckpt, deterministic=True)
    evaluator = Evaluator(base_cfg.env.model_copy(deep=True), actor_factory)
    evaluation = evaluator.evaluate(eval_seeds)
    per_seed = evaluation["per_seed"]
    sis = np.asarray([m["swarm_intelligence_score"] for m in per_seed], dtype=float)
    record.update({
        "metrics": per_seed,
        "aggregate": evaluation["aggregate"],
        "sis_iqm": float(iqm(sis)),
        "sis_ci95": [float(x) for x in bootstrap_ci(sis)],
    })
    with open(manager.path("evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def _make_figures(records: list[dict], out_dir: Path) -> None:
    evaluated = [r for r in records if "aggregate" in r]
    if not evaluated:
        return
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    csv_by_arch: dict[str, list[str]] = defaultdict(list)
    metrics_by_arch: dict[str, list[dict]] = defaultdict(list)
    for record in evaluated:
        csv_by_arch[record["architecture"]].append(record["scalars"])
        metrics_by_arch[record["architecture"]].extend(record["metrics"])
    aggregate_by_arch = {arch: aggregate(metrics) for arch, metrics in metrics_by_arch.items()}
    plots.fig_reward_curves_from_csv(
        dict(csv_by_arch), figures / "fig08_reward_curves_REAL.png"
    )
    plots.fig_algo_comparison(aggregate_by_arch, figures / "fig20_algo_comparison_REAL.png")
    success = {
        arch: values["mission_success"]["mean"]
        for arch, values in aggregate_by_arch.items()
    }
    plots.fig_success_rate(success, figures / "fig11_success_rate_REAL.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Swarm-SAR MAPPO experiments and report SIS.")
    parser.add_argument("--config", default="configs/training/mappo_easy.yaml")
    parser.add_argument("--archs", nargs="+", default=["transformer_gnn"],
                        choices=["mlp", "gru", "gnn", "transformer", "transformer_gnn"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="results/trained")
    parser.add_argument("--smoke", action="store_true", help="one tiny train + evaluation run")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if not HAS_TORCH:
        raise SystemExit("PyTorch is required. Install the project RL extras: pip install -e '.[rl]'")
    if args.timesteps is not None and args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.smoke:
        args.archs = [args.archs[0]]
        args.seeds = [args.seeds[0]]
        args.eval_seeds = [args.eval_seeds[0]]

    base_cfg = load_config(args.config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for arch in args.archs:
        for seed in args.seeds:
            print(f"\n=== Training {arch}, seed {seed} ===")
            record = _run_one(base_cfg, arch, seed, args.timesteps, output,
                              args.eval_seeds, args.smoke, args.skip_eval)
            records.append(record)
            if "aggregate" in record:
                sis = record["aggregate"]["swarm_intelligence_score"]["mean"]
                print(f"[OK] {arch} seed {seed}: SIS={sis:.2f}")

    if not args.no_figures:
        _make_figures(records, output)
    report = {
        "config": str(args.config),
        "architectures": args.archs,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "runs": records,
    }
    report_path = output / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] completed {len(records)} run(s) -> {report_path}")


if __name__ == "__main__":
    main()

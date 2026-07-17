"""Structured experiment logging: CSV + JSON always, TensorBoard if available.

Every run gets a timestamped directory containing the resolved config, scalar
logs (CSV), and a summary JSON, so experiments are fully reproducible and easy to
post-process into the figures.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path


class RunManager:
    def __init__(self, out_dir: str, name: str = "run"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out_dir) / f"{name}_{stamp}"
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)

    def path(self, *parts) -> Path:
        return self.dir.joinpath(*parts)


class ExperimentLogger:
    def __init__(self, run_dir: str | Path, use_tensorboard: bool = True,
                 use_wandb: bool = False, wandb_project: str = "swarm-sar", config=None):
        self.dir = Path(run_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self._csv = open(self.dir / "scalars.csv", "w", newline="")
        self._writer = csv.writer(self._csv); self._header = None
        self._t0 = time.time()
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(project=wandb_project, dir=str(self.dir),
                                        config=config, reinit=True)
            except Exception:
                self.wandb = None
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.dir / "tb"))
            except Exception:
                self.tb = None

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        if self._header is None:
            self._header = ["step", "wall_s", *scalars.keys()]
            self._writer.writerow(self._header)
        else:
            # The CSV schema is fixed by the first call; surface (once) any
            # keys that would otherwise be dropped silently.
            missing = set(scalars) - set(self._header)
            if missing and not getattr(self, "_warned_missing", False):
                self._warned_missing = True
                print(f"[logger] keys not in CSV header (TensorBoard still gets them): "
                      f"{sorted(missing)}")
        self._writer.writerow([step, round(time.time() - self._t0, 2),
                               *[scalars.get(k, "") for k in self._header[2:]]])
        self._csv.flush()
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log({**scalars, "step": step})

    def save_config(self, cfg) -> None:
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        with open(self.dir / "config.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_summary(self, summary: dict) -> None:
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self):
        self._csv.close()
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()

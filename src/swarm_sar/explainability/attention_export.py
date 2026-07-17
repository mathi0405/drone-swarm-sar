"""Export attention/communication weights for dashboard replay."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def export_attention_weights(weights, agents: list[str] | None = None) -> dict:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError("attention weights must be a 2-D matrix")
    agents = agents or [f"drone_{i}" for i in range(arr.shape[0])]
    links = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0:
                links.append({"source": agents[j], "target": agents[i], "weight": float(arr[i, j])})
    return {"agents": agents, "links": links}


def save_explanations(path: str | Path, traces: Iterable) -> None:
    data = [t.to_dict() if hasattr(t, "to_dict") else t for t in traces]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

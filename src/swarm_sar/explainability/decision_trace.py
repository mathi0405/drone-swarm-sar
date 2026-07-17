"""Human-readable decision traces for SAR policy actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecisionTrace:
    step: int
    agent: str
    action: str
    reason: str
    contributors: list[dict[str, Any]] = field(default_factory=list)
    attention: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def victim_priority(drone_pos: np.ndarray, victim_pos: np.ndarray, severity: float,
                    battery_soc: float = 1.0) -> float:
    """Higher score means a victim is a better immediate target."""
    dist = float(np.linalg.norm(np.asarray(drone_pos) - np.asarray(victim_pos)))
    return float(severity * 2.0 + battery_soc - 0.02 * dist)


def explain_action(agent: str, action: str, obs: np.ndarray | None = None,
                   step: int = 0, attention: dict[str, Any] | None = None,
                   context: dict[str, Any] | None = None) -> DecisionTrace:
    """Build a compact explanation from observation magnitudes and optional context."""
    contributors: list[dict[str, Any]] = []
    if obs is not None and len(obs):
        arr = np.asarray(obs, dtype=float)
        top = np.argsort(np.abs(arr))[-5:][::-1]
        contributors = [
            {"feature": int(i), "value": float(arr[int(i)])}
            for i in top
        ]
    reason = "policy_selected_highest_scoring_action"
    if context:
        if context.get("returning"):
            reason = "battery_or_return_constraint"
        elif context.get("known_victim"):
            reason = "victim_priority"
        elif context.get("frontier"):
            reason = "frontier_exploration"
    return DecisionTrace(
        step=step,
        agent=agent,
        action=action,
        reason=reason,
        contributors=contributors,
        attention=attention or {},
    )

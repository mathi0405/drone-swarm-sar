"""Decentralized communication with limited range, latency and packet loss.

Drones broadcast typed messages (victim / coverage / battery / target /
heartbeat). Delivery is only to peers within range, is delayed by `latency_steps`
and dropped with probability `packet_loss`. This module is the substrate for the
research question on how communication quality shapes swarm intelligence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from swarm_sar.config import CommConfig


@dataclass
class Message:
    sender: int
    mtype: str
    payload: Any
    step: int
    priority: int = 0
    size_bytes: int = 0
    ttl: int = 20


def message_priority(msg: Message, cfg: CommConfig) -> int:
    """Return explicit priority, then the configured per-type priority, then a
    built-in default (victim > hazard > map)."""
    if msg.priority:
        return int(msg.priority)
    if msg.mtype in cfg.priority_by_type:
        return int(cfg.priority_by_type[msg.mtype])
    return {"victim": 10, "hazard": 5, "map": 1}.get(msg.mtype, 0)


def compress_payload(payload: Any, max_cells: int = 40) -> Any:
    """Summarize large payloads while preserving critical SAR fields."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    cells = out.get("explored_cells")
    if isinstance(cells, list) and len(cells) > max_cells:
        out["explored_cells"] = cells[-max_cells:]
        out["explored_cells_dropped"] = len(cells) - max_cells
    victims = out.get("victims")
    if isinstance(victims, list):
        out["victim_count"] = len(victims)
    return out


def should_event_broadcast(event: str, severity: float = 1.0,
                           threshold: float = 0.5) -> bool:
    """Small event-driven gate used by planners and tests."""
    if event in {"victim", "target", "battery_critical", "fault"}:
        return True
    return severity >= threshold


class CommNetwork:
    def __init__(self, cfg: CommConfig, n_agents: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = n_agents
        # queue of (deliver_step, -priority, sequence, receiver, Message)
        self._inflight: List[Tuple[int, int, int, int, Message]] = []
        self.delivered: Dict[int, List[Message]] = {i: [] for i in range(n_agents)}
        self.n_sent = 0
        self.n_delivered = 0
        self.edges_this_step: List[Tuple[int, int]] = []
        self._seq = 0
        self.congestion = 0.0

    def reset(self) -> None:
        self._inflight.clear()
        self.delivered = {i: [] for i in range(self.n)}
        self.n_sent = self.n_delivered = 0
        self.edges_this_step = []
        self._seq = 0
        self.congestion = 0.0

    def broadcast(self, sender: int, positions: Dict[int, np.ndarray],
                  msg: Message, alive: Dict[int, bool], comm_ok: Dict[int, bool]) -> None:
        """Enqueue a message to every in-range, reachable peer."""
        if not comm_ok.get(sender, True):
            return
        msg = Message(
            sender=msg.sender,
            mtype=msg.mtype,
            payload=compress_payload(msg.payload, self.cfg.compress_payload_cells),
            step=msg.step,
            priority=message_priority(msg, self.cfg),
            size_bytes=msg.size_bytes,
        )
        sp = positions[sender]
        for r in range(self.n):
            if r == sender or not alive.get(r, False) or not comm_ok.get(r, True):
                continue
            d = np.linalg.norm(sp - positions[r])
            if d <= self.cfg.range_m:
                self.edges_this_step.append((sender, r))
                self.n_sent += 1
                dynamic_packet_loss = self.cfg.packet_loss + (self.congestion * 0.5)
                if self.rng.random() < dynamic_packet_loss:
                    continue                          # dropped
                deliver = msg.step + max(0, self.cfg.latency_steps)
                self._seq += 1
                self._inflight.append((deliver, -msg.priority, self._seq, r, msg))

    def step(self, now: int) -> None:
        """Deliver messages whose latency has elapsed (respecting bandwidth).

        Note: ``edges_this_step`` is cleared by the environment at the start of
        each step (before broadcasts), not here, so the frame logger can capture
        the links formed this step.
        """
        self.delivered = {i: [] for i in range(self.n)}
        remaining: List[Tuple[int, int, int, int, Message]] = []
        counts = {i: 0 for i in range(self.n)}
        ready: List[Tuple[int, int, int, int, Message]] = []
        for item in self._inflight:
            if item[0] <= now:
                ready.append(item)
            else:
                remaining.append(item)
        # Filter out expired messages
        ready = [item for item in ready if now - item[4].step <= item[4].ttl]
        # Sort by priority (neg_priority ascending means highest priority first), then deliver time, then seq
        ready.sort(key=lambda item: (item[1], item[0], item[2]))
        dropped_for_bandwidth = 0
        for deliver, neg_priority, seq, r, msg in ready:
            if deliver <= now:
                if counts[r] < self._effective_bandwidth():
                    self.delivered[r].append(msg)
                    counts[r] += 1
                    self.n_delivered += 1
                # over-bandwidth messages are dropped (congestion)
                else:
                    dropped_for_bandwidth += 1
        self._inflight = remaining
        attempted = max(1, len(ready))
        self.congestion = dropped_for_bandwidth / attempted

    def _effective_bandwidth(self) -> int:
        if self.cfg.adaptive_bandwidth and self.congestion > 0.5:
            return max(self.cfg.min_bandwidth_msgs, self.cfg.bandwidth_msgs - 1)
        return self.cfg.bandwidth_msgs

    def inbox(self, agent: int) -> List[Message]:
        return self.delivered.get(agent, [])

    def connected_components(self, positions: Dict[int, np.ndarray],
                             alive: Dict[int, bool], comm_ok: Dict[int, bool]) -> List[List[int]]:
        """Return communication components under current range/fault constraints."""
        seen = set()
        comps: List[List[int]] = []
        for start in range(self.n):
            if start in seen or not alive.get(start, False) or not comm_ok.get(start, True):
                continue
            stack = [start]
            comp = []
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for other in range(self.n):
                    if other in seen or not alive.get(other, False) or not comm_ok.get(other, True):
                        continue
                    if np.linalg.norm(positions[cur] - positions[other]) <= self.cfg.range_m:
                        seen.add(other)
                        stack.append(other)
            comps.append(comp)
        return comps

    @property
    def delivery_ratio(self) -> float:
        return self.n_delivered / max(1, self.n_sent)

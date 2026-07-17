import numpy as np
from swarm_sar.config import CommConfig
from swarm_sar.communication.comms import CommNetwork, Message, compress_payload


def _pos():
    return {0: np.array([0.0, 0.0]), 1: np.array([5.0, 0.0]), 2: np.array([100.0, 0.0])}


def test_range_limits_delivery():
    cn = CommNetwork(CommConfig(range_m=30, packet_loss=0.0, latency_steps=0), 3, np.random.default_rng(0))
    alive = {0: True, 1: True, 2: True}; ok = {0: True, 1: True, 2: True}
    cn.broadcast(0, _pos(), Message(0, "victim", {}, 0), alive, ok)
    cn.step(0)
    assert len(cn.inbox(1)) == 1          # in range
    assert len(cn.inbox(2)) == 0          # out of range


def test_total_packet_loss():
    cn = CommNetwork(CommConfig(range_m=30, packet_loss=1.0, latency_steps=0), 3, np.random.default_rng(0))
    alive = {i: True for i in range(3)}; ok = {i: True for i in range(3)}
    cn.broadcast(0, _pos(), Message(0, "victim", {}, 0), alive, ok)
    cn.step(0)
    assert cn.n_delivered == 0


def test_latency_delays_delivery():
    cn = CommNetwork(CommConfig(range_m=30, packet_loss=0.0, latency_steps=2), 3, np.random.default_rng(0))
    alive = {i: True for i in range(3)}; ok = {i: True for i in range(3)}
    cn.broadcast(0, _pos(), Message(0, "victim", {}, 0), alive, ok)
    cn.step(0); assert len(cn.inbox(1)) == 0     # not yet
    cn.step(2); assert len(cn.inbox(1)) == 1     # arrived


def test_delivery_ratio_bounded():
    cn = CommNetwork(CommConfig(range_m=30, packet_loss=0.3), 3, np.random.default_rng(0))
    alive = {i: True for i in range(3)}; ok = {i: True for i in range(3)}
    for t in range(20):
        cn.broadcast(0, _pos(), Message(0, "victim", {}, t), alive, ok)
        cn.step(t)
    assert 0.0 <= cn.delivery_ratio <= 1.0


def test_high_priority_message_survives_bandwidth_pressure():
    cfg = CommConfig(range_m=30, packet_loss=0.0, latency_steps=0, bandwidth_msgs=1)
    cn = CommNetwork(cfg, 2, np.random.default_rng(0))
    alive = {0: True, 1: True}; ok = {0: True, 1: True}
    cn.broadcast(0, _pos(), Message(0, "coverage", {"id": "low"}, 0, priority=1), alive, ok)
    cn.broadcast(0, _pos(), Message(0, "victim", {"id": "high"}, 0, priority=100), alive, ok)

    cn.step(0)

    assert len(cn.inbox(1)) == 1
    assert cn.inbox(1)[0].payload["id"] == "high"


def test_compression_reduces_payload_size():
    payload = {"explored_cells": [[i, i] for i in range(100)], "victims": [(0, [1, 2])]}

    compressed = compress_payload(payload, max_cells=10)

    assert len(compressed["explored_cells"]) == 10
    assert compressed["explored_cells_dropped"] == 90

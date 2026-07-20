"""Tests for the utility-based SIS communication component.

The 3M campaign showed the old activity signal (delivery ratio) rewarded
chattiness: silent seeds rescued the most victims yet scored worst on SIS in
both architectures.  The component now measures outcome-linked utility:
comm-assisted rescues and coverage uniqueness.
"""
import numpy as np

from swarm_sar.config import CommConfig, EnvConfig, FaultConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import episode_metrics, swarm_intelligence_score


def _env():
    cfg = EnvConfig(
        num_drones=2, max_steps=50,
        world=WorldConfig(size=24, n_victims=2, n_buildings=4, n_trees=4,
                          n_rubble=4, n_fire_zones=1, n_no_fly_zones=0,
                          n_dynamic_obstacles=0),
        faults=FaultConfig(enable=False),
        comm=CommConfig(latency_steps=0, packet_loss=0.0, auto_broadcast=False),
    )
    env = SARSwarmEnv(cfg, seed=3)
    env.reset(seed=3)
    return env


class _FakeLog:
    def __init__(self, summary, n_drones=3, steps=5):
        self.summary = summary
        self.frames = [{"pos": [[1.0 * i, 1.0 * t] for i in range(n_drones)],
                        "alive": [True] * n_drones} for t in range(steps)]
        self.victims = [{"rescued": True, "rescued_step": 10}]


def _summary(**over):
    base = {
        "steps": 100, "coverage": 0.7, "victims_total": 8, "victims_detected": 6,
        "victims_rescued": 6, "collisions": 1, "comm_sent": 0, "comm_delivered": 0,
        "delivery_ratio": 0.0, "energy_wh": 100.0, "energy_budget_wh": 270.0,
        "success": False, "comm_assisted_rescues": 0, "coverage_uniqueness": 0.0,
    }
    base.update(over)
    return base


def test_comm_assisted_rescue_requires_radio_first_knowledge():
    env = _env()
    v = env.world.victims[0]
    v.detected = True
    v.detected_by = 1
    env._victims_known[1].add(v.idx)
    env.drones[0].kinematics.pos = env.drones[1].kinematics.pos + np.array([1.0, 0.0])

    env._handle_broadcast(1)
    env.comm.step(env.t)
    env._merge_inbox(0)
    assert 0 in env._comm_informed.get(v.idx, set())     # radio-first for d0
    assert 1 not in env._comm_informed.get(v.idx, set())  # detector: first-hand

    v.rescued = True
    v.rescued_by = 0
    assert env.episode_summary()["comm_assisted_rescues"] == 1

    v.rescued_by = 1                                     # detector rescues solo
    assert env.episode_summary()["comm_assisted_rescues"] == 0


def test_first_hand_knowledge_beats_later_message():
    env = _env()
    v = env.world.victims[0]
    env._victims_known[0].add(v.idx)                     # d0 saw it itself
    env._victims_known[1].add(v.idx)
    v.detected = True
    env.drones[1].kinematics.pos = env.drones[0].kinematics.pos + np.array([1.0, 0.0])
    env._handle_broadcast(1)
    env.comm.step(env.t)
    env._merge_inbox(0)                                  # message arrives later
    assert 0 not in env._comm_informed.get(v.idx, set())


def test_coverage_uniqueness_bounds():
    env = _env()
    env.explored[0][:] = False
    env.explored[1][:] = False
    env.explored[0][0:5, 0:5] = True                     # identical footprints
    env.explored[1][0:5, 0:5] = True
    env.global_explored[:] = False
    env.global_explored[0:5, 0:5] = True
    assert env._coverage_uniqueness() < 1e-9

    env.explored[1][0:5, 0:5] = False                    # disjoint footprints
    env.explored[1][10:15, 10:15] = True
    env.global_explored[10:15, 10:15] = True
    assert env._coverage_uniqueness() > 0.999


def test_sis_no_longer_crushed_by_silence():
    silent_coordinated = episode_metrics(
        _FakeLog(_summary(coverage_uniqueness=0.8)), grid_size=24)
    silent_chaotic = episode_metrics(
        _FakeLog(_summary(coverage_uniqueness=0.0)), grid_size=24)
    assert silent_coordinated["communication_utility"] == 0.4
    assert silent_coordinated["swarm_intelligence_score"] > \
        silent_chaotic["swarm_intelligence_score"]
    # The old metric clipped a silent swarm's comm term to 1e-4, multiplying
    # total SIS by ~0.4; utility keeps a well-partitioned silent swarm sane.
    old_style = swarm_intelligence_score({
        "coverage": 0.7, "rescue": 6 / 8, "energy": 1 - 100 / 270,
        "communication": 0.0, "safety": 1 - (1 / 100) / 0.20})
    assert silent_coordinated["swarm_intelligence_score"] > old_style * 1.5


def test_assisted_rescues_raise_utility():
    helped = episode_metrics(
        _FakeLog(_summary(comm_assisted_rescues=3, coverage_uniqueness=0.6)),
        grid_size=24)
    unhelped = episode_metrics(
        _FakeLog(_summary(comm_assisted_rescues=0, coverage_uniqueness=0.6)),
        grid_size=24)
    assert helped["communication_utility"] > unhelped["communication_utility"]
    assert helped["communication_utility"] == 0.5 * (3 / 6) + 0.5 * 0.6


def test_legacy_log_falls_back_to_activity():
    s = _summary(delivery_ratio=0.35)
    del s["comm_assisted_rescues"]
    del s["coverage_uniqueness"]
    m = episode_metrics(_FakeLog(s), grid_size=24)
    assert m["communication_utility"] == 0.35

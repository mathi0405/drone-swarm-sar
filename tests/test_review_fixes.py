"""Regression tests for the external-review fixes.

Covers: metre↔cell unit conversion (comm range, movement), the peer-intent
observation leak, camera-fault sensing gates, battery/energy accounting,
swept-path collision enforcement, and checkpoint metadata handling.
"""
import numpy as np

from swarm_sar.battery.battery import Battery
from swarm_sar.communication.comms import CommNetwork, Message
from swarm_sar.config import (
    BatteryConfig,
    CommConfig,
    Config,
    EnvConfig,
    FaultConfig,
    WorldConfig,
)
from swarm_sar.drone.dynamics import PointMassDynamics
from swarm_sar.environment.entities import CellType
from swarm_sar.environment.observation import frame_layout
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.environment.world import SARWorld


def _small_env(**comm_kwargs) -> SARSwarmEnv:
    cfg = EnvConfig(
        num_drones=2, max_steps=50,
        world=WorldConfig(size=24, n_victims=2, n_buildings=4, n_trees=4,
                          n_rubble=4, n_fire_zones=1, n_no_fly_zones=0,
                          n_dynamic_obstacles=0),
        faults=FaultConfig(enable=False),
        comm=CommConfig(latency_steps=0, packet_loss=0.0, auto_broadcast=False,
                        **comm_kwargs),
    )
    env = SARSwarmEnv(cfg, seed=3)
    env.reset(seed=3)
    return env


def _newest_frame(env, obs_vec):
    fd = env._compute_obs_dim()
    return obs_vec[-fd:]


def _segment(env, frame, name):
    layout = frame_layout(env.cfg.k_nearest_drones, env.cfg.obs_map_patch)
    off = 0
    for seg, width in layout.items():
        if seg == name:
            return frame[off:off + width]
        off += width
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Units (#3)                                                                  #
# --------------------------------------------------------------------------- #
def test_comm_range_is_converted_to_cells():
    """range_m=20 with 2 m cells is 10 CELLS, not 20."""
    cfg = CommConfig(range_m=20.0, latency_steps=0, packet_loss=0.0)
    rng = np.random.default_rng(0)
    net = CommNetwork(cfg, 2, rng, cell_size_m=2.0)
    alive = {0: True, 1: True}
    ok = {0: True, 1: True}

    # 12 cells apart = 24 m: out of the 20 m range (pre-fix: "in range").
    positions = {0: np.array([0.0, 0.0]), 1: np.array([12.0, 0.0])}
    net.broadcast(0, positions, Message(0, "victim", {}, step=0), alive, ok)
    net.step(0)
    assert net.inbox(1) == []

    # 8 cells apart = 16 m: within range.
    positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    net.broadcast(0, positions, Message(0, "victim", {}, step=1), alive, ok)
    net.step(1)
    assert len(net.inbox(1)) == 1


def test_movement_displacement_is_metres_over_cells():
    """6 m/s for one 1-s kinematic step on 2 m cells moves 3 cells, not 6."""
    dyn = PointMassDynamics(max_speed=6.0, motion_noise_std=0.0, cell_size_m=2.0)
    rng = np.random.default_rng(0)
    pos, vel = dyn.step(np.zeros(2), np.zeros(2), np.array([6.0, 0.0]),
                        np.zeros(2), rng)
    assert np.allclose(pos, [3.0, 0.0])
    assert np.allclose(vel, [6.0, 0.0])          # velocity itself stays metric


# --------------------------------------------------------------------------- #
# Peer-intent leak (#1)                                                       #
# --------------------------------------------------------------------------- #
def test_peer_intent_requires_delivered_message():
    env = _small_env()
    env.drones[0].kinematics.pos = np.array([5.0, 5.0])
    env.drones[1].kinematics.pos = np.array([7.0, 5.0])
    v = env.world.victims[0]
    v.detected = True
    v.detected_by = 1
    v.assigned_drone = 1

    # World has an assignment, but no message was ever delivered: the peer
    # block must carry NO intent (pre-fix it teleported through the obs).
    obs, _ = env.obs_engine.get_obs(env)
    peers = _segment(env, _newest_frame(env, obs["drone_0"]), "peers")
    assert peers[2] == 0.0 and peers[3] == 0.0   # target rel
    assert peers[4] == 0.0                       # has_assignment
    assert peers[5] == 0.0                       # mode
    assert peers[0] != 0.0 or peers[1] != 0.0    # proximity sensing still on

    # After drone 1 BROADCASTS and the message is delivered, intent appears.
    env._victims_known[1].add(v.idx)
    env._handle_broadcast(1)
    env.comm.step(env.t)
    env._merge_inbox(0)
    obs, _ = env.obs_engine.get_obs(env)
    peers = _segment(env, _newest_frame(env, obs["drone_0"]), "peers")
    assert peers[4] == 1.0                       # has_assignment (received)
    assert peers[5] == 1.0                       # mode = rescue
    assert peers[2] != 0.0 or peers[3] != 0.0    # target direction


def test_broadcast_shares_only_known_victims():
    env = _small_env()
    v0, v1 = env.world.victims[0], env.world.victims[1]
    v0.detected = v1.detected = True
    env._victims_known[0].add(v0.idx)            # drone 0 knows only v0
    env.drones[1].kinematics.pos = env.drones[0].kinematics.pos + np.array([1.0, 0.0])
    env._handle_broadcast(0)
    env.comm.step(env.t)
    got = [vid for m in env.comm.inbox(1) for vid, _ in m.payload.get("victims", [])]
    assert v0.idx in got and v1.idx not in got


# --------------------------------------------------------------------------- #
# Camera fault (#5)                                                           #
# --------------------------------------------------------------------------- #
def test_camera_fault_blinds_exploration_and_hazards():
    env = _small_env()
    d = env.drones[0]
    d.faults.camera_ok = False
    before = int(env.explored[0].sum())
    assert env._sense_one(0) == (0, 0, 0, 0)
    assert int(env.explored[0].sum()) == before

    obs, _ = env.obs_engine.get_obs(env)
    hazards = _segment(env, _newest_frame(env, obs["drone_0"]), "hazards")
    assert list(hazards) == [1.0, 1.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Battery / energy (#6)                                                       #
# --------------------------------------------------------------------------- #
def test_soc_never_exceeds_one_after_degrading_charge():
    cfg = BatteryConfig(capacity_wh=10.0, charge_rate_w=36000.0,
                        degradation_per_cycle=0.1)
    b = Battery(cfg, dt=1.0)
    b.drain(5.0)
    for _ in range(10):                          # charge to full -> cycle -> fade
        b.charge()
        assert b.soc <= 1.0 + 1e-9
    assert b.capacity < cfg.capacity_wh          # degradation actually happened


def test_energy_metric_counts_consumption_not_net_depletion():
    cfg = BatteryConfig(capacity_wh=10.0, charge_rate_w=36000.0)
    b = Battery(cfg, dt=1.0)
    b.drain(4.0)
    b.charge()                                   # back to (almost) full
    assert b.drawn_wh >= 4.0 - 1e-9              # consumption is remembered
    assert (b.capacity - b.energy) < 0.5         # net depletion would hide it


# --------------------------------------------------------------------------- #
# Swept collision (#4)                                                        #
# --------------------------------------------------------------------------- #
def test_clip_segment_stops_at_wall():
    env = _small_env()
    w: SARWorld = env.world
    w.grid[:] = CellType.FREE
    w.grid[2, 4] = CellType.BUILDING             # wall at (x=4, y=2)
    env._refresh_world_masks()
    reached = w.clip_segment(np.array([2.0, 2.0]), np.array([8.0, 2.0]))
    assert reached[0] < 4.0                      # never crosses the building


def test_drone_apply_cannot_tunnel():
    env = _small_env()
    w = env.world
    w.grid[:] = CellType.FREE
    w.grid[2, 4] = CellType.BUILDING
    env._refresh_world_masks()
    d = env.drones[0]
    d.kinematics.pos = np.array([2.0, 2.0])
    d.kinematics.vel = np.zeros(2)
    d.dyn.max_speed = 24.0                       # 12 cells/step: must still stop
    d.dyn.motion_noise_std = 0.0
    d.cfg.max_speed_mps = 24.0
    info = d.apply(3, w, np.zeros(2))            # 3 = east
    assert info["blocked"]
    assert d.kinematics.pos[0] < 4.0


# --------------------------------------------------------------------------- #
# Checkpoints / training guards (#7, #9)                                      #
# --------------------------------------------------------------------------- #
def test_gnn_independent_policy_rejected():
    from swarm_sar.policies.base import HAS_TORCH
    if not HAS_TORCH:
        return
    from swarm_sar.training.mappo import MAPPOTrainer
    cfg = Config()
    cfg.model.arch = "gnn"
    cfg.train.policy_sharing = "independent"
    try:
        MAPPOTrainer(cfg)
        raise AssertionError("should have rejected gnn + independent")
    except ValueError as e:
        assert "independent" in str(e)


def test_independent_checkpoint_loads_by_agent():
    from swarm_sar.policies.base import HAS_TORCH
    if not HAS_TORCH:
        return
    import torch

    from swarm_sar.config import ModelConfig
    from swarm_sar.policies.base import PolicySpec, build_policy, load_checkpoint

    spec = PolicySpec(obs_dim=8 * 32, n_actions=8, global_dim=16, n_agents=2)
    model = build_policy(ModelConfig(arch="mlp", hidden_dim=32), spec,
                         centralized=False)
    import os
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "indep.pt")
    torch.save({"model_cfg": ModelConfig(arch="mlp", hidden_dim=32),
                "spec": spec.__dict__, "policy_sharing": "independent",
                "centralized_critic": False,
                "state_dict_0": model.state_dict()}, path)
    loaded = load_checkpoint(path, agent=0)      # pre-fix: KeyError
    assert loaded is not None
    try:
        load_checkpoint(path, agent=3)
        raise AssertionError("missing agent index should raise")
    except KeyError:
        pass

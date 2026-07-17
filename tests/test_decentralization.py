"""Tests for the corrected decentralized observation & reward-mode ablation."""
import numpy as np
from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv


def test_observation_is_local_not_global():
    # With comms disabled, agent i's 'found' feature must reflect ITS OWN belief,
    # not the swarm's global detections (no oracle leak).
    cfg = EnvConfig(num_drones=3, max_steps=1)
    cfg.comm.range_m = 0.0
    env = SARSwarmEnv(cfg, seed=0)
    env.reset(seed=0)
    env.victim_belief[0][:] = 0.0
    env.victim_belief[1][:] = 0.0
    v = env.world.victims[0]
    env.victim_belief[0][int(v.pos[1]), int(v.pos[0])] = 1.0   # only agent 0 "knows"
    o0 = env._obs(0)
    o1 = env._obs(1)
    assert o0[-2] > 0.0        # agent 0 believes it found a victim
    assert o1[-2] == 0.0       # agent 1 does NOT (would be >0 if obs leaked global truth)


def test_peers_hidden_beyond_comm_range():
    cfg = EnvConfig(num_drones=3, max_steps=1)
    cfg.comm.range_m = 0.0     # nobody is in range -> no peer info
    env = SARSwarmEnv(cfg, seed=1)
    env.reset(seed=1)
    o = env._obs(0)
    # peer block is 4*k floats right after the 9 own-state features
    k = cfg.k_nearest_drones
    peer_block = o[9:9 + 4 * k]
    assert np.allclose(peer_block, 0.0)   # no observable peers without comms


def test_sparse_reward_removes_shaping():
    def hover_return(mode):
        cfg = EnvConfig(num_drones=1, max_steps=10)
        cfg.reward.mode = mode
        cfg.faults.enable = False
        env = SARSwarmEnv(cfg, seed=0)
        env.reset(seed=0)
        tot = 0.0
        while env.agents:
            _, r, *_ = env.step({a: 0 for a in env.agents})   # hover only
            tot += sum(r.values())
        return tot
    shaped = hover_return("shaped")
    sparse = hover_return("sparse")
    assert sparse > shaped               # sparse omits time/idle shaping penalties
    assert abs(sparse) < 1e-6            # pure hover with no events -> zero reward

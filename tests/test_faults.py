import numpy as np
from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv


def test_faults_reproducible_and_logged():
    def run():
        cfg = EnvConfig(num_drones=4, max_steps=40)
        cfg.faults.drone_failure_prob = 0.05      # crank up so events happen
        env = SARSwarmEnv(cfg, seed=3); env.reset(seed=3)
        rng = np.random.default_rng(0)
        while env.agents:
            env.step({a: int(rng.integers(env.n_actions)) for a in env.agents})
        return env.faults.summary()
    assert run() == run()                          # deterministic given seed


def test_faults_disabled():
    cfg = EnvConfig(num_drones=3, max_steps=30)
    cfg.faults.enable = False
    env = SARSwarmEnv(cfg, seed=0); env.reset(seed=0)
    rng = np.random.default_rng(0)
    while env.agents:
        env.step({a: int(rng.integers(env.n_actions)) for a in env.agents})
    assert env.faults.summary() == {}

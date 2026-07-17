"""Distributed multi-agent PPO via Ray RLlib.

Wraps :class:`SARSwarmEnv` as an RLlib ``MultiAgentEnv`` and configures PPO with
a single *shared* policy (parameter sharing). Note this path is plain
independent PPO with shared weights — the centralized-critic (CTDE) objective
is implemented in :mod:`swarm_sar.training.mappo`. Use RLlib for large runs /
the 10-drone configuration when horizontal scaling matters more than the
centralized value function.
"""
from __future__ import annotations
from typing import Optional

from swarm_sar.config import Config

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    import gymnasium as gym
    _HAS_RLLIB = True
except Exception:
    _HAS_RLLIB = False
    MultiAgentEnv = object  # type: ignore


if _HAS_RLLIB:

    class RLlibSAREnv(MultiAgentEnv):
        def __init__(self, env_config):
            super().__init__()
            from swarm_sar.config import EnvConfig
            from swarm_sar.environment.sar_env import SARSwarmEnv
            self.env = SARSwarmEnv(EnvConfig(**env_config.get("env", {})),
                                   seed=env_config.get("seed"))
            self._agent_ids = set(self.env.possible_agents)
            self.observation_space = gym.spaces.Box(-10, 10, (self.env.obs_dim,))
            self.action_space = gym.spaces.Discrete(self.env.n_actions)

        def reset(self, *, seed=None, options=None):
            return self.env.reset(seed=seed)

        def step(self, actions):
            obs, rew, term, trunc, info = self.env.step(actions)
            term["__all__"] = all(term.values()) if term else True
            trunc["__all__"] = all(trunc.values()) if trunc else False
            return obs, rew, term, trunc, info


def build_config(cfg: Config):
    if not _HAS_RLLIB:
        raise ImportError("Ray RLlib not installed. `pip install -e '.[rl]'`")
    tune.register_env("swarm_sar", lambda ec: RLlibSAREnv(ec))
    gpu = 0 if cfg.train.device == "cpu" else 1
    return (
        PPOConfig()
        # new API stack (RLModule + Learner), Ray >= 2.9
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        # Pass the FULL environment config: dropping it here silently trained
        # on default world/reward/comm settings regardless of the YAML.
        .environment("swarm_sar",
                     env_config={"env": cfg.env.model_dump(), "seed": cfg.train.seed})
        .framework("torch")
        .env_runners(num_env_runners=cfg.train.num_envs)
        .training(gamma=cfg.train.gamma, lr=cfg.train.lr, lambda_=cfg.train.gae_lambda,
                  clip_param=cfg.train.clip, entropy_coeff=cfg.train.entropy_coef,
                  train_batch_size=cfg.train.rollout_len * cfg.train.num_envs)
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda aid, *a, **k: "shared",   # parameter sharing (CTDE)
        )
        .learners(num_gpus_per_learner=gpu)
    )


def train(cfg: Config, iterations: int = 100):
    algo = build_config(cfg).build()
    for i in range(iterations):
        result = algo.train()
        if i % 10 == 0:
            rm = result.get("env_runners", result).get("episode_return_mean",
                                                        result.get("episode_reward_mean", float("nan")))
            print(f"[{i}] return={rm:.1f}")
    return algo

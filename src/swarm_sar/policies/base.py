"""Actor-Critic backbone shared by all architectures.

Each policy factors into (encoder -> actor head, centralized critic). The critic
receives the *global* state during training (CTDE); at execution time only the
per-agent encoder + actor are used, giving fully decentralized control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    if os.environ.get("IS_WORKER") == "1":
        raise Exception("Skipping torch in worker to save VRAM")
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:                     # torch is optional
    HAS_TORCH = False
    nn = object                       # type: ignore
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural policies. Install RL extras:\n"
            "  pip install -e '.[rl]'   (or)   pip install -r requirements.txt"
        )


if HAS_TORCH:

    class MLPEncoder(nn.Module):
        def __init__(self, in_dim: int, hidden: int, layers: int = 2):
            super().__init__()
            mods = []
            d = in_dim
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Tanh()]
                d = hidden
            self.net = nn.Sequential(*mods)
            self.out_dim = hidden

        def forward(self, x, graph=None):
            if x.dim() > 2:
                x = x.flatten(start_dim=1)
            return self.net(x)

    class ActorCritic(nn.Module):
        """Wraps a per-agent encoder with an actor head and a critic.

        With ``centralized=True`` (CTDE) the critic consumes the global state
        during training; with ``centralized=False`` it consumes the local
        encoding, giving fully independent learners for ablations.
        """
        def __init__(self, encoder: nn.Module, n_actions: int, global_dim: int,
                     hidden: int = 128, centralized: bool = True):
            super().__init__()
            self.encoder = encoder
            self.actor = nn.Sequential(nn.Linear(encoder.out_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, n_actions))
            critic_in = global_dim if centralized else encoder.out_dim
            self.centralized = centralized
            self.critic = nn.Sequential(nn.Linear(critic_in, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh(),
                                        nn.Linear(hidden, 1))
            # Orthogonal initialization (standard PPO stabilizer): sqrt(2) gain
            # on hidden layers, 0.01 on the actor head so the initial policy is
            # near-uniform, 1.0 on the value head.
            self._orthogonal_init(self.actor, out_gain=0.01)
            self._orthogonal_init(self.critic, out_gain=1.0)

        @staticmethod
        def _orthogonal_init(seq: nn.Sequential, out_gain: float) -> None:
            linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            for m in linears[:-1]:
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
            if linears:
                nn.init.orthogonal_(linears[-1].weight, gain=out_gain)
                nn.init.zeros_(linears[-1].bias)

        def forward(self, obs, graph=None, global_state=None, action_mask=None):
            z = self.encoder(obs) if graph is None else self.encoder(obs, graph)
            logits = self.actor(z)
            if action_mask is not None:
                logits = logits.masked_fill(action_mask <= 0, torch.finfo(logits.dtype).min)
            if self.centralized and global_state is None:
                # decentralized EXECUTION: the centralized critic is training-only,
                # so skip it (its input is the global state, not the local encoding).
                value = torch.zeros(z.shape[0], 1, device=z.device)
            else:
                v_in = global_state if self.centralized else z
                value = self.critic(v_in)
            return logits, value

        @torch.no_grad()
        def act(self, obs, graph=None, global_state=None, action_mask=None,
                deterministic: bool = False):
            logits, value = self.forward(
                obs, graph=graph, global_state=global_state, action_mask=action_mask
            )
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, -1) if deterministic else dist.sample()
            return action, dist.log_prob(action), value.squeeze(-1)


@dataclass
class PolicySpec:
    obs_dim: int
    n_actions: int
    global_dim: int
    n_agents: int


def build_policy(model_cfg, spec: PolicySpec, centralized: bool = True):
    """Factory: return an ActorCritic with the requested encoder."""
    _require_torch()
    arch = model_cfg.arch
    if arch == "mlp":
        enc = MLPEncoder(spec.obs_dim, model_cfg.hidden_dim, model_cfg.n_layers)
    elif arch == "gru":
        from swarm_sar.policies.gru_policy import GRUEncoder
        enc = GRUEncoder(spec.obs_dim, model_cfg.hidden_dim, model_cfg)
    elif arch == "gnn":
        from swarm_sar.policies.gnn_policy import GNNEncoder
        enc = GNNEncoder(spec.obs_dim, model_cfg.hidden_dim, model_cfg)
    elif arch == "transformer":
        from swarm_sar.policies.transformer_policy import TransformerEncoder
        enc = TransformerEncoder(spec.obs_dim, model_cfg.hidden_dim, model_cfg)
    elif arch == "transformer_gnn":
        from swarm_sar.policies.transformer_gnn_policy import TransformerGNNEncoder
        enc = TransformerGNNEncoder(spec.obs_dim, model_cfg.hidden_dim, model_cfg)
    else:
        raise ValueError(f"unknown arch {arch}")
    return ActorCritic(enc, spec.n_actions, spec.global_dim,
                       model_cfg.hidden_dim, centralized=centralized)


def load_checkpoint(path: str):
    _require_torch()
    from swarm_sar.config import ModelConfig
    torch.serialization.add_safe_globals([ModelConfig])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = PolicySpec(**ckpt["spec"])
    model = build_policy(ckpt["model_cfg"], spec)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

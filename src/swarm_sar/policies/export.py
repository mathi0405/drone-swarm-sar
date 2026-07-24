"""Deployment wrapper: the decentralized actor as a plain obs -> logits module.

Edge inference only needs a single drone's own observation mapped to action
logits (argmax = action). The centralized critic and the inter-agent graph are
training-only and are dropped here, which is also what makes the module
traceable to TorchScript/ONNX. Lives in the package (not in scripts/) so both
``scripts/export_model.py`` and the tests import the same object.
"""
from __future__ import annotations

from swarm_sar.policies.base import HAS_TORCH

if HAS_TORCH:
    import torch

    class ActorExport(torch.nn.Module):
        """obs ``(B, obs_dim)`` -> action logits ``(B, n_actions)``."""

        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            logits, _ = self.policy(obs)          # graph=None, global_state=None
            return logits

else:  # pragma: no cover - exercised only without torch

    class ActorExport:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to export a policy. "
                              "Install the RL extras: pip install -e '.[rl]'")

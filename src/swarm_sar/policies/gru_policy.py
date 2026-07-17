"""Recurrent (GRU) encoder over the observation history.

The standard ablation partner for the Transformer: both consume the same
flattened HISTORY_LEN-frame temporal stack, but the GRU integrates it
sequentially while the Transformer attends over it. Comparing the two answers
"does attention over history beat recurrence?" for this task. The encoder is
stateless across environment steps — recurrence runs over the frame stack
inside a single forward pass, so it drops into MAPPO without hidden-state
plumbing in the rollout buffers.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from swarm_sar.environment.observation import HISTORY_LEN


class GRUEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg):
        super().__init__()
        self.history_len = HISTORY_LEN
        self.frame_dim = in_dim // self.history_len
        self.in_proj = nn.Linear(self.frame_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=max(1, cfg.n_layers),
                          batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, obs, graph=None):
        B = obs.shape[0] if obs.dim() > 1 else 1
        x = obs.view(B, self.history_len, self.frame_dim)
        z, _ = self.gru(torch.tanh(self.in_proj(x)))
        return self.norm(z[:, -1])            # newest-frame hidden state

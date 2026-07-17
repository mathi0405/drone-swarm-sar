"""Transformer encoder over the set of observation tokens.

Each agent tokenizes its observation into (self, peers, map-summary, messages)
tokens and attends over them. Self-attention gives a permutation-invariant,
variable-cardinality treatment of neighbours and received messages, and yields
the attention maps visualized in Figure 15.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from swarm_sar.environment.observation import HISTORY_LEN


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg):
        super().__init__()
        # The input is a flattened temporal stack (B, HISTORY_LEN * frame_dim);
        # HISTORY_LEN is shared with ObservationEngine so the two cannot drift.
        self.history_len = HISTORY_LEN
        self.frame_dim = in_dim // self.history_len

        self.in_proj = nn.Linear(self.frame_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        # Learned positional embedding: without it, self-attention treats the
        # history frames as an unordered set and cannot represent temporal
        # patterns (e.g. approaching vs. leaving a victim).
        self.pos = nn.Parameter(torch.zeros(1, self.history_len + 1, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=cfg.n_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_dim = hidden
        self.last_attn = None

    def forward(self, obs, graph=None, extract_attention=False):
        B = obs.shape[0] if obs.dim() > 1 else 1

        # Reshape into time sequence: (B, history_len, frame_dim)
        x = obs.view(B, self.history_len, self.frame_dim)

        # Project each frame to hidden dim
        tok = self.in_proj(x)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1) + self.pos

        if extract_attention:
            # Capture (approximate) attention weights for visualization
            with torch.no_grad():
                attn_out = seq
                attn_weights = None
                for layer in self.encoder.layers:
                    attn_out, attn_weights = layer.self_attn(
                        attn_out, attn_out, attn_out, need_weights=True
                    )
                self.last_attn = attn_weights.detach() if attn_weights is not None else None

        z = self.encoder(seq)
        return z[:, 0]                                # CLS token summary

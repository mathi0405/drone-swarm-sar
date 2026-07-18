"""Transformer + GNN encoder (our proposed architecture).

Intra-agent reasoning (over the agent's own token set) is handled by a Transformer;
inter-agent reasoning (over the communication graph) by a GNN. The two are composed
so the model captures both *what an agent sees* and *what its neighbours know*.
This is the ``transformer_gnn`` configuration evaluated as Model 4 / Figure 20.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from swarm_sar.policies.gnn_policy import GNNEncoder
from swarm_sar.policies.transformer_policy import TransformerEncoder


class TransformerGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg,
                 frame_layout: dict[str, int] | None = None):
        super().__init__()
        self.tf1 = TransformerEncoder(in_dim, hidden, cfg,
                                      frame_layout=frame_layout)
        self.ln_tf1 = nn.LayerNorm(hidden)

        self.gnn1 = GNNEncoder(hidden, hidden, cfg)
        self.ln_gnn1 = nn.LayerNorm(hidden)
        self.fuse1 = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.ln_fuse1 = nn.LayerNorm(hidden)

        # Second stack
        self.tf2 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.ln_tf2 = nn.LayerNorm(hidden)

        self.gnn2 = GNNEncoder(hidden, hidden, cfg)
        self.ln_gnn2 = nn.LayerNorm(hidden)
        self.fuse2 = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.ln_fuse2 = nn.LayerNorm(hidden)

        self.out_dim = hidden

    def forward(self, obs, graph=None):
        # Pass 1
        local1 = self.tf1(obs)                          # (B, hidden)
        local1 = self.ln_tf1(local1)

        relational1 = self.gnn1(local1, graph)           # message passing
        relational1 = self.ln_gnn1(relational1)

        fused1 = self.fuse1(torch.cat([local1, relational1], dim=-1))
        fused1 = self.ln_fuse1(fused1 + local1)          # Residual Fusion

        # Pass 2
        local2 = self.tf2(fused1)                        # tf2 is now an MLP since fused1 is a summary vector
        local2 = self.ln_tf2(local2 + fused1)            # Residual

        relational2 = self.gnn2(local2, graph)
        relational2 = self.ln_gnn2(relational2)

        fused2 = self.fuse2(torch.cat([local2, relational2], dim=-1))
        fused2 = self.ln_fuse2(fused2 + local2)          # Residual Fusion

        return fused2

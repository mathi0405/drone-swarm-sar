"""Transformer encoder over entity and temporal observation tokens.

Given the frame layout (see :func:`swarm_sar.environment.observation.frame_layout`),
the newest frame is tokenized into ENTITY tokens — self (own state + hazards +
LiDAR), victims, egocentric map, one token per nearest peer, and the message/
comms summary — while the full HISTORY_LEN frame stack contributes one TEMPORAL
token per frame.  Self-attention over this set gives a permutation-invariant
treatment of peers, lets attention weights be read per entity (Figure 15), and
keeps the temporal context the GRU baseline gets.

Without a frame layout (old checkpoints, synthetic test specs) the encoder
falls back to temporal-only tokens over the flattened stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from swarm_sar.environment.observation import HISTORY_LEN

# Token-type vocabulary for the learned type embedding.
_TYPE_CLS, _TYPE_SELF, _TYPE_VICTIM, _TYPE_MAP, _TYPE_PEER, _TYPE_MSG, _TYPE_TIME = range(7)


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg, frame_layout: dict[str, int] | None = None):
        super().__init__()
        # The input is a flattened temporal stack (B, HISTORY_LEN * frame_dim);
        # HISTORY_LEN is shared with ObservationEngine so the two cannot drift.
        self.history_len = HISTORY_LEN
        self.frame_dim = in_dim // self.history_len

        # Entity tokenization is only possible when the caller supplies the
        # frame layout AND it matches the actual frame width.
        self.entity_mode = (frame_layout is not None
                            and sum(frame_layout.values()) == self.frame_dim)
        if self.entity_mode:
            sizes = dict(frame_layout)
            offsets, off = {}, 0
            for name, width in sizes.items():
                offsets[name] = (off, off + width)
                off += width
            self._slices = offsets
            self.n_peers = sizes["peers"] // 6

            # Per-entity input projections (peers share one projection).
            self.proj_self = nn.Linear(
                sizes["own_state"] + sizes["hazards"] + sizes["lidar"], hidden)
            self.proj_victim = nn.Linear(sizes["victim_info"], hidden)
            self.proj_map = nn.Linear(sizes["map_patch"], hidden)
            self.proj_peer = nn.Linear(6, hidden)
            self.proj_msg = nn.Linear(sizes["comms"], hidden)
            self.type_embed = nn.Embedding(7, hidden)
            token_types = ([_TYPE_CLS, _TYPE_SELF, _TYPE_VICTIM, _TYPE_MAP]
                           + [_TYPE_PEER] * self.n_peers
                           + [_TYPE_MSG] + [_TYPE_TIME] * self.history_len)
            self.register_buffer("_token_types",
                                 torch.tensor(token_types, dtype=torch.long))
            self.token_labels = (["cls", "self", "victims", "map"]
                                 + [f"peer{i + 1}" for i in range(self.n_peers)]
                                 + ["msgs"]
                                 + [f"t-{self.history_len - 1 - i}"
                                    for i in range(self.history_len)])
            n_pos = len(token_types)
        else:
            self.token_labels = (["cls"]
                                 + [f"t-{self.history_len - 1 - i}"
                                    for i in range(self.history_len)])
            n_pos = self.history_len + 1

        self.in_proj = nn.Linear(self.frame_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        # Learned positional embedding: temporal tokens are an ordered
        # sequence (approaching vs. leaving a victim), and in entity mode it
        # doubles as a per-slot entity embedding on top of the type embedding.
        self.pos = nn.Parameter(torch.zeros(1, n_pos, hidden))
        nn.init.normal_(self.pos, std=0.02)
        # Pre-LN (norm_first): post-LN transformers under PPO at 3e-4 with no
        # warmup are a textbook instability — the 3M transformer_gnn run
        # unlearned its BC prior (stage rescue 0.69 -> 0.35) with exactly that
        # configuration.  Pre-LN keeps residual gradients well-scaled at depth.
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=cfg.n_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_dim = hidden
        self.last_attn = None

    def _entity_tokens(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the NEWEST frame into entity tokens.

        Returns ``(tokens, pad_mask)`` with tokens ``(B, E, hidden)`` and a
        boolean pad mask ``(B, E)`` that is True for absent peers (all-zero
        peer slots: out of comm range, dead, or padding), so attention never
        spends probability mass on placeholders.
        """
        latest = frames[:, -1]
        seg = {name: latest[:, a:b] for name, (a, b) in self._slices.items()}
        B = latest.shape[0]

        tok_self = self.proj_self(
            torch.cat([seg["own_state"], seg["hazards"], seg["lidar"]], dim=-1))
        tok_victim = self.proj_victim(seg["victim_info"])
        tok_map = self.proj_map(seg["map_patch"])
        peers = seg["peers"].view(B, self.n_peers, 6)
        tok_peers = self.proj_peer(peers)
        tok_msg = self.proj_msg(seg["comms"])

        tokens = torch.cat([tok_self.unsqueeze(1), tok_victim.unsqueeze(1),
                            tok_map.unsqueeze(1), tok_peers,
                            tok_msg.unsqueeze(1)], dim=1)
        pad = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=latest.device)
        pad[:, 3:3 + self.n_peers] = (peers.abs().sum(dim=-1) == 0)
        return tokens, pad

    def forward(self, obs, graph=None, extract_attention=False):
        B = obs.shape[0] if obs.dim() > 1 else 1

        # Reshape into time sequence: (B, history_len, frame_dim)
        x = obs.view(B, self.history_len, self.frame_dim)

        # Temporal tokens: one per history frame.
        tok_time = self.in_proj(x)
        cls = self.cls.expand(B, -1, -1)
        pad_mask = None
        if self.entity_mode:
            tok_entity, entity_pad = self._entity_tokens(x)
            seq = torch.cat([cls, tok_entity, tok_time], dim=1)
            pad_mask = torch.zeros(B, seq.shape[1], dtype=torch.bool,
                                   device=seq.device)
            pad_mask[:, 1:1 + entity_pad.shape[1]] = entity_pad
            seq = seq + self.pos + self.type_embed(self._token_types)
        else:
            seq = torch.cat([cls, tok_time], dim=1) + self.pos

        if extract_attention:
            # Capture (approximate) attention weights for visualization
            with torch.no_grad():
                attn_out = seq
                attn_weights = None
                for layer in self.encoder.layers:
                    attn_out, attn_weights = layer.self_attn(
                        attn_out, attn_out, attn_out, need_weights=True,
                        key_padding_mask=pad_mask,
                    )
                self.last_attn = attn_weights.detach() if attn_weights is not None else None

        z = self.encoder(seq, src_key_padding_mask=pad_mask)
        return z[:, 0]                                # CLS token summary

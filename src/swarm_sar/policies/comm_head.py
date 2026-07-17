"""Learned, bandwidth-constrained communication (the sharpened contribution).

Rather than broadcasting fixed hand-designed payloads, agents *learn what and when
to transmit*. Each agent emits a message vector and a scalar gate; under a bandwidth
budget only the highest-gated messages are sent; delivery is further masked by the
physical channel (range + packet loss, supplied as an adjacency). Receivers attend
over their inbox (TarMAC-style) to form a communication context that is concatenated
into the policy. This makes communication differentiable and studies emergent
protocols under realistic constraints. Torch is optional (import-guarded).
"""
from __future__ import annotations
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if not _HAS_TORCH:

    class LearnedCommHead:  # pragma: no cover - exercised only without torch
        """Import-safe placeholder; instantiating it without torch raises."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LearnedCommHead. "
                              "Install RL extras: pip install -e '.[rl]'")

else:

    class LearnedCommHead(nn.Module):
        def __init__(self, hidden: int, msg_dim: int = 32, n_heads: int = 4,
                     bandwidth: int = 4):
            super().__init__()
            self.msg_dim = msg_dim
            self.bandwidth = bandwidth
            self.encode = nn.Linear(hidden, msg_dim)          # what to say
            self.gate = nn.Linear(hidden, 1)                  # whether to say it
            self.q = nn.Linear(hidden, msg_dim)
            self.attn = nn.MultiheadAttention(msg_dim, n_heads, batch_first=True)
            self.out = nn.Linear(msg_dim, hidden)

        def forward(self, h: torch.Tensor, adj: torch.Tensor,
                    packet_keep: torch.Tensor = None) -> torch.Tensor:
            """h:(N,hidden) agent latents. adj:(N,N) in {0,1} in-range reachability.
            packet_keep:(N,N) in [0,1] Bernoulli-kept mask (1-packet_loss). Returns a
            per-agent communication context (N,hidden)."""
            N = h.shape[0]
            msg = self.encode(h)                              # (N, msg_dim)
            gate = torch.sigmoid(self.gate(h)).squeeze(-1)    # (N,) send-importance

            if adj.dim() == 3:
                B, A = adj.shape[0], adj.shape[1]
                msg_b = msg.view(B, A, -1)
                gate_b = gate.view(B, A)
                mask = adj.clone()
                if packet_keep is not None:
                    mask = mask * packet_keep
                gated = mask * gate_b.unsqueeze(1)
                if self.bandwidth < A:
                    kth = torch.topk(gated, k=min(self.bandwidth, A), dim=2).values[:, :, -1:]
                    gated = torch.where(gated >= kth, gated, torch.zeros_like(gated))

                q = self.q(h).view(B, A, -1)
                kv = msg_b
                key_pad = gated <= 0
                key_pad = key_pad & ~torch.eye(A, dtype=torch.bool, device=h.device).unsqueeze(0)
                attn_mask = key_pad.repeat_interleave(self.attn.num_heads, dim=0)
                ctx, _ = self.attn(q, kv, kv, attn_mask=attn_mask)
                return self.out(ctx).view(B * A, -1)
            else:
                mask = adj.clone()
                if packet_keep is not None:
                    mask = mask * packet_keep
                gated = mask * gate[None, :]                      # (N_recv, N_send)
                if self.bandwidth < N:
                    kth = torch.topk(gated, k=min(self.bandwidth, N), dim=1).values[:, -1:]
                    gated = torch.where(gated >= kth, gated, torch.zeros_like(gated))

                q = self.q(h).unsqueeze(1)                        # (N,1,msg_dim)
                kv = msg.unsqueeze(0).expand(N, N, self.msg_dim)  # (N,N,msg_dim)
                key_pad = gated <= 0                               # True = ignore
                key_pad = key_pad & ~torch.eye(N, dtype=torch.bool, device=h.device)
                ctx, _ = self.attn(q, kv, kv, key_padding_mask=key_pad)
                return self.out(ctx.squeeze(1))                   # (N, hidden)

        @staticmethod
        def bandwidth_cost(adj: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            """Differentiable comm-cost regularizer (encourage silence when uninformative)."""
            return (adj * gate[None, :]).sum()

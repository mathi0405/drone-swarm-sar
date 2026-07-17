"""Graph Neural Network encoder for inter-drone relational reasoning."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

class _ManualGAT(nn.Module):
    """Single-head Graph Attention v2 over a dense adjacency."""
    def __init__(self, dim: int):
        super().__init__()
        # GATv2 uses a shared W for query and key, then an attention vector 'a'
        self.w = nn.Linear(dim, dim, bias=False)
        self.a = nn.Linear(dim, 1, bias=False)
        self.v = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(3, dim) # Project edge features to hidden dim
        self.scale = dim ** -0.5

    def forward(self, h, adj, attr=None):                       # h:(N,D) adj:(N,N) or (B,A,A)
        wh = self.w(h)
        v = self.v(h)
        
        if adj.dim() == 3:
            B, A = adj.shape[0], adj.shape[1]
            wh = wh.view(B, A, 1, -1)
            wh_j = wh.view(B, 1, A, -1)
            
            e_attr = 0
            if attr is not None:
                e_attr = self.edge_proj(attr)
                
            e = F.leaky_relu(wh + wh_j + e_attr, 0.2)
            att = self.a(e).squeeze(-1) * self.scale
            
            att = att.masked_fill(adj < 0.5, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = torch.nan_to_num(att)
            
            v = v.view(B, A, -1)
            out = torch.bmm(att, v)
            return h + out.view(B * A, -1)
        else:
            wh_i = wh.unsqueeze(1)
            wh_j = wh.unsqueeze(0)
            
            e_attr = 0
            if attr is not None:
                e_attr = self.edge_proj(attr)
                
            e = F.leaky_relu(wh_i + wh_j + e_attr, 0.2)
            att = self.a(e).squeeze(-1) * self.scale
            
            att = att.masked_fill(adj < 0.5, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = torch.nan_to_num(att)                  # isolated nodes -> zero msg
            return h + att @ v                           # residual aggregation

class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.rounds = cfg.gnn_message_rounds
        self.out_dim = hidden
        self._pyg = _HAS_PYG
        self.comm_head = None
        if getattr(cfg, "use_attention_comm", False):
            from swarm_sar.policies.comm_head import LearnedCommHead
            self.comm_head = LearnedCommHead(hidden, bandwidth=4)
            
        if _HAS_PYG:
            if cfg.gnn_type == "gat":
                self.convs = nn.ModuleList([GATv2Conv(hidden, hidden, edge_dim=3, add_self_loops=True) for _ in range(self.rounds)])
            else:
                conv = {"gcn": GCNConv, "graphsage": SAGEConv}[cfg.gnn_type]
                self.convs = nn.ModuleList([conv(hidden, hidden) for _ in range(self.rounds)])
        else:
            self.convs = nn.ModuleList([_ManualGAT(hidden) for _ in range(self.rounds)])
            
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(self.rounds)])

    def forward(self, obs, graph=None):
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        h = self.embed(obs)
        if graph is None:
            return h
            
        if isinstance(graph, torch.Tensor):
            dense_adj = graph.float()
            edge_attr = None
            edge_idx = (torch.nonzero(dense_adj > 0, as_tuple=False).t().contiguous()
                        if self._pyg else None)
        elif self._pyg:
            edge_idx = graph.get("edge_index")
            edge_attr = graph.get("edge_attr")
            dense_adj = self._dense_adjacency(edge_idx, h.shape[0], h.device)
        else:
            dense_adj = graph.get("adj")
            edge_attr = graph.get("attr")
            
        if self.comm_head is not None and dense_adj is not None:
            h = h + self.comm_head(h, dense_adj)
            
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            if self._pyg:
                if isinstance(conv, GATv2Conv) and edge_attr is not None:
                    h_out = conv(h, edge_idx, edge_attr=edge_attr)
                else:
                    h_out = conv(h, edge_idx)
            else:
                h_out = conv(h, dense_adj, edge_attr)
                
            h = norm(h_in + torch.tanh(h_out)) # Residual + LayerNorm
        return h

    @staticmethod
    def _dense_adjacency(graph, n: int, device):
        if graph is None:
            return None
        if graph.dim() == 3 and graph.shape[1] == graph.shape[2]:
            return graph.float()
        if graph.dim() == 2 and graph.shape == (n, n):
            return graph.float()
        if graph.dim() == 2 and graph.shape[0] == 2:
            adj = torch.zeros((n, n), dtype=torch.float32, device=device)
            if graph.numel():
                adj[graph[1].long(), graph[0].long()] = 1.0
            adj.fill_diagonal_(1.0)
            return adj
        return None

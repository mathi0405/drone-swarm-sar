"""Build the dynamic communication graph adjacency used by GNN policies."""
from __future__ import annotations
from typing import Optional
import numpy as np


def comm_adjacency(positions: np.ndarray, comm_range: float) -> dict:
    """Dense (N,N) adjacency with edge attributes."""
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adj = (d <= comm_range).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    
    # normalized distance and rel_pos
    dist_norm = d / max(1.0, comm_range)
    rel_pos = (positions[None, :, :] - positions[:, None, :]) / max(1.0, comm_range)
    attr = np.concatenate([dist_norm[..., None], rel_pos], axis=-1).astype(np.float32)
    
    return {"adj": adj, "attr": attr}


def edge_index(adj: np.ndarray, attr: np.ndarray = None):
    """Convert dense adjacency to a 2xE edge_index (PyG format) and E x F edge_attr."""
    if adj.ndim == 3:
        B, A, _ = adj.shape
        src_list, dst_list = [], []
        attr_list = []
        for b in range(B):
            s, d = np.nonzero(adj[b])
            src_list.append(s + b * A)
            dst_list.append(d + b * A)
            if attr is not None:
                attr_list.append(attr[b, s, d])
        if src_list:
            e_idx = np.stack([np.concatenate(src_list), np.concatenate(dst_list)], axis=0)
            e_attr = np.concatenate(attr_list, axis=0) if attr is not None else None
            return e_idx, e_attr
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, attr.shape[-1]), dtype=np.float32) if attr is not None else None
    src, dst = np.nonzero(adj)
    e_idx = np.stack([src, dst], axis=0)
    e_attr = attr[src, dst] if attr is not None else None
    return e_idx, e_attr


def policy_uses_pyg_graph(policy) -> bool:
    """Return True when a policy encoder expects PyG edge_index graphs."""
    enc = getattr(policy, "encoder", policy)
    gnn = getattr(enc, "gnn", getattr(enc, "gnn1", enc))
    return bool(getattr(gnn, "_pyg", False))


def torch_graph_from_adjacency(graph_dict: dict, device, use_pyg: bool):
    """Convert a graph dict into the graph format a policy expects."""
    import torch
    
    if graph_dict is None:
        return None
        
    adj = graph_dict["adj"]
    attr = graph_dict["attr"]

    if use_pyg:
        e_idx, e_attr = edge_index(adj, attr)
        return {
            "edge_index": torch.tensor(e_idx, dtype=torch.long, device=device),
            "edge_attr": torch.tensor(e_attr, dtype=torch.float32, device=device) if e_attr is not None else None
        }
    return {
        "adj": torch.tensor(adj, dtype=torch.float32, device=device),
        "attr": torch.tensor(attr, dtype=torch.float32, device=device)
    }


def block_diag_adjacency(graphs: list[dict], n_agents: int) -> Optional[dict]:
    """Build batched adjacency and attribute tensors (B, A, A) and (B, A, A, F)."""
    if not graphs:
        return None
    blocks_adj = []
    blocks_attr = []
    for start in range(0, len(graphs), n_agents):
        blocks_adj.append(np.asarray(graphs[start]["adj"], dtype=np.float32))
        blocks_attr.append(np.asarray(graphs[start]["attr"], dtype=np.float32))
    return {
        "adj": np.stack(blocks_adj, axis=0),
        "attr": np.stack(blocks_attr, axis=0)
    }

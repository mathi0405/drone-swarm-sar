"""Build the dynamic communication graph adjacency used by GNN policies."""
from __future__ import annotations
from typing import Optional
import numpy as np


def comm_adjacency(positions: np.ndarray, comm_range: float,
                   packet_loss: float = 0.0,
                   rng: Optional[np.random.Generator] = None) -> dict:
    """Dense (N,N) adjacency with edge attributes and a physical-channel mask.

    ``keep`` is a Bernoulli(1 - packet_loss) mask sampled per directed edge so
    the learned communication head experiences the same lossy channel as the
    hand-designed message bus (self-links always survive).
    """
    n = positions.shape[0]
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adj = (d <= comm_range).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    # normalized distance and rel_pos
    dist_norm = d / max(1.0, comm_range)
    rel_pos = (positions[None, :, :] - positions[:, None, :]) / max(1.0, comm_range)
    attr = np.concatenate([dist_norm[..., None], rel_pos], axis=-1).astype(np.float32)

    if packet_loss > 0.0 and rng is not None:
        keep = (rng.random((n, n)) >= packet_loss).astype(np.float32)
        np.fill_diagonal(keep, 1.0)
    else:
        keep = np.ones((n, n), dtype=np.float32)

    return {"adj": adj, "attr": attr, "keep": keep}


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
    keep = graph_dict.get("keep")
    keep_t = (torch.tensor(keep, dtype=torch.float32, device=device)
              if keep is not None else None)

    if use_pyg:
        e_idx, e_attr = edge_index(adj, attr)
        return {
            "edge_index": torch.tensor(e_idx, dtype=torch.long, device=device),
            "edge_attr": torch.tensor(e_attr, dtype=torch.float32, device=device) if e_attr is not None else None,
            "keep": keep_t,
        }
    return {
        "adj": torch.tensor(adj, dtype=torch.float32, device=device),
        "attr": torch.tensor(attr, dtype=torch.float32, device=device),
        "keep": keep_t,
    }


def block_diag_adjacency(graphs: list[dict], n_agents: int) -> Optional[dict]:
    """Build batched adjacency/attribute/keep tensors (B, A, A) [+ (B, A, A, F)]."""
    if not graphs:
        return None
    blocks_adj = []
    blocks_attr = []
    blocks_keep = []
    for start in range(0, len(graphs), n_agents):
        g = graphs[start]
        blocks_adj.append(np.asarray(g["adj"], dtype=np.float32))
        blocks_attr.append(np.asarray(g["attr"], dtype=np.float32))
        keep = g.get("keep")
        blocks_keep.append(np.asarray(keep, dtype=np.float32) if keep is not None
                           else np.ones_like(blocks_adj[-1]))
    return {
        "adj": np.stack(blocks_adj, axis=0),
        "attr": np.stack(blocks_attr, axis=0),
        "keep": np.stack(blocks_keep, axis=0),
    }

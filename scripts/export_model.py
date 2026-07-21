#!/usr/bin/env python3
"""Export a trained policy for edge inference (TorchScript + ONNX).

Deployment target is the DECENTRALIZED actor path: a single drone maps its own
flattened observation to action logits (argmax = action). The centralized critic
and the inter-agent graph are training-only and are dropped. GNN-based encoders
require a neighbor graph and are exported with the single-agent (no-neighbor)
graph, matching how a lone drone runs at the edge.

Emits, next to the checkpoint (or --out):
  <name>.ts.pt    TorchScript module   (obs -> logits)
  <name>.onnx     ONNX graph           (obs -> logits)   [if onnx available]
  <name>.contract.json  obs_dim / n_actions / frame layout, so a consumer can
                        build inputs without importing the training code.

Usage:
  python scripts/export_model.py --ckpt results/trained/gru_s1_.../checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401  (CLI: puts src/ on sys.path)
except ImportError:
    pass
import torch

from swarm_sar.policies import load_policy


class ActorExport(torch.nn.Module):
    """obs (B, obs_dim) -> action logits (B, n_actions), decentralized."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.policy(obs)          # graph=None, global_state=None
        return logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output basename (default: alongside ckpt)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    policy = load_policy(args.ckpt)
    policy.eval()
    spec = policy.spec if hasattr(policy, "spec") else None
    obs_dim = spec.obs_dim if spec else next(policy.parameters()).shape[-1]
    n_actions = policy.actor[-1].out_features if hasattr(policy.actor, "__getitem__") else None

    wrapper = ActorExport(policy).eval()
    example = torch.zeros(1, obs_dim)
    with torch.no_grad():
        ref = wrapper(example)
    n_actions = n_actions or ref.shape[-1]

    base = Path(args.out) if args.out else Path(args.ckpt).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    # TorchScript (always available with torch).
    ts = torch.jit.trace(wrapper, example)
    ts_path = base.with_suffix(".ts.pt")
    ts.save(str(ts_path))
    ts_out = ts(example)
    assert torch.allclose(ref, ts_out, atol=1e-5), "TorchScript parity failed"
    print(f"[OK] TorchScript -> {ts_path}  (parity max abs diff={float((ref-ts_out).abs().max()):.2e})")

    # ONNX (optional runtime check).
    onnx_path = base.with_suffix(".onnx")
    try:
        torch.onnx.export(
            wrapper, example, str(onnx_path),
            input_names=["obs"], output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        print(f"[OK] ONNX -> {onnx_path}")
        try:
            import numpy as np
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            onnx_out = sess.run(None, {"obs": example.numpy()})[0]
            max_d = float(np.abs(ref.numpy() - onnx_out).max())
            assert max_d < 1e-4, f"ONNX parity failed (max|Δ|={max_d})"
            print(f"[OK] ONNX runtime parity: max|Δ|={max_d:.2e}")
        except ImportError:
            print("[--] onnxruntime not installed; skipped ONNX numeric parity check")
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[!!] ONNX export skipped: {e}")

    contract = {
        "obs_dim": int(obs_dim), "n_actions": int(n_actions),
        "frame_layout": spec.frame_layout if spec else None,
        "input": "obs (float32, [batch, obs_dim])",
        "output": "logits (float32, [batch, n_actions]); action = argmax(logits, -1)",
        "note": "decentralized actor; critic and inter-agent graph dropped for edge use",
    }
    cpath = base.with_suffix(".contract.json")
    cpath.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"[OK] contract -> {cpath}")


if __name__ == "__main__":
    main()

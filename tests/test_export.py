"""Export-parity tests: the deployed actor must match the training policy.

Exercises the ActorExport wrapper + TorchScript trace on a small synthetic
policy (no checkpoint needed). ONNX runtime parity is checked when onnxruntime
is installed, otherwise skipped.
"""
import numpy as np
import pytest

from swarm_sar.policies.base import HAS_TORCH

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _policy(obs_dim=64, n_actions=8):
    import torch  # noqa

    from swarm_sar.config import ModelConfig
    from swarm_sar.policies.base import PolicySpec, build_policy
    spec = PolicySpec(obs_dim=obs_dim, n_actions=n_actions, global_dim=16, n_agents=3)
    pol = build_policy(ModelConfig(arch="gru", hidden_dim=32), spec).eval()
    pol.spec = spec
    return pol


def test_torchscript_matches_eager():
    import torch

    from scripts.export_model import ActorExport
    pol = _policy()
    wrapper = ActorExport(pol).eval()
    x = torch.randn(4, pol.spec.obs_dim)
    with torch.no_grad():
        ref = wrapper(x)
    ts = torch.jit.trace(wrapper, torch.zeros(1, pol.spec.obs_dim))
    assert torch.allclose(ref, ts(x), atol=1e-5)
    # argmax action agrees — the only thing deployment actually consumes.
    assert torch.equal(ref.argmax(-1), ts(x).argmax(-1))


def test_onnx_parity_if_available(tmp_path):
    import torch

    from scripts.export_model import ActorExport
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")
    pol = _policy()
    wrapper = ActorExport(pol).eval()
    x = torch.randn(3, pol.spec.obs_dim)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    onnx_path = tmp_path / "actor.onnx"
    torch.onnx.export(wrapper, torch.zeros(1, pol.spec.obs_dim), str(onnx_path),
                      input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": x.numpy()})[0]
    assert np.abs(ref - out).max() < 1e-4

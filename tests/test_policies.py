"""Neural-policy tests. Skipped automatically when PyTorch is not installed."""
from swarm_sar.policies.base import HAS_TORCH, build_policy, PolicySpec
from swarm_sar.config import ModelConfig


def _spec():
    return PolicySpec(obs_dim=64, n_actions=10, global_dim=64 * 3, n_agents=3)


def test_build_all_archs():
    if not HAS_TORCH:                       # graceful skip without pytest dependency
        return
    import torch
    spec = _spec()
    for arch in ["mlp", "gnn", "transformer", "transformer_gnn"]:
        model = build_policy(ModelConfig(arch=arch), spec)
        obs = torch.zeros(spec.n_agents, spec.obs_dim)
        gs = torch.zeros(spec.n_agents, spec.global_dim)
        logits, value = model(obs, global_state=gs)
        assert logits.shape == (spec.n_agents, spec.n_actions)
        assert value.shape[0] == spec.n_agents


def test_policy_accepts_graph_and_action_mask():
    if not HAS_TORCH:
        return
    import torch
    spec = _spec()
    model = build_policy(ModelConfig(arch="transformer_gnn"), spec)
    obs = torch.zeros(spec.n_agents, spec.obs_dim)
    gs = torch.zeros(spec.n_agents, spec.global_dim)
    graph = torch.eye(spec.n_agents)

    logits, value = model(obs, graph=graph, global_state=gs)

    assert logits.shape == (spec.n_agents, spec.n_actions)
    assert value.shape == (spec.n_agents, 1)


def test_deterministic_action_respects_mask():
    if not HAS_TORCH:
        return
    import torch
    spec = _spec()
    model = build_policy(ModelConfig(arch="mlp"), spec)
    for p in model.parameters():
        p.data.zero_()
    obs = torch.zeros(1, spec.obs_dim)
    gs = torch.zeros(1, spec.global_dim)
    mask = torch.ones(1, spec.n_actions)
    mask[:, 0] = 0.0

    action, _, _ = model.act(obs, global_state=gs, action_mask=mask, deterministic=True)

    assert int(action.item()) == 1


def test_factory_requires_torch_message():
    if HAS_TORCH:
        return
    try:
        build_policy(ModelConfig(arch="mlp"), _spec())
        assert False, "should have raised"
    except ImportError as e:
        assert "PyTorch" in str(e)

"""Neural-policy tests. Skipped automatically when PyTorch is not installed."""
from swarm_sar.config import ModelConfig
from swarm_sar.policies.base import HAS_TORCH, PolicySpec, build_policy


def _spec():
    return PolicySpec(obs_dim=64, n_actions=10, global_dim=64 * 3, n_agents=3)


def test_build_all_archs():
    if not HAS_TORCH:                       # graceful skip without pytest dependency
        return
    import torch
    spec = _spec()
    for arch in ["mlp", "gru", "gnn", "transformer", "transformer_gnn"]:
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
        raise AssertionError("should have raised")
    except ImportError as e:
        assert "PyTorch" in str(e)


def test_transformer_entity_tokenization():
    """With a frame layout the Transformer builds entity + temporal tokens."""
    if not HAS_TORCH:
        return
    import torch

    from swarm_sar.environment.observation import HISTORY_LEN, frame_layout

    layout = frame_layout(3, 7)
    frame_dim = sum(layout.values())
    spec = PolicySpec(obs_dim=HISTORY_LEN * frame_dim, n_actions=8,
                      global_dim=51, n_agents=3, frame_layout=layout)
    for arch in ["transformer", "transformer_gnn"]:
        model = build_policy(ModelConfig(arch=arch, hidden_dim=64), spec)
        enc = model.encoder.tf1 if arch == "transformer_gnn" else model.encoder
        assert enc.entity_mode
        # CLS + self + victims + map + k peers + msgs + HISTORY_LEN temporal.
        assert len(enc.token_labels) == 4 + 3 + 1 + HISTORY_LEN
        obs = torch.randn(3, spec.obs_dim)
        logits, value = model(obs, global_state=torch.randn(3, 51),
                              graph=torch.eye(3))
        assert logits.shape == (3, 8)
        assert torch.isfinite(logits).all()


def test_transformer_entity_peer_padding_is_masked():
    """All-zero peer slots (absent peers) must not poison attention."""
    if not HAS_TORCH:
        return
    import torch

    from swarm_sar.environment.observation import HISTORY_LEN, frame_layout

    layout = frame_layout(3, 7)
    frame_dim = sum(layout.values())
    spec = PolicySpec(obs_dim=HISTORY_LEN * frame_dim, n_actions=8,
                      global_dim=51, n_agents=2, frame_layout=layout)
    model = build_policy(ModelConfig(arch="transformer", hidden_dim=64), spec)
    obs = torch.randn(2, spec.obs_dim).view(2, HISTORY_LEN, frame_dim)
    peer_start = sum(w for name, w in layout.items() if name != "peers"
                     ) - layout["comms"]
    obs[:, -1, peer_start:peer_start + layout["peers"]] = 0.0
    logits, _ = model(obs.reshape(2, -1), global_state=torch.randn(2, 51))
    assert torch.isfinite(logits).all()


def test_transformer_without_layout_falls_back_to_temporal():
    if not HAS_TORCH:
        return
    import torch
    spec = _spec()                          # no frame_layout
    model = build_policy(ModelConfig(arch="transformer"), spec)
    assert not model.encoder.entity_mode
    logits, _ = model(torch.zeros(3, spec.obs_dim),
                      global_state=torch.zeros(3, spec.global_dim))
    assert logits.shape == (3, spec.n_actions)

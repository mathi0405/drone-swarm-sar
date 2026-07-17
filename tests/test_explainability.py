import json

import numpy as np

from swarm_sar.explainability import (
    explain_action,
    export_attention_weights,
    save_explanations,
    victim_priority,
)


def test_explain_action_returns_top_contributors():
    trace = explain_action("drone_0", "north", np.array([0.1, 3.0, -2.0]), step=4)

    assert trace.step == 4
    assert trace.contributors[0]["feature"] == 1


def test_attention_export_links_positive_weights():
    exported = export_attention_weights(np.array([[1.0, 0.0], [0.5, 1.0]]))

    assert len(exported["links"]) == 3


def test_victim_priority_prefers_severe_near_target():
    near = victim_priority(np.array([0.0, 0.0]), np.array([1.0, 0.0]), severity=1.0)
    far = victim_priority(np.array([0.0, 0.0]), np.array([100.0, 0.0]), severity=0.5)

    assert near > far


def test_explanation_json_export():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "trace.json"
        trace = explain_action("drone_0", "hover")

        save_explanations(out, [trace])

        assert json.loads(out.read_text())[0]["agent"] == "drone_0"

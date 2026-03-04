"""Tests for GraphEdge model (T1: intent nullable + intent_failure_reason)."""

from graph_agent.models import (
    ActionType,
    GraphEdge,
    GraphNode,
    GraphData,
    Intent,
)


def test_graph_edge_with_intent_none_new_sample():
    """New sample: intent=None with intent_failure_reason should construct."""
    edge = GraphEdge(
        source="https://a.com",
        target="https://b.com",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM output format invalid",
    )
    assert edge.intent is None
    assert edge.intent_failure_reason == "LLM output format invalid"


def test_graph_edge_with_intent_old_sample():
    """Old sample: intent=Intent(...) without failure reason should construct."""
    intent = Intent(
        raw="click login button",
        verb="Click",
        object="Login button",
        summary="Click login button",
    )
    edge = GraphEdge(
        source="https://a.com",
        target="https://b.com",
        selector="#login",
        action=ActionType.CLICK,
        intent=intent,
        intent_failure_reason=None,
    )
    assert edge.intent is not None
    assert edge.intent.summary == "Click login button"
    assert edge.intent_failure_reason is None


def test_graph_edge_from_json_with_intent_null():
    """Pydantic must accept intent=null in JSON."""
    json_str = """{
        "source": "https://a.com",
        "target": "https://b.com",
        "selector": "#btn",
        "action": "click",
        "intent": null,
        "intent_failure_reason": "parse failed"
    }"""
    edge = GraphEdge.model_validate_json(json_str)
    assert edge.intent is None
    assert edge.intent_failure_reason == "parse failed"


def test_graph_edge_json_roundtrip_with_null_intent():
    """Round-trip: serialize and deserialize preserves intent=null."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="button",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="reason",
    )
    json_str = edge.model_dump_json()
    restored = GraphEdge.model_validate_json(json_str)
    assert restored.intent is None
    assert restored.intent_failure_reason == "reason"


def test_graph_data_with_null_intent_edges():
    """GraphData containing edges with intent=null should construct."""
    nodes = [
        GraphNode(id="a", url="https://a.com", title="A"),
        GraphNode(id="b", url="https://b.com", title="B"),
    ]
    edges = [
        GraphEdge(
            source="a",
            target="b",
            selector="#x",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="failed",
        ),
    ]
    data = GraphData(nodes=nodes, edges=edges)
    assert len(data.edges) == 1
    assert data.edges[0].intent is None
    assert data.edges[0].intent_failure_reason == "failed"


def test_graph_edge_defaults():
    """Optional fields default to None when omitted."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="x",
        action=ActionType.NAVIGATE,
    )
    assert edge.intent is None
    assert edge.intent_failure_reason is None
    assert edge.data_key is None
    assert edge.constraints is None

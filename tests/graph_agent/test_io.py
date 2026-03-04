"""Tests for Graph I/O (T3: intent=null compatibility)."""

import tempfile
from pathlib import Path

import networkx as nx

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.models import ActionType, Intent


def test_save_load_roundtrip_intent_null():
    """Round-trip: save graph with intent=null edge, load and verify fields preserved."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com", title="A")
    G.add_node("b", url="https://b.com", title="B")
    G.add_edge(
        "a",
        "b",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM output format invalid",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        assert loaded.number_of_nodes() == 2
        assert loaded.number_of_edges() == 1

        u, v, data = next(iter(loaded.edges(data=True)))
        assert data["intent"] is None
        assert data["intent_failure_reason"] == "LLM output format invalid"


def test_save_load_roundtrip_mixed_intents():
    """Round-trip: graph with both intent=null and intent=Intent edges."""
    intent = Intent(
        raw="click login",
        verb="Click",
        object="Login button",
        summary="Click login button",
    )
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=intent, intent_failure_reason=None)
    G.add_edge("b", "c", selector="#y", action=ActionType.FILL, intent=None, intent_failure_reason="parse failed")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        edges = list(loaded.edges(data=True))
        assert len(edges) == 2

        # Find edges by target
        by_target = {e[1]: e[2] for e in edges}
        assert by_target["b"]["intent"] is not None
        assert by_target["b"]["intent"].summary == "Click login button"
        assert by_target["b"]["intent_failure_reason"] is None

        assert by_target["c"]["intent"] is None
        assert by_target["c"]["intent_failure_reason"] == "parse failed"


def test_load_json_with_intent_null():
    """load_graph correctly parses JSON file containing intent=null."""
    json_content = """{
  "nodes": [
    {"id": "a", "url": "https://a.com", "title": null},
    {"id": "b", "url": "https://b.com", "title": null}
  ],
  "edges": [
    {
      "source": "a",
      "target": "b",
      "selector": "#btn",
      "action": "click",
      "intent": null,
      "intent_failure_reason": "parse failed"
    }
  ],
  "metadata": {}
}"""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        path.write_text(json_content, encoding="utf-8")

        loaded = load_graph(path)
        assert loaded.number_of_edges() == 1
        u, v, data = next(iter(loaded.edges(data=True)))
        assert data["intent"] is None
        assert data["intent_failure_reason"] == "parse failed"


def test_save_preserves_intent_null_in_json():
    """Saved JSON file explicitly contains intent=null."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="reason")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        content = path.read_text(encoding="utf-8")
        assert '"intent": null' in content
        assert '"intent_failure_reason": "reason"' in content

"""Tests for pathfinding (T6: empty intent tolerance)."""

import networkx as nx

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import ActionType, Intent


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


def test_pathfinding_skips_null_intent_for_matching():
    """Edges with intent=None are skipped for matching but traversable."""
    intent = _make_intent("Click login", key="submit_login")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    # a -> b has null intent, b -> c has submit_login
    G.add_edge(
        "a", "b",
        selector="#nav",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="parse failed",
    )
    G.add_edge(
        "b", "c",
        selector="#login",
        action=ActionType.CLICK,
        intent=intent,
        intent_failure_reason=None,
    )

    path = get_path_from_intent("submit_login", G)
    assert len(path) == 2
    assert path[0].source == "a" and path[0].target == "b"
    assert path[0].intent is None
    assert path[0].selector == "#nav"
    assert path[1].source == "b" and path[1].target == "c"
    assert path[1].intent is not None
    assert path[1].intent.key == "submit_login"


def test_pathfinding_all_null_intent_returns_empty():
    """When all edges have null intent, returns [] without crashing."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a", "b",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM failed",
    )

    path = get_path_from_intent("login", G)
    assert path == []


def test_pathfinding_mixed_intents_returns_valid_path():
    """Path with mix of null and non-null intents is valid for playback."""
    intent = _make_intent("Fill username", key="fill_username")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="x")
    G.add_edge("b", "c", selector="#user", action=ActionType.FILL, intent=intent, intent_failure_reason=None)

    path = get_path_from_intent("fill_username", G)
    assert len(path) == 2
    for edge in path:
        assert edge.selector
        assert edge.action
        assert edge.source and edge.target


def test_pathfinding_no_crash_on_null_intent_access():
    """No crash when iterating edges with intent=None."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#btn", action=ActionType.CLICK, intent=None)

    path = get_path_from_intent("anything", G)
    assert path == []

"""Tests for Mapping run module (T4:构图与统计联动, T5:re-infer-missing)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import networkx as nx

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.mapping.run import (
    _build_graph_from_history,
    FILTERED_ACTION_KEYS,
    re_infer_missing_intents,
)
from graph_agent.models import ActionType, Intent


def _make_mock_history(actions: list[dict], thoughts: list[dict], urls: list[str]):
    """Create a minimal mock history object."""

    class MockHistory:
        def model_actions(self):
            return actions

        def model_thoughts(self):
            return thoughts

        def urls(self):
            return urls

    return MockHistory()


@pytest.mark.asyncio
async def test_build_graph_preserves_intent_and_failure_reason():
    """T4: Edges with intent=None and intent_failure_reason are preserved in graph."""
    history = _make_mock_history(
        actions=[
            {"click": {"element": "button"}, "interacted_element": {"xpath": "//button[@id='x']"}},
        ],
        thoughts=[{"next_goal": "Click"}],
        urls=["https://a.com", "https://b.com"],
    )

    edge_with_null_intent = type(
        "EdgeModel",
        (),
        {
            "selector": "xpath=//button[@id='x']",
            "action": ActionType.CLICK,
            "intent": None,
            "intent_failure_reason": "parse_error:invalid_json",
            "data_key": None,
            "constraints": None,
        },
    )()

    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        return_value=edge_with_null_intent,
    ):
        G = await _build_graph_from_history(history)

    assert G.number_of_edges() == 1
    u, v, data = next(iter(G.edges(data=True)))
    assert data["intent"] is None
    assert data["intent_failure_reason"] == "parse_error:invalid_json"
    assert data["selector"] == "xpath=//button[@id='x']"
    assert data["action"] == ActionType.CLICK


@pytest.mark.asyncio
async def test_build_graph_filtered_non_ui_edges_in_metadata():
    """T4: filtered_non_ui_edges is written to graph metadata."""
    history = _make_mock_history(
        actions=[
            {"read_file": {"path": "/tmp/x"}},
            {"write_file": {"path": "/tmp/y"}},
            {"click": {"element": "btn"}, "interacted_element": {"xpath": "//button"}},
        ],
        thoughts=[{}, {}, {}],
        urls=["https://a.com", "https://a.com", "https://a.com", "https://b.com"],
    )

    valid_edge = type(
        "EdgeModel",
        (),
        {
            "selector": "xpath=//button",
            "action": ActionType.CLICK,
            "intent": Intent(summary="Click btn", raw="click", verb="Click", object="Btn"),
            "intent_failure_reason": None,
            "data_key": None,
            "constraints": None,
        },
    )()

    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        return_value=valid_edge,
    ):
        G = await _build_graph_from_history(history)

    assert G.graph["filtered_non_ui_edges"] == 2  # read_file, write_file
    assert G.number_of_edges() == 1


@pytest.mark.asyncio
async def test_build_graph_filters_unknown_action_key():
    """T4: unknown action_key is filtered (FILTERED_ACTION_KEYS includes unknown)."""
    assert "unknown" in FILTERED_ACTION_KEYS

    history = _make_mock_history(
        actions=[{"unknown": {"foo": "bar"}}],
        thoughts=[{}],
        urls=["https://a.com", "https://b.com"],
    )

    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
    ):
        G = await _build_graph_from_history(history)

    assert G.number_of_edges() == 0
    assert G.graph["filtered_non_ui_edges"] == 1


# --- T5: re-infer-missing ---


def _make_graph_with_missing_intents(tmp_path: Path) -> Path:
    """Create a graph file with edges having intent=None for re-infer tests."""
    G = nx.DiGraph()
    G.add_node("https://a.com", label="a", url="https://a.com")
    G.add_node("https://b.com", label="b", url="https://b.com")
    G.add_node("https://c.com", label="c", url="https://c.com")
    G.add_edge(
        "https://a.com",
        "https://b.com",
        selector="xpath=//button[@id='x']",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="parse_error",
        data_key=None,
        constraints=None,
    )
    G.add_edge(
        "https://b.com",
        "https://c.com",
        selector="xpath=//input[@id='y']",
        action=ActionType.FILL,
        intent=None,
        intent_failure_reason="llm_timeout",
        data_key=None,
        constraints=None,
    )
    path = tmp_path / "graph.json"
    save_graph(G, path)
    return path


@pytest.mark.asyncio
async def test_re_infer_missing_intents_success(tmp_path: Path):
    """T5: Re-infer fills missing intents and clears intent_failure_reason on success."""
    graph_path = _make_graph_with_missing_intents(tmp_path)
    expected_intent = Intent(
        raw="click submit",
        verb="Click",
        object="Submit",
        summary="Click submit button",
    )

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ):
        stats = await re_infer_missing_intents(graph_path)

    assert stats["total"] == 2
    assert stats["succeeded"] == 2
    assert stats["failed"] == 0

    G = load_graph(graph_path)
    for u, v, data in G.edges(data=True):
        assert data["intent"] is not None
        assert data["intent"].summary == "Click submit button"
        assert data["intent_failure_reason"] is None


@pytest.mark.asyncio
async def test_re_infer_missing_intents_failure(tmp_path: Path):
    """T5: Re-infer keeps intent=None and sets intent_failure_reason on failure."""
    graph_path = _make_graph_with_missing_intents(tmp_path)

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(None, "llm_error:rate_limit"),
    ):
        stats = await re_infer_missing_intents(graph_path)

    assert stats["total"] == 2
    assert stats["succeeded"] == 0
    assert stats["failed"] == 2

    G = load_graph(graph_path)
    for u, v, data in G.edges(data=True):
        assert data["intent"] is None
        assert data["intent_failure_reason"] == "llm_error:rate_limit"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_mixed(tmp_path: Path):
    """T5: Re-infer handles mixed success/failure correctly."""
    graph_path = _make_graph_with_missing_intents(tmp_path)
    success_intent = Intent(
        raw="click",
        verb="Click",
        object="Button",
        summary="Click button",
    )

    call_count = 0

    async def mock_infer(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (success_intent, None)
        return (None, "parse_error")

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        side_effect=mock_infer,
    ):
        stats = await re_infer_missing_intents(graph_path)

    assert stats["total"] == 2
    assert stats["succeeded"] == 1
    assert stats["failed"] == 1

    G = load_graph(graph_path)
    # a->b should have intent (first call succeeded)
    ab_data = G.edges["https://a.com", "https://b.com"]
    assert ab_data["intent"] is not None
    assert ab_data["intent_failure_reason"] is None
    # b->c should stay None (second call failed)
    bc_data = G.edges["https://b.com", "https://c.com"]
    assert bc_data["intent"] is None
    assert bc_data["intent_failure_reason"] == "parse_error"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_skips_non_null(tmp_path: Path):
    """T5: Re-infer skips edges that already have intent."""
    G = nx.DiGraph()
    G.add_node("a", label="a", url="a")
    G.add_node("b", label="b", url="b")
    existing_intent = Intent(
        raw="existing",
        verb="Click",
        object="Btn",
        summary="Existing intent",
    )
    G.add_edge("a", "b", selector="#btn", action=ActionType.CLICK, intent=existing_intent)
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
    ) as mock_infer:
        stats = await re_infer_missing_intents(graph_path)

    assert stats["total"] == 0
    assert stats["succeeded"] == 0
    assert stats["failed"] == 0
    mock_infer.assert_not_called()

    G2 = load_graph(graph_path)
    u, v, data = next(iter(G2.edges(data=True)))
    assert data["intent"] is not None
    assert data["intent"].summary == "Existing intent"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_file_not_found():
    """T5: Re-infer raises FileNotFoundError when graph file does not exist."""
    with pytest.raises(FileNotFoundError, match="Graph file not found"):
        await re_infer_missing_intents("/nonexistent/path/graph.json")

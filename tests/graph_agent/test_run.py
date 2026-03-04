"""Tests for Mapping run module (T4:构图与统计联动)."""

import pytest
from unittest.mock import AsyncMock, patch

from graph_agent.mapping.run import _build_graph_from_history, FILTERED_ACTION_KEYS
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

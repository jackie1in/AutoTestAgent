"""Tests for web API (T8: API 与前端展示联动)."""

from unittest.mock import patch

import networkx as nx

from fastapi.testclient import TestClient

from graph_agent.models import ActionType, Intent
from graph_agent.web.app import app


def test_api_intents_ignores_null_intent_edges():
    """T8: /api/intents excludes edges with intent=null from the dropdown options."""
    intent = Intent(
        raw="click login",
        verb="Click",
        object="Login",
        summary="Click login button",
        key="submit_login",
    )
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="parse failed")
    G.add_edge("b", "c", selector="#login", action=ActionType.CLICK, intent=intent, intent_failure_reason=None)

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["value"] == "submit_login"
    assert "submit_login" in (data[0].get("label") or "")


def test_api_graph_returns_intent_failure_reason():
    """T8: /api/graph returns intent_failure_reason on edges for diagnostics."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a",
        "b",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM output format invalid",
    )

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["edges"]) == 1
    assert data["edges"][0]["intent"] is None
    assert data["edges"][0]["intent_failure_reason"] == "LLM output format invalid"


def test_api_graph_returns_missing_count_and_failure_reasons():
    """T8: /api/graph returns missing_count and failure_reasons for frontend diagnostics."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="reason1")
    G.add_edge("b", "c", selector="#y", action=ActionType.FILL, intent=None, intent_failure_reason="reason1")

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["missing_count"] == 2
    assert data["failure_reasons"] == ["reason1"]


def test_api_graph_empty_when_file_missing():
    """Empty graph response includes missing_count and failure_reasons."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["missing_count"] == 0
    assert data["failure_reasons"] == []

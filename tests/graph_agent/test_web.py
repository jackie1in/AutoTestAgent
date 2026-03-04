"""Tests for web API (T8: API 与前端展示联动) and auth."""

from unittest.mock import patch

import networkx as nx

from fastapi.testclient import TestClient

from graph_agent.models import ActionType, Intent
from graph_agent.web.app import app


def _auth_headers(client: TestClient) -> dict:
    """Login and return Authorization headers for protected endpoints."""
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_auth_login_success():
    """POST /api/auth/login returns token for valid credentials."""
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 0


def test_auth_login_invalid_credentials():
    """POST /api/auth/login returns 401 for invalid credentials."""
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert "error" in resp.json()


def test_api_graph_requires_auth():
    """GET /api/graph returns 401 without token."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 401


def test_api_intents_requires_auth():
    """GET /api/intents returns 401 without token."""
    client = TestClient(app)
    resp = client.get("/api/intents")
    assert resp.status_code == 401


def test_api_playback_requires_auth():
    """POST /api/playback returns 401 without token."""
    client = TestClient(app)
    resp = client.post("/api/playback", json={"intent": "x", "test_data": {}})
    assert resp.status_code == 401


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
            resp = client.get("/api/intents", headers=_auth_headers(client))
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
            resp = client.get("/api/graph", headers=_auth_headers(client))
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
            resp = client.get("/api/graph", headers=_auth_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["missing_count"] == 2
    assert data["failure_reasons"] == ["reason1"]


def test_api_graph_empty_when_file_missing():
    """Empty graph response includes missing_count and failure_reasons."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/graph", headers=_auth_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["missing_count"] == 0
    assert data["failure_reasons"] == []


def test_api_dashboard_requires_auth():
    """GET /api/dashboard returns 401 without token."""
    client = TestClient(app)
    resp = client.get("/api/dashboard")
    assert resp.status_code == 401


def test_api_dashboard_empty_when_file_missing():
    """Dashboard returns zeroed stats when graph file does not exist."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/dashboard", headers=_auth_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 0
    assert data["edge_count"] == 0
    assert data["intent_missing_count"] == 0
    assert data["intent_success_rate"] == 1.0
    assert data["filtered_non_ui_edges"] == 0
    assert data["mapping_stopped"] is None
    assert data["stop_reason"] is None


def test_api_dashboard_returns_stats():
    """Dashboard returns graph statistics with metadata."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="parse failed")
    G.add_edge("b", "c", selector="#login", action=ActionType.CLICK, intent=Intent(
        raw="click login", verb="Click", object="Login", summary="Click login", key="submit_login"
    ), intent_failure_reason=None)
    G.graph["filtered_non_ui_edges"] = 3
    G.graph["mapping_stopped"] = True
    G.graph["stop_reason"] = "Stopped: unfillable form"

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/dashboard", headers=_auth_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 3
    assert data["edge_count"] == 2
    assert data["intent_missing_count"] == 1
    assert data["intent_success_rate"] == 0.5
    assert data["filtered_non_ui_edges"] == 3
    assert data["mapping_stopped"] is True
    assert data["stop_reason"] == "Stopped: unfillable form"

"""Tests for web API (T8: API 与前端展示联动)."""

from unittest.mock import patch

import networkx as nx

from fastapi.testclient import TestClient

from graph_agent.models import ActionType, Intent
from graph_agent.web.app import app


def test_auth_login_route_removed():
    """POST /api/auth/login should not exist anymore."""
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert resp.status_code in (404, 405)


def test_api_graph_no_auth_required():
    """GET /api/graph returns 200 without token."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200


def test_api_intents_no_auth_required():
    """GET /api/intents returns 200 without token."""
    client = TestClient(app)
    resp = client.get("/api/intents")
    assert resp.status_code == 200


def test_api_playback_no_auth_required():
    """POST /api/playback no longer returns 401 without token."""
    client = TestClient(app)
    resp = client.post("/api/playback", json={"intent": "x", "test_data": {}})
    assert resp.status_code != 401


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
    assert "metadata" in data


def test_api_graph_returns_param_name_action_value_and_element():
    """Graph API should expose new fill metadata fields."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a",
        "b",
        selector="[name='username']",
        action=ActionType.FILL,
        intent=None,
        intent_failure_reason=None,
        param_name="username",
        action_value="tomsmith",
        element={
            "selector": "[name='username']",
            "css_selector": "input[name='username']",
            "name": "username",
            "id": "user",
            "type": "text",
            "attributes": {"name": "username", "id": "user", "type": "text"},
        },
    )

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edges"][0]["param_name"] == "username"
    assert data["edges"][0]["action_value"] == "tomsmith"
    assert data["edges"][0]["element"]["attributes"]["id"] == "user"


def test_api_graph_returns_business_template_metrics():
    """Graph API metadata should expose business template counters."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None)
    G.graph["business_template_count"] = 2
    G.graph["business_template_generation_failures"] = 1

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["business_template_count"] == 2
    assert data["metadata"]["business_template_generation_failures"] == 1


def test_api_graph_returns_business_templates_with_dependencies():
    """Graph API should expose business templates and their prerequisite links."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.graph["business_templates"] = [
        {
            "template_id": "tpl-auth-login",
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "entry_node": "login",
            "exit_node": "secure",
            "path_length": 3,
            "confidence": 0.95,
            "steps": [],
            "slots": {},
            "evidence": {},
            "depends_on": [],
        },
        {
            "template_id": "tpl-project-dashboard",
            "business_key": "project.dashboard.open",
            "summary": "打开项目看板",
            "entry_node": "secure",
            "exit_node": "dashboard",
            "path_length": 2,
            "confidence": 0.9,
            "steps": [],
            "slots": {},
            "evidence": {},
            "depends_on": ["auth.login"],
        },
    ]

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["business_templates"]) == 2
    assert data["business_templates"][1]["business_key"] == "project.dashboard.open"
    assert data["business_templates"][1]["depends_on"] == ["auth.login"]


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
    assert data["business_templates"] == []
    assert data["metadata"]["intent_success_rate"] == 1.0


def test_index_page_contains_template_links_section():
    """Index page should include a dedicated template dependency section."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="template-links"' in resp.text
    assert "前置业务关联" in resp.text


def test_api_dashboard_no_auth_required():
    """GET /api/dashboard returns 200 without token."""
    client = TestClient(app)
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200


def test_api_dashboard_empty_when_file_missing():
    """Dashboard returns zeroed stats when graph file does not exist."""
    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = False
        client = TestClient(app)
        resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 0
    assert data["edge_count"] == 0
    assert data["intent_missing_count"] == 0
    assert data["intent_success_rate"] == 1.0
    assert data["filtered_non_ui_edges"] == 0
    assert data["mapping_stopped"] is None
    assert data["stop_reason"] is None
    assert data["semantic_consistency_rate"] == 1.0
    assert data["inventory_non_empty_rate"] == 0.0
    assert data["re_infer_success_rate"] is None


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
    G.graph["semantic_consistency_rate"] = 0.75
    G.graph["inventory_non_empty_rate"] = 1.0
    G.graph["re_infer_success_rate"] = 0.5
    G.graph["business_template_count"] = 1
    G.graph["business_template_generation_failures"] = 0

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 3
    assert data["edge_count"] == 2
    assert data["intent_missing_count"] == 1
    assert data["intent_success_rate"] == 0.5
    assert data["filtered_non_ui_edges"] == 3
    assert data["mapping_stopped"] is True
    assert data["stop_reason"] == "Stopped: unfillable form"
    assert data["semantic_consistency_rate"] == 0.75
    assert data["inventory_non_empty_rate"] == 1.0
    assert data["re_infer_success_rate"] == 0.5
    assert data["business_template_count"] == 1
    assert data["business_template_generation_failures"] == 0


def test_api_playback_uses_path_source_url_as_start_url():
    """Playback should start from the first path source URL, not fixed login default."""
    G = nx.DiGraph()
    G.add_node("state-start", url="https://a.com/start")
    G.add_node("state-next", url="https://a.com/next")
    edge_intent = Intent(
        raw="go next",
        verb="Click",
        object="Next",
        summary="Go next",
        key="navigation.go.next",
    )
    from graph_agent.models import GraphEdge

    edge_list = [
        GraphEdge(
            source="state-start",
            target="state-next",
            selector="#next",
            action=ActionType.CLICK,
            intent=edge_intent,
        )
    ]

    captured: dict[str, str | None] = {"start_url": None}

    async def _fake_sse(edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network):
        captured["start_url"] = start_url
        yield "data: {\"level\":\"success\"}\n\n"

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G), patch(
            "graph_agent.web.app.get_path_from_query", return_value=edge_list
        ), patch("graph_agent.web.app._sse_generator", side_effect=_fake_sse):
            client = TestClient(app)
            resp = client.post(
                "/api/playback",
                json={"intent": "go next", "test_data": {}},
            )
    assert resp.status_code == 200
    assert captured["start_url"] == "https://a.com/start"


def test_api_playback_uses_template_first_query_resolution():
    """Playback API should resolve path through template-aware query lookup."""
    G = nx.DiGraph()
    G.add_node("https://a.com/start", url="https://a.com/start")
    G.add_node("https://a.com/secure", url="https://a.com/secure")
    from graph_agent.models import GraphEdge

    edge_list = [
        GraphEdge(
            edge_id="step-1",
            source="https://a.com/start",
            target="https://a.com/secure",
            selector="#submit",
            action=ActionType.CLICK,
            intent=Intent(
                raw="submit login",
                verb="Submit",
                object="Login form",
                summary="Submit login form",
                key="auth.submit.login",
            ),
        )
    ]

    async def _fake_sse(edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network):
        yield "data: {\"level\":\"success\"}\n\n"

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G), patch(
            "graph_agent.web.app.get_path_from_query", return_value=edge_list
        ) as mock_query, patch("graph_agent.web.app._sse_generator", side_effect=_fake_sse):
            client = TestClient(app)
            resp = client.post(
                "/api/playback",
                json={"intent": "登录", "test_data": {}},
            )
    assert resp.status_code == 200
    mock_query.assert_called_once_with("登录", G)


def test_api_playback_passes_wait_for_network_to_sse_generator():
    """Playback API should forward wait configuration into the SSE worker."""
    G = nx.DiGraph()
    G.add_node("state-start", url="https://a.com/start")
    G.add_node("state-next", url="https://a.com/next")
    from graph_agent.models import GraphEdge

    edge_list = [
        GraphEdge(
            source="state-start",
            target="state-next",
            selector="#next",
            action=ActionType.CLICK,
            intent=Intent(
                raw="go next",
                verb="Click",
                object="Next",
                summary="Go next",
                key="navigation.go.next",
            ),
        )
    ]

    captured: dict[str, bool | None] = {"wait_for_network": None}

    async def _fake_sse(edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network):
        captured["wait_for_network"] = wait_for_network
        yield "data: {\"level\":\"success\"}\n\n"

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G), patch(
            "graph_agent.web.app.get_path_from_query", return_value=edge_list
        ), patch("graph_agent.web.app._sse_generator", side_effect=_fake_sse):
            client = TestClient(app)
            resp = client.post(
                "/api/playback",
                json={"intent": "go next", "test_data": {}, "wait_for_network": True},
            )
    assert resp.status_code == 200
    assert captured["wait_for_network"] is True


def test_api_playback_expands_template_dependencies():
    """Playback API should stream the full prerequisite-expanded template path."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("dashboard", url="https://a.com/dashboard")
    G.add_edge(
        "login",
        "secure",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#login",
        action=ActionType.CLICK,
        intent=Intent(
            raw="submit login",
            verb="Click",
            object="Login",
            summary="Submit login form",
            key="auth.submit.login",
        ),
    )
    G.add_edge(
        "secure",
        "dashboard",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#project",
        action=ActionType.CLICK,
        intent=Intent(
            raw="open dashboard",
            verb="Click",
            object="Project dashboard",
            summary="Open project dashboard",
            key="project.dashboard.open",
        ),
    )
    G.graph["business_templates"] = [
        {
            "template_id": "tpl-auth-login",
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "entry_node": "login",
            "exit_node": "secure",
            "path_length": 1,
            "confidence": 0.95,
            "steps": [{"edge_id": "step-1", "source": "login", "target": "secure", "selector": "#login", "action": "click", "intent_key": "auth.submit.login", "param_name": None}],
            "slots": {"submit": 0},
            "evidence": {},
        },
        {
            "template_id": "tpl-project-dashboard",
            "business_key": "project.dashboard.open",
            "summary": "打开项目看板",
            "entry_node": "secure",
            "exit_node": "dashboard",
            "path_length": 1,
            "confidence": 0.9,
            "steps": [{"edge_id": "step-2", "source": "secure", "target": "dashboard", "selector": "#project", "action": "click", "intent_key": "project.dashboard.open", "param_name": None}],
            "slots": {},
            "evidence": {},
            "depends_on": ["auth.login"],
        },
    ]

    captured: dict[str, list[str] | None] = {"edge_ids": None}

    async def _fake_sse(edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network):
        captured["edge_ids"] = [edge.edge_id for edge in edge_list_arg]
        yield "data: {\"level\":\"success\"}\n\n"

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G), patch(
            "graph_agent.web.app._sse_generator", side_effect=_fake_sse
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/playback",
                json={"intent": "project.dashboard.open", "test_data": {}},
            )
    assert resp.status_code == 200
    assert captured["edge_ids"] == ["step-1", "step-2"]


def test_api_intents_returns_templates_only_when_available():
    """Template options should replace atomic intents when business templates exist."""
    G = nx.MultiDiGraph()
    G.add_node("a", url="https://a.com/login")
    G.add_node("b", url="https://a.com/secure")
    G.add_edge(
        "a",
        "b",
        selector="#login",
        action=ActionType.CLICK,
        intent=Intent(
            raw="submit login",
            verb="Submit",
            object="Login form",
            summary="Submit login form",
            key="auth.submit.login",
            confidence=0.8,
        ),
        edge_id="step-1",
    )
    G.graph["business_templates"] = [
        {
            "template_id": "tpl-auth-login",
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "entry_node": "a",
            "exit_node": "b",
            "path_length": 3,
            "confidence": 0.95,
            "steps": [{"edge_id": "step-1", "source": "a", "target": "b", "selector": "#login", "action": "click", "intent_key": "auth.submit.login", "param_name": None}],
            "slots": {"submit": "step-1"},
            "evidence": {},
        }
    ]

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["type"] == "template"
    assert data[0]["value"] == "auth.login"
    assert data[0]["path_length"] == 3


def test_api_intents_falls_back_to_atomic_intents_without_templates():
    """Atomic intent options should remain available when no business templates exist."""
    G = nx.MultiDiGraph()
    G.add_node("a", url="https://a.com/login")
    G.add_node("b", url="https://a.com/secure")
    G.add_edge(
        "a",
        "b",
        selector="#login",
        action=ActionType.CLICK,
        intent=Intent(
            raw="submit login",
            verb="Submit",
            object="Login form",
            summary="Submit login form",
            key="auth.submit.login",
            confidence=0.8,
        ),
        edge_id="step-1",
    )

    with patch("graph_agent.web.app.GRAPH_PATH") as mock_path:
        mock_path.exists.return_value = True
        with patch("graph_agent.web.app.load_graph", return_value=G):
            client = TestClient(app)
            resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["type"] == "intent"
    assert data[0]["value"] == "auth.submit.login"

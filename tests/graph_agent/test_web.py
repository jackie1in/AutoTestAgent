"""Tests for web API (T8: API 与前端展示联动)."""

from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from graph_agent.models import ActionType, Intent, GraphEdge
from graph_agent.web.app import app


# -- Fake Neo4j async driver helpers --


class FakeRecord:
    """Mock Neo4j record that supports dict() conversion and .get()."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


class FakeResult:
    """Mock Neo4j result that supports async iteration and .single()."""

    def __init__(self, records):
        self._records = records
        self._idx = 0

    async def single(self):
        return self._records[0] if self._records else None

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= len(self._records):
            raise StopAsyncIteration
        r = self._records[self._idx]
        self._idx += 1
        return r


class FakeSession:
    """Mock Neo4j session that routes queries to pre-configured results."""

    def __init__(self, query_to_results):
        self._query_to_results = query_to_results

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, **kwargs):
        for q, results in self._query_to_results.items():
            if query.strip().startswith(q.strip()):
                return results  # already a FakeResult
        return FakeResult([])


class FakeDriver:
    """Mock Neo4j async driver.

    ``session()`` is intentionally a regular method (not async) because
    the caller does ``async with driver.session() as session:`` which
    expects the *result* of ``driver.session()`` to be an async context
    manager — not a coroutine.
    """

    def __init__(self, query_to_results):
        self._query_to_results = query_to_results

    def session(self):
        return FakeSession(self._query_to_results)


def _make_fake_driver(query_to_results):
    return FakeDriver(query_to_results)


# -- Tests that don't need graph mocking --


def test_auth_login_route_removed():
    """POST /api/auth/login should not exist anymore."""
    client = TestClient(app)
    resp = client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    assert resp.status_code in (404, 405)


def test_api_playback_no_auth_required():
    """POST /api/playback no longer returns 401 without token."""
    client = TestClient(app)
    resp = client.post("/api/playback", json={"intent": "x", "test_data": {}})
    assert resp.status_code != 401


def test_index_page_contains_template_links_section():
    """Index page should include a dedicated template dependency section."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="template-links"' in resp.text
    assert "前置业务关联" in resp.text


# -- /api/graph tests (mock _get_graph_from_neo4j) --


def test_api_graph_no_auth_required():
    """GET /api/graph returns 200 without token."""
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_graph_from_neo4j",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200


def test_api_graph_returns_intent_failure_reason():
    """T8: /api/graph returns intent_failure_reason on edges for diagnostics."""
    mock_data = {
        "nodes": [
            {"id": "a", "url": "https://a.com"},
            {"id": "b", "url": "https://b.com"},
        ],
        "edges": [
            {
                "edge_id": "e1",
                "source": "a",
                "target": "b",
                "selector": "#btn",
                "action": "click",
                "intent": None,
                "intent_failure_reason": "LLM output format invalid",
                "param_name": None,
                "action_value": None,
            }
        ],
        "missing_count": 1,
        "failure_reasons": ["LLM output format invalid"],
        "business_templates": [],
        "metadata": {
            "filtered_non_ui_edges": 0,
            "intent_missing_count": 1,
            "intent_success_rate": 0.0,
            "business_template_count": 0,
            "business_template_generation_failures": 0,
            "semantic_consistency_rate": 1.0,
            "inventory_non_empty_rate": 0.0,
            "re_infer_success_rate": None,
            "runtime_non_ui_action_count": 0,
            "state_like_node_ratio": 0.0,
            "business_intent_edge_ratio": 0.0,
            "multi_edge_preserved_count": 0,
        },
    }
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_graph_from_neo4j",
            new_callable=AsyncMock,
            return_value=mock_data,
        ),
    ):
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["edges"]) == 1
    assert data["edges"][0]["intent"] is None
    assert data["edges"][0]["intent_failure_reason"] == "LLM output format invalid"


def test_api_graph_returns_missing_count_and_failure_reasons():
    """T8: /api/graph returns missing_count and failure_reasons for frontend diagnostics."""
    mock_data = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {
                "edge_id": "e1",
                "source": "a",
                "target": "b",
                "selector": "#x",
                "action": "click",
                "intent": None,
                "intent_failure_reason": "reason1",
                "param_name": None,
                "action_value": None,
            },
            {
                "edge_id": "e2",
                "source": "b",
                "target": "c",
                "selector": "#y",
                "action": "fill",
                "intent": None,
                "intent_failure_reason": "reason1",
                "param_name": None,
                "action_value": None,
            },
        ],
        "missing_count": 2,
        "failure_reasons": ["reason1"],
        "business_templates": [],
        "metadata": {
            "filtered_non_ui_edges": 0,
            "intent_missing_count": 2,
            "intent_success_rate": 0.0,
            "business_template_count": 0,
            "business_template_generation_failures": 0,
            "semantic_consistency_rate": 1.0,
            "inventory_non_empty_rate": 0.0,
            "re_infer_success_rate": None,
            "runtime_non_ui_action_count": 0,
            "state_like_node_ratio": 0.0,
            "business_intent_edge_ratio": 0.0,
            "multi_edge_preserved_count": 0,
        },
    }
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_graph_from_neo4j",
            new_callable=AsyncMock,
            return_value=mock_data,
        ),
    ):
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["missing_count"] == 2
    assert data["failure_reasons"] == ["reason1"]
    assert "metadata" in data


def test_api_graph_returns_param_name_action_value():
    """Graph API should expose fill metadata fields."""
    mock_data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {
                "edge_id": "e1",
                "source": "a",
                "target": "b",
                "selector": "[name='username']",
                "action": "fill",
                "intent": None,
                "intent_failure_reason": None,
                "param_name": "username",
                "action_value": "tomsmith",
            }
        ],
        "missing_count": 1,
        "failure_reasons": [],
        "business_templates": [],
        "metadata": {
            "filtered_non_ui_edges": 0,
            "intent_missing_count": 1,
            "intent_success_rate": 0.0,
            "business_template_count": 0,
            "business_template_generation_failures": 0,
            "semantic_consistency_rate": 1.0,
            "inventory_non_empty_rate": 0.0,
            "re_infer_success_rate": None,
            "runtime_non_ui_action_count": 0,
            "state_like_node_ratio": 0.0,
            "business_intent_edge_ratio": 0.0,
            "multi_edge_preserved_count": 0,
        },
    }
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_graph_from_neo4j",
            new_callable=AsyncMock,
            return_value=mock_data,
        ),
    ):
        client = TestClient(app)
        resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edges"][0]["param_name"] == "username"
    assert data["edges"][0]["action_value"] == "tomsmith"


def test_api_graph_empty_on_neo4j_failure():
    """Graph returns empty fallback when Neo4j query fails."""
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_graph_from_neo4j",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
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


# -- /api/intents tests --


def test_api_intents_no_auth_required():
    """GET /api/intents returns 200 without token."""
    fake_driver = _make_fake_driver(
        {"MATCH (a:App)": FakeResult([])}
    )
    with patch(
        "graph_agent.web.app._get_driver",
        new_callable=AsyncMock,
        return_value=fake_driver,
    ):
        client = TestClient(app)
        resp = client.get("/api/intents")
    assert resp.status_code == 200


def test_api_intents_ignores_null_intent_edges():
    """T8: /api/intents excludes edges with intent=null from the dropdown options."""
    fake_driver = _make_fake_driver(
        {
            "MATCH (a:App)": FakeResult(
                [
                    FakeRecord(
                        {
                            "key": "submit_login",
                            "summary": "Click login button",
                            "confidence": 0.9,
                        }
                    ),
                ]
            ),
        }
    )
    with patch(
        "graph_agent.web.app._get_driver",
        new_callable=AsyncMock,
        return_value=fake_driver,
    ):
        client = TestClient(app)
        resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["value"] == "submit_login"
    assert "submit_login" in (data[0].get("label") or "")


# -- /api/dashboard tests --


def test_api_dashboard_no_auth_required():
    """GET /api/dashboard returns 200 without token."""
    fake_driver = _make_fake_driver(
        {
            "MATCH (a:App)": FakeResult(
                [FakeRecord({"node_count": 0, "edge_count": 0, "missing_count": 0})]
            ),
        }
    )
    with patch(
        "graph_agent.web.app._get_driver",
        new_callable=AsyncMock,
        return_value=fake_driver,
    ):
        client = TestClient(app)
        resp = client.get("/api/dashboard")
    assert resp.status_code == 200


def test_api_dashboard_empty_on_neo4j_failure():
    """Dashboard returns zeroed stats when Neo4j is unavailable."""
    with patch(
        "graph_agent.web.app._get_driver",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Neo4j down"),
    ):
        client = TestClient(app)
        resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 0
    assert data["edge_count"] == 0
    assert data["intent_missing_count"] == 0
    assert data["intent_success_rate"] == 1.0


def test_api_dashboard_returns_stats():
    """Dashboard returns graph statistics with metadata."""
    fake_driver = _make_fake_driver(
        {
            "MATCH (a:App)": FakeResult(
                [FakeRecord({"node_count": 3, "edge_count": 2, "missing_count": 1})]
            ),
        }
    )
    with patch(
        "graph_agent.web.app._get_driver",
        new_callable=AsyncMock,
        return_value=fake_driver,
    ):
        client = TestClient(app)
        resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 3
    assert data["edge_count"] == 2
    assert data["intent_missing_count"] == 1
    assert data["intent_success_rate"] == 0.5


# -- /api/playback tests --


def test_api_playback_uses_path_source_url_as_start_url():
    """Playback should start from the first path source URL, not fixed login default."""
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
            source_url="https://a.com/start",
        )
    ]

    captured = {"start_url": None}

    async def _fake_sse(
        edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network
    ):
        captured["start_url"] = start_url
        yield 'data: {"level":"success"}\n\n'

    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_edges_from_neo4j",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("graph_agent.web.app.get_path_from_query", return_value=edge_list),
        patch("graph_agent.web.app._sse_generator", side_effect=_fake_sse),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/playback",
            json={"intent": "go next", "test_data": {}},
        )
    assert resp.status_code == 200
    assert captured["start_url"] == "https://a.com/start"


def test_api_playback_passes_wait_for_network_to_sse_generator():
    """Playback API should forward wait configuration into the SSE worker."""
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

    captured = {"wait_for_network": None}

    async def _fake_sse(
        edge_list_arg, test_data_arg, start_url, expected_end_url, wait_for_network
    ):
        captured["wait_for_network"] = wait_for_network
        yield 'data: {"level":"success"}\n\n'

    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_edges_from_neo4j",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("graph_agent.web.app.get_path_from_query", return_value=edge_list),
        patch("graph_agent.web.app._sse_generator", side_effect=_fake_sse),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/playback",
            json={"intent": "go next", "test_data": {}, "wait_for_network": True},
        )
    assert resp.status_code == 200
    assert captured["wait_for_network"] is True


def test_api_playback_returns_error_when_no_path():
    """Playback returns error SSE when no matching path found."""
    with (
        patch(
            "graph_agent.web.app._get_driver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.web.app._get_edges_from_neo4j",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("graph_agent.web.app.get_path_from_query", return_value=[]),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/playback",
            json={"intent": "nonexistent", "test_data": {}},
        )
    assert resp.status_code == 200
    assert "no matching path" in resp.text

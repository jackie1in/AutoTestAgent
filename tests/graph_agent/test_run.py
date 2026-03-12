"""Tests for Mapping run module (T4:构图与统计联动, T5:re-infer-missing)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import networkx as nx

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.mapping.run import (
    _build_graph_from_history,
    _build_mapping_task_with_env_hints,
    _compute_snapshot_delta,
    FILTERED_ACTION_KEYS,
    re_infer_missing_intents,
    _resolve_snapshot_path,
    _resolve_target_state,
    _runtime_filter_snapshots,
    _semantic_consistency,
    _write_acceptance_snapshot,
)
from graph_agent.models import ActionType, BusinessTemplate, BusinessTemplateStep, FrameLocatorSnapshot, Intent, TabActionType, TabSnapshot


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
            "param_name": None,
            "action_value": None,
            "element": None,
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
            "param_name": None,
            "action_value": None,
            "element": None,
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


@pytest.mark.asyncio
async def test_build_graph_records_context_level_used():
    """Graph edge should preserve parser context_level_used for observability."""
    history = _make_mock_history(
        actions=[{"click": {"element": "button"}, "interacted_element": {"xpath": "//button[@id='ok']"}}],
        thoughts=[{"next_goal": "Click confirm"}],
        urls=["https://a.com", "https://b.com"],
    )
    edge_with_level = type(
        "EdgeModel",
        (),
        {
            "selector": "xpath=//button[@id='ok']",
            "action": ActionType.CLICK,
            "intent": Intent(summary="Confirm action", raw="confirm", verb="Click", object="Confirm"),
            "context_level_used": "L1",
            "intent_failure_reason": None,
            "param_name": None,
            "action_value": None,
            "element": None,
            "constraints": None,
        },
    )()
    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        return_value=edge_with_level,
    ):
        G = await _build_graph_from_history(history)

    assert G.number_of_edges() == 1
    _u, _v, data = next(iter(G.edges(data=True)))
    assert data["context_level_used"] == "L1"


@pytest.mark.asyncio
async def test_build_graph_tab_context_preserved_on_edge():
    history = _make_mock_history(
        actions=[{"click": {"element": "button"}, "interacted_element": {"css_selector": "#quality"}}],
        thoughts=[{"next_goal": "Open quality page"}],
        urls=["https://example.com/home", "https://example.com/quality"],
    )
    edge_with_tab_context = type(
        "EdgeModel",
        (),
        {
            "selector": "#quality",
            "action": ActionType.CLICK,
            "intent": Intent(
                summary="Open quality page",
                raw="open quality",
                verb="Open",
                object="Quality page",
            ),
            "tab_id": "tab-0",
            "target_tab_id": "tab-1",
            "tab_action": TabActionType.OPEN,
            "tab": TabSnapshot(
                tab_id="tab-1",
                opener_tab_id="tab-0",
                url="https://example.com/quality",
                title="Quality",
            ),
            "frame_path": [
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
            "context_level_used": "L0",
            "intent_failure_reason": None,
            "param_name": None,
            "action_value": None,
            "element": None,
            "constraints": None,
        },
    )()
    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        return_value=edge_with_tab_context,
    ):
        G = await _build_graph_from_history(history)

    assert G.number_of_edges() == 1
    _u, _v, data = next(iter(G.edges(data=True)))
    assert data["tab_id"] == "tab-0"
    assert data["target_tab_id"] == "tab-1"
    assert data["tab_action"] == TabActionType.OPEN
    assert data["tab"].tab_id == "tab-1"
    assert data["tab"].opener_tab_id == "tab-0"
    assert data["tab"].url == "https://example.com/quality"
    assert data["tab"].title == "Quality"
    assert [frame.selector for frame in data["frame_path"]] == [
        "iframe[name='outer']",
        "iframe[name='inner']",
    ]


@pytest.mark.asyncio
async def test_build_graph_uses_opaque_state_ids_for_same_url_steps():
    """Same URL form states should not collapse to one URL-keyed node."""
    history = _make_mock_history(
        actions=[
            {"input_text": {"text": "alice"}, "interacted_element": {"attributes": {"id": "username"}}},
            {"input_text": {"text": "secret"}, "interacted_element": {"attributes": {"id": "password"}}},
        ],
        thoughts=[
            {"next_goal": "Fill username"},
            {"next_goal": "Fill password"},
        ],
        urls=[
            "https://example.com/login",
            "https://example.com/login",
            "https://example.com/login",
        ],
    )
    fill_user = type(
        "EdgeModel",
        (),
        {
            "selector": "#username",
            "action": ActionType.FILL,
            "intent": Intent(summary="Fill username", raw="fill user", verb="Fill", object="Username", key="auth.fill.username"),
            "intent_failure_reason": None,
            "param_name": "username",
            "action_value": "alice",
            "element": None,
            "constraints": None,
        },
    )()
    fill_pass = type(
        "EdgeModel",
        (),
        {
            "selector": "#password",
            "action": ActionType.FILL,
            "intent": Intent(summary="Fill password", raw="fill pass", verb="Fill", object="Password", key="auth.fill.password"),
            "intent_failure_reason": None,
            "param_name": "password",
            "action_value": "secret",
            "element": None,
            "constraints": None,
        },
    )()

    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        side_effect=[fill_user, fill_pass],
    ):
        G = await _build_graph_from_history(history)

    assert G.number_of_edges() == 2
    assert G.number_of_nodes() >= 3
    urls = [data.get("url") for _, data in G.nodes(data=True)]
    assert urls.count("https://example.com/login") >= 3
    assert all(not str(node_id).startswith("https://") for node_id in G.nodes)


# --- T5: re-infer-missing ---


def _make_graph_with_missing_intents(tmp_path: Path) -> Path:
    """Create a graph file with edges having intent=None for re-infer tests."""
    G: nx.DiGraph = nx.DiGraph()
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
        param_name=None,
        constraints=None,
    )
    G.add_edge(
        "https://b.com",
        "https://c.com",
        selector="xpath=//input[@id='y']",
        action=ActionType.FILL,
        intent=None,
        intent_failure_reason="llm_timeout",
        param_name=None,
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
    ab_multi = G.get_edge_data("https://a.com", "https://b.com") or {}
    ab_data = next(iter(ab_multi.values()))
    assert ab_data["intent"] is not None
    assert ab_data["intent_failure_reason"] is None
    # b->c should stay None (second call failed)
    bc_multi = G.get_edge_data("https://b.com", "https://c.com") or {}
    bc_data = next(iter(bc_multi.values()))
    assert bc_data["intent"] is None
    assert bc_data["intent_failure_reason"] == "parse_error"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_refreshes_business_templates(tmp_path: Path):
    """Successful re-infer should regenerate business templates before saving graph."""
    graph_path = _make_graph_with_missing_intents(tmp_path)
    refreshed = [
        BusinessTemplate(
            template_id="tpl-auth-login",
            business_key="auth.login",
            summary="登录流程",
            entry_node="https://a.com",
            exit_node="https://c.com",
            path_length=2,
            confidence=0.91,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-1",
                    source="https://a.com",
                    target="https://b.com",
                    selector="xpath=//button[@id='x']",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                )
            ],
            slots={"submit": "step-1"},
            evidence={"intent_keys": ["auth.submit.login"]},
        )
    ]
    expected_intent = Intent(
        raw="submit login",
        verb="Submit",
        object="Login form",
        summary="Submit login form",
        key="auth.submit.login",
        confidence=0.93,
    )

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ), patch(
        "graph_agent.mapping.run.generate_business_templates",
        new_callable=AsyncMock,
        return_value=refreshed,
    ) as mock_templates:
        stats = await re_infer_missing_intents(graph_path)

    assert stats["succeeded"] == 2
    mock_templates.assert_awaited_once()
    G = load_graph(graph_path)
    assert G.graph["business_template_count"] == 1
    assert G.graph["business_templates"][0]["business_key"] == "auth.login"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_tolerates_template_generation_failure(tmp_path: Path):
    """Template generation failure should not block saving successful re-infer results."""
    graph_path = _make_graph_with_missing_intents(tmp_path)
    expected_intent = Intent(
        raw="submit login",
        verb="Submit",
        object="Login form",
        summary="Submit login form",
        key="auth.submit.login",
        confidence=0.93,
    )

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ), patch(
        "graph_agent.mapping.run.generate_business_templates",
        new_callable=AsyncMock,
        side_effect=RuntimeError("template llm failed"),
    ):
        stats = await re_infer_missing_intents(graph_path)

    assert stats["succeeded"] == 2
    G = load_graph(graph_path)
    assert G.graph["business_template_count"] == 0
    assert G.graph["business_template_generation_failures"] == 1


@pytest.mark.asyncio
async def test_re_infer_missing_intents_uses_node_url_metadata(tmp_path: Path):
    """Re-infer should pass real page URLs, not opaque state ids, into intent inference."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("state-login-empty", label="login", url="https://example.com/login")
    G.add_node("state-login-filled", label="login-filled", url="https://example.com/login")
    G.add_edge(
        "state-login-empty",
        "state-login-filled",
        key="step-1",
        edge_id="step-1",
        selector="#username",
        action=ActionType.FILL,
        intent=None,
        intent_failure_reason="missing",
        param_name="username",
    )
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    expected_intent = Intent(
        raw="fill username",
        verb="Fill",
        object="Username",
        summary="Fill username",
        key="auth.fill.username",
        confidence=0.88,
    )

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ) as mock_infer:
        await re_infer_missing_intents(graph_path)

    kwargs = mock_infer.await_args.kwargs
    assert kwargs["source_url"] == "https://example.com/login"
    assert kwargs["target_url"] == "https://example.com/login"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_skips_non_null(tmp_path: Path):
    """T5: Re-infer skips edges that already have intent."""
    G: nx.DiGraph = nx.DiGraph()
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


def test_resolve_target_state_looks_ahead_for_real_url():
    """When immediate next URL is missing, resolver should look ahead."""
    actions = [{"click": {}}, {"done": {}}, {"click": {}}]
    thoughts = [{}, {}, {}]
    urls = ["https://a.com/login", "", "https://a.com/secure"]
    target_state, target_url = _resolve_target_state(0, urls, thoughts, actions)
    assert target_state.startswith("state-")
    assert target_url == "https://a.com/secure"


def test_runtime_filter_snapshots_stabilizes_missing_urls():
    actions = [{"click": {}}, {"click": {}}, {"click": {}}]
    thoughts = [{}, {}, {}]
    urls = ["https://a.com/start", "", "", "https://a.com/next"]
    out_actions, out_thoughts, out_urls, filtered = _runtime_filter_snapshots(actions, thoughts, urls)
    assert filtered == 0
    assert len(out_actions) == 3
    assert len(out_thoughts) == 3
    assert out_urls == [
        "https://a.com/start",
        "https://a.com/start",
        "https://a.com/start",
        "https://a.com/next",
    ]


def test_semantic_consistency_fill_allows_navigation_wording_on_input_selector():
    intent = Intent(
        raw="to login",
        verb="Click",
        object="Form Authentication",
        summary="Click form authentication link to navigate to login",
        key="auth.navigation.login",
        confidence=0.8,
    )
    assert _semantic_consistency(ActionType.FILL, intent, selector="xpath=//input[@id='username']") is True


def test_semantic_consistency_click_accepts_navigation_intent():
    intent = Intent(
        raw="go checkboxes",
        verb="Navigate",
        object="Checkboxes",
        summary="Navigate to checkboxes page",
        key="elements.select.checkboxes",
        confidence=0.9,
    )
    assert _semantic_consistency(ActionType.CLICK, intent, selector="xpath=//a[@href='/checkboxes']") is True


def test_semantic_consistency_navigate_accepts_navigation_key():
    intent = Intent(
        raw="open home",
        verb="Open",
        object="Home page",
        summary="Navigate back to home page",
        key="navigation.return.home",
        confidence=0.9,
    )
    assert _semantic_consistency(ActionType.NAVIGATE, intent, selector="") is True


def test_compute_snapshot_delta_with_previous_metrics():
    previous = {
        "metrics": {
            "nodes": 10,
            "edges": 8,
            "semantic_consistency_rate": 0.8,
        }
    }
    current = {
        "nodes": 12.0,
        "edges": 9.0,
        "semantic_consistency_rate": 1.0,
    }
    delta = _compute_snapshot_delta(previous, current)
    assert delta["nodes"] == 2.0
    assert delta["edges"] == 1.0
    assert delta["semantic_consistency_rate"] == 0.2


def test_build_mapping_task_without_env_login_hints():
    with patch.dict("os.environ", {}, clear=True):
        text = _build_mapping_task_with_env_hints(None, "https://example.com/login")
    assert "MAPPING_USERNAME" not in text
    assert "username=" not in text
    assert "password=" not in text


def test_build_mapping_task_with_env_login_hints():
    with patch.dict(
        "os.environ",
        {"MAPPING_USERNAME": "tomsmith", "MAPPING_PASSWORD": "SuperSecretPassword!"},
        clear=True,
    ):
        text = _build_mapping_task_with_env_hints(None, "https://example.com/login")
    assert "username=tomsmith" in text
    assert "password=SuperSecretPassword!" in text


def test_task_template_includes_derived_exploration_hint():
    """Task 2: Default task template should encourage exploring derived pages."""
    from graph_agent.mapping.run import DEFAULT_TASK_TEMPLATE
    task = DEFAULT_TASK_TEMPLATE.format(start_url="https://example.com/")
    assert "派生" in task
    assert "继续探索" in task
    assert "菜单" in task or "列表" in task or "详情" in task


def test_write_acceptance_snapshot_creates_file(tmp_path: Path):
    graph_path = tmp_path / "graph.json"
    inventory_path = tmp_path / "element_inventory.json"
    inventory_path.write_text(
        '{"mode":"multi_page","metadata":{"page_count":2,"aggregated_element_count":3,"type_counts":{"input":1}}}',
        encoding="utf-8",
    )
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("a", label="a", url="a")
    G.add_node("b", label="b", url="b")
    G.add_edge("a", "b", action=ActionType.CLICK, selector="#x", intent=Intent(raw="x", verb="Click", object="X", summary="Click x"))
    G.graph["intent_missing_count"] = 0
    G.graph["intent_success_rate"] = 1.0
    G.graph["semantic_consistency_rate"] = 1.0
    G.graph["business_intent_edge_ratio"] = 1.0
    G.graph["runtime_non_ui_action_count"] = 0
    G.graph["state_like_node_ratio"] = 0.0
    G.graph["re_infer_success_rate"] = 1.0

    snapshot_path = _write_acceptance_snapshot(
        graph=G,
        graph_output_path=graph_path,
        inventory_path=inventory_path,
        target_url="https://example.com",
    )

    assert snapshot_path == _resolve_snapshot_path(graph_path)
    assert snapshot_path.exists()
    raw = snapshot_path.read_text(encoding="utf-8")
    assert '"target_url": "https://example.com"' in raw
    assert '"has_previous": false' in raw

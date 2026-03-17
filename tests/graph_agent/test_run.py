"""Tests for Mapping run module (T4:构图与统计联动, T5:re-infer-missing)."""

import asyncio
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
    _resolve_mapping_channel,
    _resolve_mapping_headless,
    _resolve_intent_context_window,
    re_infer_missing_intents,
    re_infer_with_feedback,
    _resolve_snapshot_path,
    _resolve_target_state,
    _runtime_filter_snapshots,
    _semantic_consistency,
    _write_acceptance_snapshot,
)
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    ElementSnapshot,
    FrameLocatorSnapshot,
    Intent,
    TabActionType,
    TabSnapshot,
)


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


def test_collect_playback_diagnostics_includes_path_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Task 1: diagnostics should expose resolved path context and playback result."""
    from graph_agent.acceptance.playback_diagnostics import (
        collect_playback_diagnostics,
    )

    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node("home", url="https://example.com/")
    graph.add_node("login", url="https://example.com/login")
    graph.add_edge(
        "home",
        "login",
        key="e1",
        edge_id="e1",
        selector='a[href="/login"]',
        action=ActionType.CLICK,
        tab_id="tab-0",
        frame_path=[],
        intent=Intent(
            raw="Go to login",
            verb="Go",
            object="Login",
            summary="Go to login",
            key="auth.login",
        ),
    )
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)

    async def _fake_run_playback(*args, **kwargs):
        return {
            "success": False,
            "actual_url": "https://example.com/login",
            "error": "selector: missing",
        }

    monkeypatch.setattr(
        "graph_agent.acceptance.playback_diagnostics.run_playback",
        _fake_run_playback,
    )

    report = asyncio.run(
        collect_playback_diagnostics(
            graph_path=graph_path,
            intent_query="auth.login",
            start_url="https://example.com/",
            test_data={},
        )
    )

    assert report["intent_query"] == "auth.login"
    assert report["path_length"] == 1
    assert report["edges"][0]["tab_id"] == "tab-0"
    assert report["edges"][0]["selector"] == 'a[href="/login"]'
    assert report["playback"]["success"] is False
    assert report["root_cause"] == "selector"
    assert report["root_cause_detail"] == "selector.unknown"
    assert report["failed_step_index"] is None
    assert report["failed_edge_id"] is None
    assert "failure_output_path" in report
    assert Path(report["failure_output_path"]).exists()


def test_get_path_from_query_prefers_executable_path_with_prerequisites():
    """Task 2: template-expanded query path should include required prerequisite steps."""
    from graph_agent.graph.pathfinding import get_path_from_query
    from graph_agent.graph.templates import store_business_templates

    graph = nx.MultiDiGraph()
    graph.add_node("entry", url="https://example.com/login")
    graph.add_node("dashboard", url="https://example.com/dashboard")
    graph.add_node("module", url="https://example.com/module")

    login_intent = Intent(
        raw="Login",
        verb="Submit",
        object="Login",
        summary="Login",
        key="auth.login",
        confidence=0.9,
    )
    module_intent = Intent(
        raw="Open module",
        verb="Open",
        object="Module",
        summary="登录后进入目标模块",
        key="navigation.module.select",
        confidence=0.9,
    )

    graph.add_edge(
        "entry",
        "dashboard",
        key="e-login",
        edge_id="e-login",
        selector="#submit",
        action=ActionType.CLICK,
        intent=login_intent,
    )
    graph.add_edge(
        "dashboard",
        "module",
        key="e-module",
        edge_id="e-module",
        selector="#module",
        action=ActionType.CLICK,
        intent=module_intent,
    )

    templates = [
        BusinessTemplate(
            template_id="t-login",
            business_key="auth.login",
            summary="Login",
            entry_node="entry",
            exit_node="dashboard",
            path_length=1,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="e-login",
                    source="entry",
                    target="dashboard",
                    selector="#submit",
                    action=ActionType.CLICK,
                    intent_key="auth.login",
                )
            ],
        ),
        BusinessTemplate(
            template_id="t-module",
            business_key="navigation.module.select",
            summary="登录后进入目标模块",
            entry_node="dashboard",
            exit_node="module",
            path_length=1,
            confidence=0.9,
            depends_on=["auth.login"],
            steps=[
                BusinessTemplateStep(
                    edge_id="e-module",
                    source="dashboard",
                    target="module",
                    selector="#module",
                    action=ActionType.CLICK,
                    intent_key="navigation.module.select",
                )
            ],
        ),
    ]

    store_business_templates(graph, templates)
    path = get_path_from_query("登录后进入目标模块", graph)

    assert [edge.edge_id for edge in path] == ["e-login", "e-module"]


def test_get_path_from_atomic_intent_keeps_required_same_state_prerequisites():
    """Task 3: atomic intent search should retain earlier same-state prerequisite edges."""
    from graph_agent.graph.pathfinding import get_path_from_query

    graph = nx.MultiDiGraph()
    graph.add_node("state", url="https://example.com/dashboard")
    graph.add_node("done", url="https://example.com/done")

    prereq_intent = Intent(
        raw="Accept terms",
        verb="Accept",
        object="Terms",
        summary="Accept terms",
        key="form.accept.terms",
        confidence=0.9,
    )
    target_intent = Intent(
        raw="Submit form",
        verb="Submit",
        object="Form",
        summary="Submit form",
        key="form.submit",
        confidence=0.9,
    )

    graph.add_edge(
        "state",
        "state",
        key="e-prereq",
        edge_id="e-prereq",
        step_index=1,
        selector="#accept",
        action=ActionType.CLICK,
        intent=prereq_intent,
    )
    graph.add_edge(
        "state",
        "done",
        key="e-submit",
        edge_id="e-submit",
        step_index=2,
        selector="#submit",
        action=ActionType.CLICK,
        intent=target_intent,
    )

    path = get_path_from_query("Submit form", graph, prefer_templates=False)

    assert [edge.edge_id for edge in path] == ["e-prereq", "e-submit"]


@pytest.mark.asyncio
async def test_build_graph_preserves_intent_and_failure_reason():
    """T4: Edges with intent=None and intent_failure_reason are preserved in graph."""
    history = _make_mock_history(
        actions=[
            {
                "click": {"element": "button"},
                "interacted_element": {"xpath": "//button[@id='x']"},
            },
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
        G = await _build_graph_from_history(history, intent_mode="sync")

    assert G.number_of_edges() == 1
    u, v, data = next(iter(G.edges(data=True)))
    assert data["intent"] is None
    assert data["intent_failure_reason"] == "parse_error:invalid_json"
    assert data["selector"] == "xpath=//button[@id='x']"
    assert data["action"] == ActionType.CLICK


@pytest.mark.asyncio
async def test_build_graph_uses_configurable_neighbor_window():
    history = _make_mock_history(
        actions=[
            {"click": {"element": "a"}, "interacted_element": {"xpath": "//a[@id='a']"}},
            {"click": {"element": "b"}, "interacted_element": {"xpath": "//a[@id='b']"}},
            {"click": {"element": "c"}, "interacted_element": {"xpath": "//a[@id='c']"}},
        ],
        thoughts=[
            {"next_goal": "Open module"},
            {"next_goal": "Open detail"},
            {"next_goal": "Open tab"},
        ],
        urls=[
            "https://a.com/home",
            "https://a.com/module",
            "https://a.com/detail",
            "https://a.com/tab",
        ],
    )

    async def _fake_parse(action, thought, source_url, target_url, **kwargs):
        selector = "xpath=" + str(
            (action.get("interacted_element") or {}).get("xpath") or ""
        )
        return type(
            "EdgeModel",
            (),
            {
                "selector": selector,
                "action": ActionType.CLICK,
                "intent": None,
                "intent_failure_reason": "missing",
                "param_name": None,
                "action_value": None,
                "element": None,
                "constraints": None,
                "frame_path": [],
            },
        )()

    with (
        patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "2"}, clear=False),
        patch(
            "graph_agent.mapping.run.parse_browser_use_step",
            new_callable=AsyncMock,
            side_effect=_fake_parse,
        ) as mock_parse,
    ):
        await _build_graph_from_history(history, intent_mode="sync")

    assert len(mock_parse.await_args_list) == 3
    first_neighbors = mock_parse.await_args_list[0].kwargs["neighbor_steps"]
    third_neighbors = mock_parse.await_args_list[2].kwargs["neighbor_steps"]
    assert first_neighbors == []
    assert len(third_neighbors) == 2
    assert third_neighbors[0]["thought"] == "Open module"
    assert third_neighbors[1]["thought"] == "Open detail"


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
            "intent": Intent(
                summary="Click btn", raw="click", verb="Click", object="Btn"
            ),
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
        G = await _build_graph_from_history(history, intent_mode="sync")

    assert G.graph["filtered_non_ui_edges"] == 2  # read_file, write_file
    assert G.number_of_edges() == 1
    assert "intent_alignment_warning_count" in G.graph
    assert "intent_alignment_warning_breakdown" in G.graph


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
        G = await _build_graph_from_history(history, intent_mode="sync")

    assert G.number_of_edges() == 0
    assert G.graph["filtered_non_ui_edges"] == 1


@pytest.mark.asyncio
async def test_build_graph_records_context_level_used():
    """Graph edge should preserve parser context_level_used for observability."""
    history = _make_mock_history(
        actions=[
            {
                "click": {"element": "button"},
                "interacted_element": {"xpath": "//button[@id='ok']"},
            }
        ],
        thoughts=[{"next_goal": "Click confirm"}],
        urls=["https://a.com", "https://b.com"],
    )
    edge_with_level = type(
        "EdgeModel",
        (),
        {
            "selector": "xpath=//button[@id='ok']",
            "action": ActionType.CLICK,
            "intent": Intent(
                summary="Confirm action", raw="confirm", verb="Click", object="Confirm"
            ),
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
        G = await _build_graph_from_history(history, intent_mode="sync")

    assert G.number_of_edges() == 1
    _u, _v, data = next(iter(G.edges(data=True)))
    assert data["context_level_used"] == "L1"


@pytest.mark.asyncio
async def test_build_graph_tab_context_preserved_on_edge():
    history = _make_mock_history(
        actions=[
            {
                "click": {"element": "button"},
                "interacted_element": {"css_selector": "#quality"},
            }
        ],
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
        G = await _build_graph_from_history(history, intent_mode="sync")

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
            {
                "input_text": {"text": "alice"},
                "interacted_element": {"attributes": {"id": "username"}},
            },
            {
                "input_text": {"text": "secret"},
                "interacted_element": {"attributes": {"id": "password"}},
            },
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
            "intent": Intent(
                summary="Fill username",
                raw="fill user",
                verb="Fill",
                object="Username",
                key="auth.fill.username",
            ),
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
            "intent": Intent(
                summary="Fill password",
                raw="fill pass",
                verb="Fill",
                object="Password",
                key="auth.fill.password",
            ),
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
        G = await _build_graph_from_history(history, intent_mode="sync")

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
async def test_re_infer_with_feedback_updates_targeted_edge(tmp_path: Path):
    """Playback feedback should target failed edge and pass hint into re-infer."""
    graph_path = _make_graph_with_missing_intents(tmp_path)
    failures_path = tmp_path / "playback_failures.json"
    failures_path.write_text(
        """
{
  "failures": [
    {
      "root_cause": "selector",
      "error": "selector: timeout",
      "failed_edge_id": "step-1",
      "edges": [
        {"edge_id": "step-1", "source": "https://a.com", "target": "https://b.com"}
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    expected_intent = Intent(
        raw="click submit",
        verb="Click",
        object="Submit",
        summary="Click submit button",
        key="auth.submit.login",
    )

    with patch(
        "graph_agent.mapping.run.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ) as mock_infer:
        stats = await re_infer_with_feedback(
            graph_path=graph_path,
            feedback=failures_path,
        )

    assert stats["total"] == 1
    assert stats["succeeded"] == 1
    assert mock_infer.await_args is not None
    kwargs = mock_infer.await_args.kwargs
    assert kwargs["playback_error_hint"] == "selector: timeout"
    G = load_graph(graph_path)
    ab_multi = G.get_edge_data("https://a.com", "https://b.com") or {}
    ab_data = next(iter(ab_multi.values()))
    assert ab_data["intent"] is not None
    bc_multi = G.get_edge_data("https://b.com", "https://c.com") or {}
    bc_data = next(iter(bc_multi.values()))
    assert bc_data["intent"] is None


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

    with (
        patch(
            "graph_agent.mapping.run.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(expected_intent, None),
        ),
        patch(
            "graph_agent.mapping.run.generate_business_templates",
            new_callable=AsyncMock,
            return_value=refreshed,
        ) as mock_templates,
    ):
        stats = await re_infer_missing_intents(graph_path)

    assert stats["succeeded"] == 2
    mock_templates.assert_awaited_once()
    G = load_graph(graph_path)
    assert G.graph["business_template_count"] == 1
    assert G.graph["business_templates"][0]["business_key"] == "auth.login"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_tolerates_template_generation_failure(
    tmp_path: Path,
):
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

    with (
        patch(
            "graph_agent.mapping.run.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(expected_intent, None),
        ),
        patch(
            "graph_agent.mapping.run.generate_business_templates",
            new_callable=AsyncMock,
            side_effect=RuntimeError("template llm failed"),
        ),
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
    G.add_node(
        "state-login-filled", label="login-filled", url="https://example.com/login"
    )
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

    assert mock_infer.await_args is not None
    kwargs = mock_infer.await_args.kwargs
    assert kwargs["source_url"] == "https://example.com/login"
    assert kwargs["target_url"] == "https://example.com/login"


@pytest.mark.asyncio
async def test_re_infer_missing_intents_uses_edge_thought_and_neighbors(tmp_path: Path):
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("s1", url="https://example.com/a")
    G.add_node("s2", url="https://example.com/b")
    G.add_node("s3", url="https://example.com/c")
    G.add_edge(
        "s1",
        "s2",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#first",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="missing",
        thought="open first section",
    )
    G.add_edge(
        "s2",
        "s3",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#second",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="missing",
        thought="open second section",
    )
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    with (
        patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "1"}, clear=False),
        patch(
            "graph_agent.mapping.run.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(None, "parse_error"),
        ) as mock_infer,
        patch(
            "graph_agent.mapping.run._refresh_business_templates",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await re_infer_missing_intents(graph_path)

    calls = {c.kwargs["selector"]: c.kwargs for c in mock_infer.await_args_list}
    assert calls["#first"]["thought_text"] == "open first section"
    assert calls["#first"]["neighbor_steps"] == []
    assert calls["#second"]["thought_text"] == "open second section"
    assert len(calls["#second"]["neighbor_steps"]) == 1
    assert calls["#second"]["neighbor_steps"][0]["selector"] == "#first"
    assert calls["#second"]["page_signals"]["action_key"] == "click"


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
    G.add_edge(
        "a", "b", selector="#btn", action=ActionType.CLICK, intent=existing_intent
    )
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
    out_actions, out_thoughts, out_urls, filtered = _runtime_filter_snapshots(
        actions, thoughts, urls
    )
    assert filtered == 0
    assert len(out_actions) == 3
    assert len(out_thoughts) == 3
    assert out_urls == [
        "https://a.com/start",
        "https://a.com/start",
        "https://a.com/start",
        "https://a.com/next",
    ]


def test_runtime_filter_snapshots_drops_wait_actions_to_preserve_path_continuity():
    """Wait actions should be removed before graph building to keep actionable states connected."""
    actions = [
        {"click": {"element": "login"}},
        {"wait": {"seconds": 3}},
        {"click": {"element": "project"}},
    ]
    thoughts = [
        {"next_goal": "Submit login"},
        {"next_goal": "Wait for dashboard"},
        {"next_goal": "Open project module"},
    ]
    urls = [
        "https://example.com/login",
        "https://example.com/dashboard",
        "https://example.com/dashboard",
        "https://example.com/project",
    ]

    out_actions, out_thoughts, out_urls, filtered = _runtime_filter_snapshots(
        actions, thoughts, urls
    )

    assert [next(iter(action.keys())) for action in out_actions] == ["click", "click"]
    assert [thought["next_goal"] for thought in out_thoughts] == [
        "Submit login",
        "Open project module",
    ]
    assert out_urls == [
        "https://example.com/login",
        "https://example.com/dashboard",
        "https://example.com/project",
    ]
    assert filtered == 1


def test_semantic_consistency_fill_allows_navigation_wording_on_input_selector():
    intent = Intent(
        raw="to login",
        verb="Click",
        object="Form Authentication",
        summary="Click form authentication link to navigate to login",
        key="auth.navigation.login",
        confidence=0.8,
    )
    assert (
        _semantic_consistency(
            ActionType.FILL, intent, selector="xpath=//input[@id='username']"
        )
        is True
    )


def test_semantic_consistency_click_accepts_navigation_intent():
    intent = Intent(
        raw="go checkboxes",
        verb="Navigate",
        object="Checkboxes",
        summary="Navigate to checkboxes page",
        key="elements.select.checkboxes",
        confidence=0.9,
    )
    assert (
        _semantic_consistency(
            ActionType.CLICK, intent, selector="xpath=//a[@href='/checkboxes']"
        )
        is True
    )


def test_semantic_consistency_rich_text_accepts_type_intent():
    intent = Intent(
        raw="type content",
        verb="Type",
        object="Rich text editor",
        summary="Type content in the rich text editor",
        key="elements.iframe.type",
        confidence=0.9,
    )
    assert (
        _semantic_consistency(
            ActionType.RICH_TEXT, intent, selector="#tinymce"
        )
        is True
    )


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


@pytest.mark.parametrize(
    "error_text, expected_primary, expected_detail",
    [
        ("tab: missing tab", "tab", "tab.missing"),
        ("iframe: Failed to locate iframe #f1", "iframe", "iframe.not_found"),
        (
            "iframe: Target page, context or browser has been closed",
            "iframe",
            "iframe.not_attached",
        ),
        ("iframe: some other error", "iframe", "iframe.unknown"),
        (
            "async_load: Target page, context or browser has been closed",
            "iframe",
            "iframe.not_attached",
        ),
        ("async_load: Timeout 30000ms exceeded", "async_load", "async_load.timeout"),
        (
            "Login flow did not leave login page",
            "dependency",
            "dependency.nav_failed",
        ),
        (
            "Click was expected to navigate away",
            "dependency",
            "dependency.nav_failed",
        ),
        (
            "iframe for the next step did not appear",
            "dependency",
            "dependency.lookahead",
        ),
        (
            "selector: element is not attached to the DOM",
            "selector",
            "selector.stale",
        ),
        ("selector: waiting for selector '#x'", "selector", "selector.not_found"),
        ("selector: Timeout 30000ms exceeded", "selector", "selector.timeout"),
        ("selector: generic error", "selector", "selector.unknown"),
        (None, "dependency", "dependency.unknown"),
        ("", "dependency", "dependency.unknown"),
    ],
)
def test_classify_root_cause_detail(error_text, expected_primary, expected_detail):
    from graph_agent.acceptance.failure_chain import classify_root_cause_detail

    primary, detail = classify_root_cause_detail(error_text)
    assert primary.value == expected_primary
    assert detail == expected_detail


def test_resolve_mapping_headless_from_env():
    with patch.dict("os.environ", {}, clear=True):
        assert _resolve_mapping_headless() is True
    with patch.dict("os.environ", {"MAPPING_HEADLESS": "true"}, clear=True):
        assert _resolve_mapping_headless() is True
    with patch.dict("os.environ", {"MAPPING_HEADLESS": "1"}, clear=True):
        assert _resolve_mapping_headless() is True
    with patch.dict("os.environ", {"MAPPING_HEADLESS": "false"}, clear=True):
        assert _resolve_mapping_headless() is False
    with patch.dict("os.environ", {"MAPPING_HEADLESS": "0"}, clear=True):
        assert _resolve_mapping_headless() is False


def test_resolve_mapping_channel_from_env():
    with patch.dict("os.environ", {}, clear=True):
        assert _resolve_mapping_channel() is None
    with patch.dict("os.environ", {"MAPPING_CHANNEL": "chrome"}, clear=True):
        assert _resolve_mapping_channel() == "chrome"


def test_resolve_intent_context_window_from_env():
    with patch.dict("os.environ", {}, clear=True):
        assert _resolve_intent_context_window() == 1
    with patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "3"}, clear=True):
        assert _resolve_intent_context_window() == 3
    with patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "-2"}, clear=True):
        assert _resolve_intent_context_window() == 0
    with patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "99"}, clear=True):
        assert _resolve_intent_context_window() == 5
    with patch.dict("os.environ", {"MAPPING_INTENT_CONTEXT_WINDOW": "abc"}, clear=True):
        assert _resolve_intent_context_window() == 1


def test_task_template_includes_derived_exploration_hint():
    """Task 2: Default task template should encourage exploring derived pages."""
    from graph_agent.mapping.run import DEFAULT_TASK_TEMPLATE

    task = DEFAULT_TASK_TEMPLATE.format(start_url="https://example.com/")
    assert "派生" in task
    assert "继续探索" in task
    assert "菜单" in task or "列表" in task or "详情" in task


def test_task_template_mentions_rich_text_editors():
    from graph_agent.mapping.run import DEFAULT_TASK_TEMPLATE

    task = DEFAULT_TASK_TEMPLATE.format(start_url="https://example.com/")
    assert "富文本" in task
    assert "contenteditable" in task
    assert "input_text" in task


@pytest.mark.asyncio
async def test_mapping_save_load_preserves_recording_context(tmp_path: Path):
    """Task 4: mapping -> save -> load preserves tab, frame_path, node URL, selector."""
    history = _make_mock_history(
        actions=[
            {
                "click": {"element": "button"},
                "interacted_element": {"css_selector": "#submit"},
            }
        ],
        thoughts=[{"next_goal": "Submit form"}],
        urls=["https://example.com/form", "https://example.com/done"],
    )
    frame_path = [
        FrameLocatorSnapshot(selector="iframe[name='outer']", name="outer"),
        FrameLocatorSnapshot(selector="iframe[name='inner']", name="inner"),
    ]
    edge_with_full_context = type(
        "EdgeModel",
        (),
        {
            "source": "https://example.com/form",
            "target": "https://example.com/done",
            "selector": "#submit",
            "action": ActionType.CLICK,
            "intent": Intent(
                summary="Submit form",
                raw="submit",
                verb="Submit",
                object="Form",
            ),
            "tab_id": "tab-0",
            "target_tab_id": "tab-1",
            "tab_action": TabActionType.OPEN,
            "tab": TabSnapshot(
                tab_id="tab-1",
                opener_tab_id="tab-0",
                url="https://example.com/done",
                title="Done",
            ),
            "frame_path": frame_path,
            "element": ElementSnapshot(selector="#submit", frame_path=frame_path),
            "context_level_used": "L0",
            "intent_failure_reason": None,
            "param_name": None,
            "action_value": None,
            "constraints": None,
        },
    )()
    with patch(
        "graph_agent.mapping.run.parse_browser_use_step",
        new_callable=AsyncMock,
        return_value=edge_with_full_context,
    ):
        G = await _build_graph_from_history(history, intent_mode="sync")

    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)
    loaded = load_graph(graph_path)

    # Node URL preserved
    for nid, data in loaded.nodes(data=True):
        assert "url" in data
        assert (
            data["url"] in ("https://example.com/form", "https://example.com/done")
            or data["url"]
        )

    # Edge context preserved
    edges = list(loaded.edges(keys=True, data=True))
    assert len(edges) >= 1
    for _u, _v, _k, data in edges:
        assert data["selector"] == "#submit"
        assert data["tab_id"] == "tab-0"
        assert data["target_tab_id"] == "tab-1"
        assert data["tab_action"] == TabActionType.OPEN
        assert data["tab"] is not None
        assert data["tab"].tab_id == "tab-1"
        assert data["tab"].url == "https://example.com/done"
        assert [f.selector for f in data["frame_path"]] == [
            "iframe[name='outer']",
            "iframe[name='inner']",
        ]
        if data.get("element"):
            assert [f.name for f in data["element"].frame_path] == ["outer", "inner"]
        break


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
    G.add_edge(
        "a",
        "b",
        action=ActionType.CLICK,
        selector="#x",
        intent=Intent(raw="x", verb="Click", object="X", summary="Click x"),
    )
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


# --- Task 6: 重新录制真实图谱 ---


def _make_mock_history_for_mapping(
    actions: list[dict],
    thoughts: list[dict],
    urls: list[str],
    final_result: str = "",
) -> object:
    """Create mock history for run_mapping (Agent.run return value)."""

    class MockHistory:
        def model_actions(self):
            return actions

        def model_thoughts(self):
            return thoughts

        def urls(self):
            return urls

        def final_result(self):
            return final_result

    return MockHistory()


@pytest.mark.asyncio
async def test_run_mapping_produces_graph_with_required_metadata(tmp_path: Path):
    """Task 6: run_mapping produces graph with visited_urls, start_url, acceptance_snapshot."""
    inventory_path = tmp_path / "element_inventory.json"
    inventory_path.write_text(
        '{"elements":[{"selector":"#username","type":"input"},{"selector":"#password","type":"input"},'
        '{"selector":"button[type=submit]","type":"button"}],'
        '"metadata":{"page_count":1,"aggregated_element_count":3}}',
        encoding="utf-8",
    )
    output_path = tmp_path / "graph.json"

    mock_history = _make_mock_history_for_mapping(
        actions=[
            {
                "click": {"element": "a"},
                "interacted_element": {"attributes": {"href": "/login"}},
            },
            {
                "input_text": {"text": "user"},
                "interacted_element": {"attributes": {"id": "username"}},
            },
            {
                "input_text": {"text": "pass"},
                "interacted_element": {"attributes": {"id": "password"}},
            },
            {
                "click": {"element": "button"},
                "interacted_element": {"attributes": {"type": "submit"}},
            },
        ],
        thoughts=[
            {"next_goal": "Go to login"},
            {"next_goal": "Fill username"},
            {"next_goal": "Fill password"},
            {"next_goal": "Submit"},
        ],
        urls=[
            "https://the-internet.herokuapp.com/",
            "https://the-internet.herokuapp.com/login",
            "https://the-internet.herokuapp.com/login",
            "https://the-internet.herokuapp.com/login",
            "https://the-internet.herokuapp.com/secure",
        ],
    )

    class MockBrowser:
        async def stop(self):
            pass

        async def close(self):
            pass

    class MockAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, max_steps: int = 30):
            return mock_history

    def mock_browser(*args, **kwargs):
        return MockBrowser()

    def _edge(sel, key, action_type):
        return type(
            "E",
            (),
            {
                "selector": sel,
                "action": action_type,
                "intent": Intent(
                    summary=key, raw=key, verb="Click", object=key, key=key
                ),
                "intent_failure_reason": None,
                "param_name": "username"
                if "username" in key
                else ("password" if "password" in key else None),
                "action_value": None,
                "element": None,
                "constraints": None,
                "tab_id": "tab-0",
                "target_tab_id": None,
                "tab_action": None,
                "tab": None,
                "frame_path": [],
                "context_level_used": None,
            },
        )()

    edge_models = [
        _edge('a[href="/login"]', "go_to_login", ActionType.CLICK),
        _edge("#username", "fill_username", ActionType.FILL),
        _edge("#password", "fill_password", ActionType.FILL),
        _edge("button[type=submit]", "submit_login", ActionType.CLICK),
    ]
    with (
        patch.dict("os.environ", {"MAPPING_INTENT_MODE": "sync"}, clear=False),
        patch("browser_use.Agent", MockAgent),
        patch("browser_use.Browser", mock_browser),
        patch(
            "graph_agent.mapping.run.parse_browser_use_step",
            new_callable=AsyncMock,
            side_effect=edge_models,
        ),
        patch(
            "graph_agent.mapping.run.generate_business_templates",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        from graph_agent.mapping.run import run_mapping

        G = await run_mapping(
            url="https://the-internet.herokuapp.com/",
            output_path=str(output_path),
            inventory_path=inventory_path,
        )

    assert output_path.exists()
    assert G.graph["start_url"] == "https://the-internet.herokuapp.com/"
    assert G.graph.get("data_source") == "mapping.run"
    assert "generated_at" in G.graph
    assert "intent_alignment_warning_count" in G.graph
    assert "intent_alignment_warning_breakdown" in G.graph
    assert "visited_urls" in G.graph
    assert "https://the-internet.herokuapp.com/" in G.graph["visited_urls"]
    assert "https://the-internet.herokuapp.com/login" in G.graph["visited_urls"]
    assert "https://the-internet.herokuapp.com/secure" in G.graph["visited_urls"]

    snapshot_path = tmp_path / "acceptance_snapshot.json"
    assert snapshot_path.exists()

    loaded = load_graph(output_path)
    assert loaded.number_of_nodes() >= 2
    assert loaded.graph["start_url"] == "https://the-internet.herokuapp.com/"
    assert loaded.graph.get("intent_alignment_warning_count", 0) >= 0


def test_mapping_output_graph_has_structure_for_playback(tmp_path: Path):
    """Task 6: Graph from mapping has structure required for playback (selector, intent, url)."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("state-0", label="home", url="https://the-internet.herokuapp.com/")
    G.add_node("state-1", label="login", url="https://the-internet.herokuapp.com/login")
    G.add_node(
        "state-2", label="secure", url="https://the-internet.herokuapp.com/secure"
    )
    G.add_edge(
        "state-0",
        "state-1",
        key="step-0",
        selector='a[href="/login"]',
        action=ActionType.CLICK,
        intent=Intent(
            summary="Go to login",
            raw="go login",
            verb="Click",
            object="Login link",
            key="go_to_login",
        ),
    )
    G.add_edge(
        "state-1",
        "state-2",
        key="step-1",
        selector="button[type=submit]",
        action=ActionType.CLICK,
        intent=Intent(
            summary="Submit login",
            raw="submit",
            verb="Submit",
            object="Login form",
            key="submit_login",
        ),
    )
    G.graph["start_url"] = "https://the-internet.herokuapp.com/"
    G.graph["visited_urls"] = [
        "https://the-internet.herokuapp.com/",
        "https://the-internet.herokuapp.com/login",
        "https://the-internet.herokuapp.com/secure",
    ]

    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)
    loaded = load_graph(graph_path)

    assert loaded.number_of_edges() >= 2
    for _u, _v, data in loaded.edges(data=True):
        assert "selector" in data
        assert data["selector"]
        assert "action" in data
    node_urls = {n.get("url") for _, n in loaded.nodes(data=True) if n.get("url")}
    assert "https://the-internet.herokuapp.com/" in node_urls
    assert "https://the-internet.herokuapp.com/secure" in node_urls

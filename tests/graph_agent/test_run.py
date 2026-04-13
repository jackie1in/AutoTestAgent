"""Tests for Mapping run module (T4:构图与统计联动, T5:re-infer-missing)."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import networkx as nx

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.cartography.runner import (
    _build_mapping_task_with_env_hints,
    _resolve_mapping_channel,
    _resolve_mapping_headless,
    _resolve_intent_context_window,
    _resolve_target_state,
    _runtime_filter_snapshots,
    _semantic_consistency,
)
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    Intent,
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

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

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

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

@pytest.mark.asyncio

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
    from graph_agent.cartography.runner import DEFAULT_TASK_TEMPLATE

    task = DEFAULT_TASK_TEMPLATE.format(start_url="https://example.com/")
    assert "派生" in task
    assert "继续探索" in task
    assert "菜单" in task or "列表" in task or "详情" in task

def test_task_template_mentions_rich_text_editors():
    from graph_agent.cartography.runner import DEFAULT_TASK_TEMPLATE

    task = DEFAULT_TASK_TEMPLATE.format(start_url="https://example.com/")
    assert "富文本" in task
    assert "contenteditable" in task
    assert "input_text" in task

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
            "graph_agent.cartography.runner.parse_browser_use_step",
            new_callable=AsyncMock,
            side_effect=edge_models,
        ),
        patch(
            "graph_agent.cartography.runner.generate_business_templates",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        from graph_agent.cartography.runner import run_mapping

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

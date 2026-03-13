"""Task 7: 用新图谱做真实回放验收测试.

PRD 8.3 / 9.7:
- 至少 3 个真实业务意图回放成功
- 至少 1 个成功意图包含多标签页或 iframe 场景（若图谱中存在）
- 测试数据来自最新 mapping.run 输出 (graph_agent/data/graph.json)
"""

import socket
from unittest.mock import AsyncMock, patch

import pytest
import networkx as nx

from graph_agent.acceptance.playback_acceptance import (
    PlaybackAcceptanceResult,
    _prioritize_iframe_intents,
    is_graph_from_mapping_run,
    run_playback_acceptance,
)
from graph_agent.graph.io import save_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.models import ActionType, FrameLocatorSnapshot, Intent


def _target_site_reachable() -> bool:
    """检查目标站点是否可达（用于跳过无网络环境）。"""
    try:
        sock = socket.create_connection(("the-internet.herokuapp.com", 443), timeout=5)
        sock.close()
        return True
    except (socket.timeout, OSError):
        return False


requires_network = pytest.mark.skipif(
    not _target_site_reachable(),
    reason="Target site the-internet.herokuapp.com unreachable (no network?)",
)

DEFAULT_GRAPH = "graph_agent/data/graph.json"
TARGET_SITE = "https://the-internet.herokuapp.com"
TARGET_HOME = f"{TARGET_SITE}/"
TARGET_LOGIN = f"{TARGET_SITE}/login"
TARGET_SECURE = f"{TARGET_SITE}/secure"
TARGET_FRAMES = f"{TARGET_SITE}/frames"
TARGET_IFRAME = f"{TARGET_SITE}/iframe"


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


def _build_three_intent_fixture_graph() -> nx.MultiDiGraph:
    """构建含 3 个可回放意图的 fixture 图（模拟 mapping.run 产出）。"""
    G = nx.MultiDiGraph()
    G.add_node("home", url=TARGET_HOME)
    G.add_node("login-empty", url=TARGET_LOGIN)
    G.add_node("login-user", url=TARGET_LOGIN)
    G.add_node("login-password", url=TARGET_LOGIN)
    G.add_node("secure", url=TARGET_SECURE)
    G.graph["start_url"] = TARGET_HOME

    def add_edge(u, v, key, **kw):
        G.add_edge(u, v, key=key, edge_id=key, **kw)

    add_edge(
        "home",
        "login-empty",
        "e1",
        selector='a[href="/login"]',
        action=ActionType.CLICK,
        intent=_make_intent("Go to login", key="go_to_login"),
    )
    add_edge(
        "login-empty",
        "login-user",
        "e2",
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="auth.fill.username"),
        param_name="username",
    )
    add_edge(
        "login-user",
        "login-password",
        "e3",
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="auth.fill.password"),
        param_name="password",
    )
    add_edge(
        "login-password",
        "secure",
        "e4",
        selector='button[type="submit"]',
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="auth.submit.login"),
    )
    return G


def _build_fixture_with_iframe_graph() -> nx.MultiDiGraph:
    """构建含 iframe 场景的 fixture 图（PRD 8.3: 至少 1 个成功意图包含 iframe）。"""
    G = _build_three_intent_fixture_graph()
    G.add_node("frames", url=TARGET_FRAMES)
    G.add_node("iframe-page", url=TARGET_IFRAME)

    def add_edge(u, v, key, **kw):
        G.add_edge(u, v, key=key, edge_id=key, **kw)

    add_edge(
        "home",
        "frames",
        "e-frames",
        selector='a[href="/frames"]',
        action=ActionType.CLICK,
        intent=_make_intent("Go to frames", key="frames.navigate"),
    )
    add_edge(
        "frames",
        "iframe-page",
        "e-iframe",
        selector='a[href="/iframe"]',
        action=ActionType.CLICK,
        intent=_make_intent("Go to iframe page", key="iframe.navigate"),
    )
    add_edge(
        "iframe-page",
        "iframe-page",
        "e-iframe-type",
        selector="#tinymce",
        action=ActionType.FILL,
        intent=_make_intent("Type in iframe editor", key="elements.iframe.type"),
        frame_path=[FrameLocatorSnapshot(selector="#mce_0_ifr")],
        param_name="iframe_content",
    )
    return G


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_playback_acceptance_at_least_three_intents(tmp_path):
    """
    Task 7: 至少 3 个真实业务意图回放成功。
    使用含 3 个可解析意图的 fixture 图验证验收逻辑（模拟 mapping.run 产出）。
    """
    G = _build_three_intent_fixture_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    result = await run_playback_acceptance(
        str(graph_path),
        start_url=TARGET_HOME,
        intent_queries=["auth.login", "auth.fill.username", "auth.fill.password"],
        min_success=3,
        min_with_iframe_or_tab=0,
    )
    assert result.total_attempted >= 3, "应尝试至少 3 个意图"
    assert result.total_succeeded >= 3, (
        f"PRD 8.3: 至少 3 个成功回放。"
        f"实际: succeeded={result.total_succeeded}, "
        f"results={[(r.intent_query, r.success, r.error) for r in result.results]}"
    )


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_playback_acceptance_real_graph_at_least_three_intents():
    """
    Task 7: 基于 mapping.run 产出的 graph.json 做真实回放。
    PRD 8.3: 至少 3 个真实业务意图回放成功。
    测试数据来自最新 mapping.run 输出 (graph_agent/data/graph.json)。
    """
    ok, reason = is_graph_from_mapping_run(DEFAULT_GRAPH)
    if not ok:
        pytest.skip(reason)
    result = await run_playback_acceptance(
        DEFAULT_GRAPH,
        start_url=TARGET_HOME,
        min_success=3,
        min_with_iframe_or_tab=0,
    )
    assert result.total_attempted >= 3, "应尝试至少 3 个意图"
    assert result.total_succeeded >= 3, (
        f"PRD 8.3: 至少 3 个真实业务意图回放成功。"
        f"实际: succeeded={result.total_succeeded}, "
        f"results={[(r.intent_query, r.success, r.error) for r in result.results]}"
    )


def test_playback_acceptance_three_intents_resolvable():
    """Fixture 图中 fill_username、fill_password、auth.login 均可解析为路径。"""
    G = _build_three_intent_fixture_graph()
    for q in ["auth.login", "auth.fill.username", "auth.fill.password"]:
        path = get_path_from_query(q, G)
        assert len(path) >= 2, f"{q} 应解析为至少 2 条边"


def test_playback_acceptance_iframe_intent_resolvable():
    """Fixture 图中 elements.iframe.type 可解析为含 frame_path 的路径。"""
    G = _build_fixture_with_iframe_graph()
    path = get_path_from_query("elements.iframe.type", G)
    assert len(path) >= 2, "elements.iframe.type 应解析为至少 2 条边"
    has_frame = any(
        getattr(e, "frame_path", None) and len(e.frame_path) > 0 for e in path
    )
    assert has_frame, "路径应包含 frame_path（iframe 场景）"


def test_prioritize_iframe_intents_puts_iframe_first():
    """意图 A: 包含 iframe 的业务动作应优先于其他意图。"""
    G = _build_fixture_with_iframe_graph()
    queries = [
        "auth.login",
        "auth.fill.username",
        "elements.iframe.type",
    ]
    prioritized = _prioritize_iframe_intents(queries, G)
    queries_ordered = [q for q, _ in prioritized]
    iframe_idx = queries_ordered.index("elements.iframe.type")
    auth_login_idx = queries_ordered.index("auth.login")
    auth_username_idx = queries_ordered.index("auth.fill.username")
    assert iframe_idx < auth_login_idx, (
        f"意图 A: elements.iframe.type 应排在 auth.login 前面, "
        f"实际顺序: {queries_ordered}"
    )
    assert iframe_idx < auth_username_idx, (
        f"意图 A: elements.iframe.type 应排在 auth.fill.username 前面, "
        f"实际顺序: {queries_ordered}"
    )


@pytest.mark.asyncio
async def test_playback_acceptance_iframe_intent_attempted_first(tmp_path):
    """
    意图 A: 优先选择包含 iframe 的业务动作。
    验证 run_playback_acceptance 会先尝试 iframe 意图。
    """
    G = _build_fixture_with_iframe_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    async def _mock_run_playback(path, **kwargs):
        has_iframe = any(
            getattr(e, "frame_path", None) and len(e.frame_path) > 0 for e in path
        )
        return {"success": True, "actual_url": TARGET_IFRAME if has_iframe else TARGET_SECURE}

    with patch(
        "graph_agent.acceptance.playback_acceptance.run_playback",
        new_callable=AsyncMock,
        side_effect=_mock_run_playback,
    ):
        result = await run_playback_acceptance(
            str(graph_path),
            start_url=TARGET_HOME,
            test_data={"username": "tomsmith", "password": "x", "iframe_content": "t"},
            intent_queries=[
                "auth.login",
                "auth.fill.username",
                "elements.iframe.type",
            ],
            min_success=3,
            min_with_iframe_or_tab=1,
        )

    # 意图 A: iframe 意图应排在前面尝试，故 results 中首个含 iframe 的应在首个不含 iframe 的之前
    iframe_indices = [i for i, r in enumerate(result.results) if r.has_iframe_or_tab]
    non_iframe_indices = [i for i, r in enumerate(result.results) if not r.has_iframe_or_tab]
    if iframe_indices and non_iframe_indices:
        assert min(iframe_indices) < min(non_iframe_indices), (
            f"意图 A: iframe 意图应优先尝试。"
            f"results 顺序: {[(r.intent_query, r.has_iframe_or_tab) for r in result.results]}"
        )


@pytest.mark.asyncio
async def test_playback_acceptance_with_iframe_mock_success(tmp_path):
    """
    PRD 8.3: 验收逻辑在至少 1 个成功意图含 iframe 时通过。
    使用 mock 的 run_playback 验证，不依赖真实网络。
    """
    G = _build_fixture_with_iframe_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    async def _mock_run_playback(path, **kwargs):
        has_iframe = any(
            getattr(e, "frame_path", None) and len(e.frame_path) > 0 for e in path
        )
        return {
            "success": True,
            "actual_url": TARGET_IFRAME if has_iframe else TARGET_SECURE,
        }

    with patch(
        "graph_agent.acceptance.playback_acceptance.run_playback",
        new_callable=AsyncMock,
        side_effect=_mock_run_playback,
    ):
        result = await run_playback_acceptance(
            str(graph_path),
            start_url=TARGET_HOME,
            test_data={
                "username": "tomsmith",
                "password": "SuperSecretPassword!",
                "iframe_content": "test",
            },
            intent_queries=[
                "auth.login",
                "auth.fill.username",
                "elements.iframe.type",
            ],
            min_success=3,
            min_with_iframe_or_tab=1,
        )
    assert result.total_succeeded >= 3
    assert result.with_iframe_or_tab_succeeded >= 1
    assert result.meets_minimum


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_playback_acceptance_at_least_one_intent_with_iframe(tmp_path):
    """
    PRD 8.3: 至少 1 个成功意图包含 iframe 场景。
    使用含 iframe 的 fixture 图验证验收逻辑。
    """
    G = _build_fixture_with_iframe_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    test_data = {
        "username": "tomsmith",
        "password": "SuperSecretPassword!",
        "iframe_content": "test",
    }
    result = await run_playback_acceptance(
        str(graph_path),
        start_url=TARGET_HOME,
        test_data=test_data,
        intent_queries=[
            "auth.login",
            "auth.fill.username",
            "elements.iframe.type",
        ],
        min_success=3,
        min_with_iframe_or_tab=1,
    )
    assert result.total_attempted >= 3, "应尝试至少 3 个意图"
    assert result.total_succeeded >= 3, (
        f"PRD 8.3: 至少 3 个成功回放。"
        f"实际: succeeded={result.total_succeeded}, "
        f"results={[(r.intent_query, r.success, r.error) for r in result.results]}"
    )
    assert result.with_iframe_or_tab_succeeded >= 1, (
        f"PRD 8.3: 至少 1 个成功意图包含 iframe 场景。"
        f"实际: with_iframe_or_tab_succeeded={result.with_iframe_or_tab_succeeded}, "
        f"results={[(r.intent_query, r.success, r.has_iframe_or_tab) for r in result.results]}"
    )


def test_playback_acceptance_result_meets_minimum_logic():
    """PlaybackAcceptanceResult.meets_minimum 逻辑正确。"""
    r = PlaybackAcceptanceResult(
        graph_path="x",
        total_attempted=3,
        total_succeeded=3,
        with_iframe_or_tab_succeeded=0,
        results=[],
    )
    assert r.meets_minimum is True


def test_is_graph_from_mapping_run_missing_file():
    """is_graph_from_mapping_run 对不存在文件返回 False。"""
    ok, reason = is_graph_from_mapping_run("/nonexistent/graph.json")
    assert ok is False
    assert "not found" in reason or "graph" in reason.lower()


def test_is_graph_from_mapping_run_fixture_without_metadata(tmp_path):
    """is_graph_from_mapping_run 对无 data_source/generated_at 的图返回 False。"""
    G = _build_three_intent_fixture_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)
    ok, reason = is_graph_from_mapping_run(graph_path)
    assert ok is False
    assert "mapping.run" in reason


def test_is_graph_from_mapping_run_fixture_with_metadata(tmp_path):
    """is_graph_from_mapping_run 对含 data_source 与 generated_at 的图返回 True。"""
    G = _build_three_intent_fixture_graph()
    G.graph["data_source"] = "mapping.run"
    G.graph["generated_at"] = "2026-03-13T00:00:00+00:00"
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)
    ok, reason = is_graph_from_mapping_run(graph_path)
    assert ok is True
    assert reason == ""

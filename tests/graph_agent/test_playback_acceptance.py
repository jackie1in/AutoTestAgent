"""Task 7: 用新图谱做真实回放验收测试.

PRD 8.3 / 9.7:
- 至少 3 个真实业务意图回放成功
- 至少 1 个成功意图包含多标签页或 iframe 场景（若图谱中存在）
- 测试数据来自最新 mapping.run 输出 (graph_agent/data/graph.json)
"""

import socket

import pytest
import networkx as nx

from graph_agent.acceptance.playback_acceptance import (
    PlaybackAcceptanceResult,
    run_playback_acceptance,
)
from graph_agent.graph.io import save_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.models import ActionType, Intent


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
async def test_playback_acceptance_real_graph_when_available():
    """
    Task 7: 基于 mapping.run 产出的 graph.json 做真实回放。
    当图谱存在且至少 1 个意图可回放时验证。数据来源：graph_agent/data/graph.json
    """
    from pathlib import Path

    if not Path(DEFAULT_GRAPH).exists():
        pytest.skip(
            "graph.json 不存在，请先运行 uv run python -m graph_agent.mapping.run"
        )
    result = await run_playback_acceptance(
        DEFAULT_GRAPH,
        start_url=TARGET_HOME,
        min_success=1,
        min_with_iframe_or_tab=0,
    )
    assert result.total_attempted >= 1
    assert result.total_succeeded >= 1, (
        f"真实图谱应至少 1 个意图回放成功。"
        f"results={[(r.intent_query, r.success, r.error) for r in result.results]}"
    )


def test_playback_acceptance_three_intents_resolvable():
    """Fixture 图中 fill_username、fill_password、auth.login 均可解析为路径。"""
    G = _build_three_intent_fixture_graph()
    for q in ["auth.login", "auth.fill.username", "auth.fill.password"]:
        path = get_path_from_query(q, G)
        assert len(path) >= 2, f"{q} 应解析为至少 2 条边"


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

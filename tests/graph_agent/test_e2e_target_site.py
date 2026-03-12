"""T10 端到端验收（目标站点）: 验证 测绘 -> 意图 -> 寻径 -> 回放 链路可执行。

目标站点: https://the-internet.herokuapp.com
PRD 6.2: pathfinding -> playback 端到端测试至少 1 条通过。
"""

import os
import socket

import pytest
import networkx as nx

from graph_agent.graph.io import save_graph
from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import ActionType, Intent
from graph_agent.playback.engine import run_playback
from graph_agent.run_e2e_acceptance import _compute_stats, _find_path_length_ge_2, run_acceptance


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


def _build_target_site_fixture_graph() -> nx.DiGraph:
    """构建目标站点 (the-internet.herokuapp.com) 的 fixture 图。

    流程: 首页 -> 点击 Form Authentication -> 登录页 -> 填写用户名 -> 填写密码 -> 提交 -> 安全页
    使用 pseudo-state 区分同 URL 的中间状态，避免 pathfinding 自环问题。
    """
    G: nx.DiGraph = nx.DiGraph()
    # 节点
    G.add_node("home", url=TARGET_HOME, label="Home")
    G.add_node("login-empty", url=TARGET_LOGIN, label="Login")
    G.add_node("login-user", url=TARGET_LOGIN, label="Login filled user")
    G.add_node("login-password", url=TARGET_LOGIN, label="Login filled both")
    G.add_node("secure", url=TARGET_SECURE, label="Secure")

    # 边
    G.add_edge(
        "home",
        "login-empty",
        selector='a[href="/login"]',
        action=ActionType.CLICK,
        intent=_make_intent("Go to login page", key="go_to_login"),
        intent_failure_reason=None,
    )
    G.add_edge(
        "login-empty",
        "login-user",
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="fill_username"),
        intent_failure_reason=None,
        param_name="username",
    )
    G.add_edge(
        "login-user",
        "login-password",
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="fill_password"),
        intent_failure_reason=None,
        param_name="password",
    )
    G.add_edge(
        "login-password",
        "secure",
        selector='button[type="submit"]',
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="submit_login"),
        intent_failure_reason=None,
    )
    return G


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_pathfinding_to_playback_on_target_site():
    """
    T10 端到端验收: 在目标站点上执行 pathfinding -> playback。
    PRD 6.2: 至少 1 条长度 >= 2 的可执行业务路径完成回放。
    """
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        G = _build_target_site_fixture_graph()

        # Pathfinding: 获取 fill_username 路径 (长度 >= 2)
        path = get_path_from_intent("fill_username", G)
        assert len(path) >= 2, "需要至少 2 条边的路径"
        assert path[0].selector == 'a[href="/login"]'
        assert path[1].selector == "#username"

        # Playback: 在真实目标站点上执行
        result = await run_playback(
            path,
            test_data={"username": "tomsmith", "password": "SuperSecretPassword!"},
            start_url=TARGET_HOME,
        )

        assert result["success"] is True, f"回放失败: {result.get('error')}"
        assert result["error"] is None
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


def test_acceptance_compute_stats():
    """验收脚本: _compute_stats 正确计算指标。"""
    G = _build_target_site_fixture_graph()
    stats = _compute_stats(G)
    assert stats["edge_count"] == 4
    assert stats["intent_missing_count"] == 0
    assert stats["intent_success_rate"] == 1.0


def test_acceptance_find_path_length_ge_2():
    """验收脚本: _find_path_length_ge_2 找到长度 >= 2 的路径。"""
    G = _build_target_site_fixture_graph()
    path, intent = _find_path_length_ge_2(G)
    assert len(path) >= 2
    assert intent in ("fill_username", "fill_password", "submit_login", "login", "click", "fill")


def test_acceptance_start_url_uses_node_metadata_for_opaque_state_id():
    """Acceptance start URL resolution should use node metadata, not state ids."""
    G = _build_target_site_fixture_graph()
    from graph_agent.run_e2e_acceptance import _get_start_url_from_graph

    assert _get_start_url_from_graph(G) == TARGET_HOME


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_run_acceptance_with_fixture_graph(tmp_path):
    """验收脚本: 使用 fixture 图执行完整验收流程。"""
    G = _build_target_site_fixture_graph()
    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)
    ok = await run_acceptance(graph_path, re_infer=False, start_url=TARGET_HOME)
    assert ok is True


@pytest.mark.integration
@requires_network
@pytest.mark.asyncio
async def test_full_login_path_on_target_site():
    """
    T10: 完整登录流程 pathfinding -> playback (路径长度 >= 2)。
    """
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        G = _build_target_site_fixture_graph()

        # Pathfinding: 获取 submit_login 完整路径
        path = get_path_from_intent("submit_login", G)
        assert len(path) >= 2, "需要至少 2 条边的路径"

        # Playback: 完整登录流程
        result = await run_playback(
            path,
            test_data={"username": "tomsmith", "password": "SuperSecretPassword!"},
            start_url=TARGET_HOME,
            expected_end_url=TARGET_SECURE,
        )

        assert result["success"] is True, f"回放失败: {result.get('error')}"
        assert result["error"] is None
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)

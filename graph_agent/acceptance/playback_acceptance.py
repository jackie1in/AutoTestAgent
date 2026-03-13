"""Task 7: 用新图谱做真实回放验收.

PRD 8.3 / 9.7: 基于 mapping.run 产出的新图谱完成真实业务回放。
- 至少 3 个真实业务意图成功回放
- 至少 1 个成功意图包含多标签页或 iframe 场景（若图谱中存在）
- 测试数据来自最新 mapping.run 输出 (graph_agent/data/graph.json)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph_agent.graph.io import load_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.playback.engine import run_playback


@dataclass
class PlaybackResult:
    """单次回放结果。"""

    intent_query: str
    success: bool
    path_length: int
    actual_url: str = ""
    error: str | None = None
    has_iframe_or_tab: bool = False


@dataclass
class PlaybackAcceptanceResult:
    """Task 7 回放验收汇总。"""

    graph_path: str
    data_source: str = "uv run python -m graph_agent.mapping.run"
    total_attempted: int = 0
    total_succeeded: int = 0
    with_iframe_or_tab_succeeded: int = 0
    results: list[PlaybackResult] = field(default_factory=list)

    @property
    def meets_minimum(self) -> bool:
        """PRD 8.3: 至少 3 个成功；若图谱含 iframe/tab 边则至少 1 个成功含之。"""
        if self.total_succeeded < 3:
            return False
        if self.with_iframe_or_tab_succeeded >= 1:
            return True
        if not any(r.has_iframe_or_tab for r in self.results):
            return True
        return False


# 优先回放的意图（PRD 9.7 建议）- 按成功概率排序
# 真实图谱 (the-internet.herokuapp.com) 中可解析的意图
PRIORITY_INTENTS = [
    "auth.login",
    "登录",
    "auth.fill.username",  # 原子意图，可单独回放
    "auth.fill.password",
    "elements.iframe.type",  # 意图 A: 包含 iframe 的业务动作
    "navigation.module.select",  # 意图 B: 登录后进入目标模块（依赖 auth.login）
    "登录后进入目标模块",  # 意图 B 中文别名
    "elements.navigation.select",  # 进入 Add/Remove 模块（依赖 auth.login）
    "elements.management.add",  # 意图 C: 项目 -> 子项目
    "elements.add",
]

# the-internet 标准测试账号
THE_INTERNET_CREDENTIALS = {
    "username": "tomsmith",
    "password": "SuperSecretPassword!",
}


def is_graph_from_mapping_run(graph_path: str | Path) -> tuple[bool, str]:
    """PRD 8.3 / 9.7: 验证图谱是否来自 mapping.run 输出。

    Returns:
        (True, "") 若图谱 metadata 含 data_source=mapping.run 且 generated_at 存在；
        (False, reason) 否则。
    """
    path_obj = Path(graph_path)
    if not path_obj.exists():
        return False, f"graph not found: {graph_path}"
    graph = load_graph(path_obj)
    ds = graph.graph.get("data_source")
    ga = graph.graph.get("generated_at")
    if ds != "mapping.run":
        return False, (
            f"graph metadata data_source={ds!r}, expected 'mapping.run'. "
            "请先运行: uv run python -m graph_agent.mapping.run"
        )
    if not ga:
        return False, (
            "graph metadata missing generated_at. "
            "请先运行: uv run python -m graph_agent.mapping.run"
        )
    return True, ""


def _has_iframe_or_tab(edges: list[Any]) -> bool:
    """检查路径是否包含 iframe 或 tab 动作。"""
    for e in edges:
        if getattr(e, "frame_path", None) and len(e.frame_path) > 0:
            return True
        if getattr(e, "tab_action", None) is not None:
            return True
    return False


def _prioritize_iframe_intents(
    queries: list[str], graph: Any
) -> list[tuple[str, list[Any]]]:
    """意图 A: 优先选择包含 iframe 的业务动作。

    解析每个 query 的路径，将含 iframe/tab 的意图排在前面，
    以便验收时优先尝试 iframe 场景。
    """
    resolved: list[tuple[str, list[Any], bool]] = []
    for q in queries:
        path = get_path_from_query(q, graph)
        if len(path) < 2:
            continue
        has_iframe_tab = _has_iframe_or_tab(path)
        resolved.append((q, path, has_iframe_tab))
    # 含 iframe/tab 的排前面（意图 A 优先）
    resolved.sort(key=lambda x: (not x[2], x[0]))
    return [(q, path) for q, path, _ in resolved]


async def run_playback_acceptance(
    graph_path: str | Path,
    start_url: str | None = None,
    test_data: dict[str, Any] | None = None,
    intent_queries: list[str] | None = None,
    min_success: int = 3,
    min_with_iframe_or_tab: int = 1,
) -> PlaybackAcceptanceResult:
    """执行 Task 7 真实回放验收。

    Args:
        graph_path: 图谱 JSON 路径（默认来自 mapping.run 输出）
        start_url: 回放起始 URL，默认从图 metadata 读取
        test_data: 回放用 test_data（如 username/password）
        intent_queries: 要尝试的意图查询列表，默认使用 PRIORITY_INTENTS
        min_success: 最少成功回放数
        min_with_iframe_or_tab: 最少含 iframe/tab 的成功数

    Returns:
        PlaybackAcceptanceResult 汇总
    """
    path_obj = Path(graph_path)
    if not path_obj.exists():
        return PlaybackAcceptanceResult(
            graph_path=str(graph_path),
            results=[],
        )

    graph = load_graph(path_obj)
    if graph.number_of_edges() == 0:
        return PlaybackAcceptanceResult(
            graph_path=str(graph_path),
            results=[],
        )

    url = start_url
    if not url:
        url = graph.graph.get("start_url") if hasattr(graph, "graph") else None
    if not url or not str(url).startswith("http"):
        entries = [n for n in graph if graph.in_degree(n) == 0]
        for n in entries:
            u = graph.nodes[n].get("url") if isinstance(n, str) else None
            if isinstance(u, str) and u.startswith("http"):
                url = u
                break
    if not url:
        url = "https://the-internet.herokuapp.com/"

    data = test_data or THE_INTERNET_CREDENTIALS
    queries = intent_queries or PRIORITY_INTENTS

    # 意图 A: 优先选择包含 iframe 的业务动作
    prioritized = _prioritize_iframe_intents(queries, graph)

    results: list[PlaybackResult] = []
    succeeded = 0
    with_iframe_tab_succeeded = 0

    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        for q, path in prioritized:
            has_iframe_tab = _has_iframe_or_tab(path)
            result = await run_playback(
                path,
                test_data=data,
                start_url=str(url),
            )
            pr = PlaybackResult(
                intent_query=q,
                success=result["success"],
                path_length=len(path),
                actual_url=result.get("actual_url", ""),
                error=result.get("error"),
                has_iframe_or_tab=has_iframe_tab,
            )
            results.append(pr)
            if result["success"]:
                succeeded += 1
                if has_iframe_tab:
                    with_iframe_tab_succeeded += 1
            if (
                succeeded >= min_success
                and with_iframe_tab_succeeded >= min_with_iframe_or_tab
            ):
                break
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)

    return PlaybackAcceptanceResult(
        graph_path=str(graph_path),
        total_attempted=len(results),
        total_succeeded=succeeded,
        with_iframe_or_tab_succeeded=with_iframe_tab_succeeded,
        results=results,
    )

"""Task 7: 用新图谱做真实回放验收.

PRD 8.3 / 9.7: 基于 mapping.run 产出的新图谱完成真实业务回放。
- 至少 3 个真实业务意图成功回放
- 至少 1 个成功意图包含多标签页或 iframe 场景（若图谱中存在）
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_agent.acceptance.failure_chain import classify_root_cause_detail
from graph_agent.graph.pathfinding import (
    get_path_from_query,
    neo4j_transition_to_edge_data,
    _edge_to_model,
)
from graph_agent.models import GraphEdge
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
    root_cause: str | None = None
    root_cause_detail: str | None = None
    failed_step_index: int | None = None
    failed_edge_id: str | None = None


@dataclass
class PlaybackAcceptanceResult:
    """Task 7 回放验收汇总。"""

    graph_path: str
    data_source: str = "uv run python -m graph_agent.cartography.runner"
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
PRIORITY_INTENTS = [
    "auth.login",
    "登录",
    "auth.fill.username",
    "auth.fill.password",
    "elements.iframe.type",
    "navigation.module.select",
    "登录后进入目标模块",
    "elements.navigation.select",
    "项目列表进入子项目并打开概览",
    "elements.management.add",
    "elements.add",
]

# the-internet 标准测试账号
THE_INTERNET_CREDENTIALS = {
    "username": "tomsmith",
    "password": "SuperSecretPassword!",
}


async def _load_edges_from_neo4j() -> list[GraphEdge]:
    """Query transitions from Neo4j and build GraphEdge list."""
    from graph_agent.neo4j_client.driver import Neo4jDriver

    driver = Neo4jDriver()
    await driver.connect()
    try:
        async with driver.driver.session() as session:
            result = await session.run(
                """
                MATCH (a:App)
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN t.id AS id, t.step_index AS step_index,
                       s.id AS from_state_id, target.id AS to_state_id,
                       t.source_url AS source_url, t.target_url AS target_url,
                       t.selector AS selector, t.action AS action,
                       t.tab_id AS tab_id, t.target_tab_id AS target_tab_id,
                       t.tab_action AS tab_action,
                       t.intent_failure_reason AS intent_failure_reason,
                       t.param_name AS param_name, t.action_value AS action_value,
                       t.thought AS thought, t.element_snapshot AS element_snapshot,
                       t.frame_path AS frame_path,
                       i{.*} AS intent
                ORDER BY t.step_index
                """
            )
            edges: list[GraphEdge] = []
            async for record in result:
                t = dict(record)
                data = neo4j_transition_to_edge_data(t)
                u = str(t.get("from_state_id", ""))
                v = str(t.get("to_state_id", ""))
                if u and v:
                    edges.append(_edge_to_model(u, v, data))
            return edges
    finally:
        await driver.close()


def _has_iframe_or_tab(edges: list[Any]) -> bool:
    """检查路径是否包含 iframe 或 tab 动作。"""
    for e in edges:
        if getattr(e, "frame_path", None) and len(e.frame_path) > 0:
            return True
        if getattr(e, "tab_action", None) is not None:
            return True
    return False


def _prioritize_iframe_intents(
    queries: list[str], edges: list[GraphEdge]
) -> list[tuple[str, list[Any]]]:
    """意图 A: 优先选择包含 iframe 的业务动作。

    解析每个 query 的路径，将含 iframe/tab 的意图排在前面，
    以便验收时优先尝试 iframe 场景。
    """
    resolved: list[tuple[str, list[Any], bool]] = []
    for q in queries:
        path = get_path_from_query(q, edges)
        if len(path) < 2:
            continue
        has_iframe_tab = _has_iframe_or_tab(path)
        resolved.append((q, path, has_iframe_tab))
    # 含 iframe/tab 的排前面（意图 A 优先）
    resolved.sort(key=lambda x: (not x[2], x[0]))
    return [(q, path) for q, path, _ in resolved]


async def run_playback_acceptance(
    graph_path: str | Path = "",
    start_url: str | None = None,
    test_data: dict[str, Any] | None = None,
    intent_queries: list[str] | None = None,
    min_success: int = 3,
    min_with_iframe_or_tab: int = 1,
) -> PlaybackAcceptanceResult:
    """执行 Task 7 真实回放验收。

    Args:
        graph_path: 保留参数以兼容旧 API，实际从 Neo4j 读取
        start_url: 回放起始 URL，默认从 edges 中解析
        test_data: 回放用 test_data（如 username/password）
        intent_queries: 要尝试的意图查询列表，默认使用 PRIORITY_INTENTS
        min_success: 最少成功回放数
        min_with_iframe_or_tab: 最少含 iframe/tab 的成功数

    Returns:
        PlaybackAcceptanceResult 汇总
    """
    edges = await _load_edges_from_neo4j()
    if not edges:
        return PlaybackAcceptanceResult(
            graph_path=str(graph_path),
            results=[],
        )

    url = start_url
    if not url:
        # Find entry node URL from edges
        from collections import Counter as _Counter
        in_degree = _Counter()
        for e in edges:
            in_degree[e.target] += 1
        for e in edges:
            if in_degree[e.source] == 0 and e.source_url and str(e.source_url).startswith("http"):
                url = e.source_url
                break
    if not url:
        url = "https://the-internet.herokuapp.com/"

    data = test_data or THE_INTERNET_CREDENTIALS
    queries = intent_queries or PRIORITY_INTENTS

    # 意图 A: 优先选择包含 iframe 的业务动作
    prioritized = _prioritize_iframe_intents(queries, edges)

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
            rc_primary, rc_detail = classify_root_cause_detail(result.get("error"))
            pr = PlaybackResult(
                intent_query=q,
                success=result["success"],
                path_length=len(path),
                actual_url=result.get("actual_url", ""),
                error=result.get("error"),
                has_iframe_or_tab=has_iframe_tab,
                root_cause=rc_primary.value if not result["success"] else None,
                root_cause_detail=rc_detail if not result["success"] else None,
                failed_step_index=result.get("failed_step_index"),
                failed_edge_id=result.get("failed_edge_id"),
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

    acceptance = PlaybackAcceptanceResult(
        graph_path=str(graph_path),
        total_attempted=len(results),
        total_succeeded=succeeded,
        with_iframe_or_tab_succeeded=with_iframe_tab_succeeded,
        results=results,
    )
    _write_acceptance_report(acceptance, Path(graph_path).parent if graph_path else Path("."))
    return acceptance


def _write_acceptance_report(
    result: PlaybackAcceptanceResult,
    output_dir: Path,
) -> Path:
    """Serialize *PlaybackAcceptanceResult* to ``acceptance_report.json``."""
    root_cause_counts: dict[str, int] = Counter()
    root_cause_detail_counts: dict[str, int] = Counter()
    for r in result.results:
        if r.root_cause:
            root_cause_counts[r.root_cause] += 1
        if r.root_cause_detail:
            root_cause_detail_counts[r.root_cause_detail] += 1

    report = {
        "run_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph_path": result.graph_path,
        "total_attempted": result.total_attempted,
        "total_succeeded": result.total_succeeded,
        "with_iframe_or_tab_succeeded": result.with_iframe_or_tab_succeeded,
        "results": [asdict(r) for r in result.results],
        "summary": {
            "root_cause_distribution": dict(root_cause_counts),
            "root_cause_detail_distribution": dict(root_cause_detail_counts),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "acceptance_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path

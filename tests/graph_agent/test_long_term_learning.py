"""长期学习沉淀闭环 (L1-L5) 的集成单测。

覆盖以下功能：
- CoverageSnapshot 模型本身可建可序列化
- ExplorationScheduler 注入的 candidate 在 ``rank_warm_start_candidates`` 中
  排在静态 warm-start 之前
- persistence 层会在写完 GraphRelease 后调用 ``add_coverage_snapshot`` /
  ``link_session_coverage`` / ``link_release_coverage``，并在每条带 intent
  的 transition 上调 ``link_zone_covers_intent``
- ``add_transition_entity_with_session`` 替代了原来的 ``add_transition_entity``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph_agent.cartography.mapping_pipeline import rank_warm_start_candidates
from graph_agent.models import (
    ActionType,
    CoverageReport,
    CoverageSnapshot,
    Intent,
    State,
    Transition,
    TransitionConfidenceDistribution,
)


# ──────────────────────────── L1: model + ranking ───────────────────────────


def test_coverage_snapshot_model_defaults():
    snap = CoverageSnapshot(
        id="cov:s:1",
        app_id="app:demo",
        session_id="session:1",
        release_id="release:1",
        menu_coverage=0.5,
        zone_coverage=0.7,
        interaction_coverage=0.4,
        overall_completeness=0.55,
        recommendation="needs_more",
    )
    dumped = snap.model_dump()
    assert dumped["app_id"] == "app:demo"
    assert dumped["zone_coverage"] == 0.7
    assert dumped["recommendation"] == "needs_more"
    assert isinstance(snap.captured_at, datetime)


def test_scheduler_candidates_rank_above_warm_start_candidates():
    """ExplorationScheduler 注入的 ``scheduler_priority`` 高 → 排第一。
    其次是 ``zone_unexplored=True`` 的静态 warm-start，再次是按 confidence。"""
    candidates = [
        # 普通 warm-start: 高置信但已探完 zone
        {
            "transition_id": "t:high-conf-no-zone",
            "confidence": 0.95,
            "target_url": "https://demo/a",
            "zone_unexplored": False,
        },
        # 普通 warm-start: 中等 confidence 但 zone 未探
        {
            "transition_id": "t:mid-conf-with-zone",
            "confidence": 0.6,
            "target_url": "https://demo/b",
            "zone_unexplored": True,
        },
        # scheduler 注入: priority=80（explore_zone）
        {
            "transition_id": "sched:zid:1",
            "confidence": 0.4,
            "target_url": "https://demo/c",
            "zone_unexplored": False,
            "scheduler_priority": 80,
            "scheduler_task_type": "explore_zone",
        },
        # scheduler 注入: priority=100（discover_page，未探页面）
        {
            "transition_id": "sched:sid:1",
            "confidence": 0.0,
            "target_url": "https://demo/d",
            "zone_unexplored": True,
            "scheduler_priority": 100,
            "scheduler_task_type": "discover_page",
        },
    ]
    ranked = rank_warm_start_candidates(candidates)
    ids = [str(c["transition_id"]) for c in ranked]
    assert ids[0] == "sched:sid:1"  # discover_page (priority 100) 排第一
    assert ids[1] == "sched:zid:1"  # explore_zone (priority 80) 排第二
    # zone_unexplored=True 的 warm-start 排在 zone_unexplored=False 之前
    assert ids[2] == "t:mid-conf-with-zone"
    assert ids[3] == "t:high-conf-no-zone"


# ────────────────────── L1 + L2 + L4: persistence flow ──────────────────────


def _make_transition(
    *,
    tid: str,
    intent_key: str = "demo.intent",
    selector: str = ".action-bar button.create",
    confidence: float = 0.8,
) -> Transition:
    return Transition(
        id=tid,
        selector=selector,
        action=ActionType.CLICK,
        from_state_id="s:from",
        to_state_id="s:to",
        confidence=confidence,
        intent=Intent(
            id="",
            key=intent_key,
            summary=f"summary-{intent_key}",
            confidence=0.7,
        ),
        semantic_action_key=f"sak:{intent_key}",
    )


def _make_state(*, sid: str, url: str) -> State:
    return State(id=sid, url=url, title="t")


@dataclass
class _CapturedCalls:
    coverage_snapshots: list[CoverageSnapshot]
    session_cov_links: list[tuple[str, str]]
    release_cov_links: list[tuple[str, str]]
    intent_added: list[Intent]
    realizes_links: list[tuple[str, str]]
    zone_intent_links: list[dict[str, Any]]
    entity_session_calls: list[tuple[str, str]]
    app_session_links: list[tuple[str, str]]
    zone_batches: list[list[dict[str, Any]]]
    deactivated_release_calls: list[tuple[str, str]]
    session_stats_payloads: list[dict[str, Any]]


class _FakeManager:
    """足够用于 persist_mapping_result 的 GraphManager 替身。

    所有写接口都是 no-op；只把我们关心的几个调用记下来。
    """

    def __init__(self, captured: _CapturedCalls) -> None:
        self._captured = captured
        self._driver = MagicMock()
        self._driver.driver = MagicMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def get_driver(self):
        return self._driver.driver

    # — generic no-ops —
    async def add_app(self, *_a, **_k):
        return None

    async def add_session(self, *_a, **_k):
        return None

    async def link_app_session(self, *_a, **_k):
        app_id = str(_k.get("app_id") if "app_id" in _k else _a[0])
        session_id = str(_k.get("session_id") if "session_id" in _k else _a[1])
        self._captured.app_session_links.append((app_id, session_id))
        return None

    async def add_ingestion_run(self, *_a, **_k):
        return None

    async def run_write_transaction(self, *_a, **_k):
        return None

    async def set_session_inventory(self, *_a, **_k):
        return None

    async def add_state(self, *_a, **_k):
        return None

    async def link_app_state(self, *_a, **_k):
        return None

    async def link_session_discovered(self, *_a, **_k):
        return None

    async def link_ingestion_emits_state(self, *_a, **_k):
        return None

    async def add_transition(self, *_a, **_k):
        return None

    async def link_session_transition(self, *_a, **_k):
        return None

    async def link_ingestion_emits_transition(self, *_a, **_k):
        return None

    async def link_ingestion_emits_evidence(self, *_a, **_k):
        return None

    async def get_active_transition_revision(self, *_a, **_k):
        return None

    async def add_transition_revision(self, *_a, **_k):
        return None

    async def link_ingestion_emits_revision(self, *_a, **_k):
        return None

    async def activate_transition_revision(self, *_a, **_k):
        return None

    async def attach_transition_revision(self, *_a, **_k):
        return None

    async def add_evidence(self, *_a, **_k):
        return None

    async def link_transition_evidence(self, *_a, **_k):
        return None

    async def link_session_evidence(self, *_a, **_k):
        return None

    async def add_menus(self, *_a, **_k):
        return None

    async def link_ingestion_emits_menu(self, *_a, **_k):
        return None

    async def add_zones(self, *_a, **_k):
        zones = _k.get("zones")
        if isinstance(zones, list):
            self._captured.zone_batches.append(zones)
        return None

    async def link_ingestion_emits_zone(self, *_a, **_k):
        return None

    async def link_transition_navigated_via(self, *_a, **_k):
        return None

    async def add_graph_release(self, *_a, **_k):
        return None

    async def deactivate_other_active_releases(self, *_a, **_k):
        app_id = str(_k.get("app_id") if "app_id" in _k else _a[0])
        keep_release_id = str(
            _k.get("keep_release_id") if "keep_release_id" in _k else _a[1]
        )
        self._captured.deactivated_release_calls.append((app_id, keep_release_id))
        return None

    async def link_release_revision(self, *_a, **_k):
        return None

    async def update_session_stats(self, *_a, **_k):
        stats = _k.get("stats")
        if isinstance(stats, dict):
            self._captured.session_stats_payloads.append(stats)
        return None

    async def touch_app_last_session(self, *_a, **_k):
        return None

    async def update_app_stats(self, *_a, **_k):
        return None

    async def get_latest_semantic_baseline(self, *_a, **_k):
        return {}

    # — 我们关心的几个 —
    async def add_transition_entity_with_session(
        self, entity, session_id: str
    ) -> None:
        self._captured.entity_session_calls.append((entity.stable_key, session_id))

    async def add_intent(self, intent: Intent) -> None:
        self._captured.intent_added.append(intent)

    async def link_transition_intent(
        self, transition_id: str, intent_id: str
    ) -> None:
        self._captured.realizes_links.append((transition_id, intent_id))

    async def link_zone_covers_intent(
        self,
        *,
        from_state_id: str,
        selector: str,
        intent_id: str,
        confidence: float,
        session_id: str,
    ) -> None:
        self._captured.zone_intent_links.append(
            {
                "from_state_id": from_state_id,
                "selector": selector,
                "intent_id": intent_id,
                "confidence": confidence,
                "session_id": session_id,
            }
        )

    async def add_coverage_snapshot(self, snap: CoverageSnapshot) -> None:
        self._captured.coverage_snapshots.append(snap)

    async def link_session_coverage(
        self, session_id: str, coverage_id: str
    ) -> None:
        self._captured.session_cov_links.append((session_id, coverage_id))

    async def link_release_coverage(
        self, release_id: str, coverage_id: str
    ) -> None:
        self._captured.release_cov_links.append((release_id, coverage_id))


@pytest.mark.asyncio
async def test_persist_writes_coverage_snapshot_and_zone_intent_edges():
    from graph_agent.cartography import persistence
    from graph_agent.graph.merger import CartographyResult

    captured = _CapturedCalls(
        coverage_snapshots=[],
        session_cov_links=[],
        release_cov_links=[],
        intent_added=[],
        realizes_links=[],
        zone_intent_links=[],
        entity_session_calls=[],
        app_session_links=[],
        zone_batches=[],
        deactivated_release_calls=[],
        session_stats_payloads=[],
    )

    fake_report = CoverageReport(
        menu_coverage=0.6,
        zone_coverage=0.8,
        interaction_coverage=0.5,
        state_coverage=0.4,
        transition_confidence=TransitionConfidenceDistribution(
            high=3, medium=2, low=1
        ),
        overall_completeness=0.65,
        recommendation="needs_more",
    )
    analyzer_mock = MagicMock()
    analyzer_mock.compute = AsyncMock(return_value=fake_report)
    analyzer_factory = MagicMock(return_value=analyzer_mock)

    transitions = [
        _make_transition(tid="t:1", intent_key="create_user"),
        _make_transition(tid="t:2", intent_key="search_user"),
    ]
    states = [
        _make_state(sid="s:from", url="https://demo/users"),
        _make_state(sid="s:to", url="https://demo/users/new"),
    ]
    result = CartographyResult(
        states=states,
        transitions=transitions,
        zones=[],
        zone_hints=[
            {
                "zone_type": "form",
                "selector": ".main-form",
                "description": "Main form",
                "source_url": "https://demo/users",
                "exploration_status": "discovered",
            }
        ],
        history=[{"url": "https://demo/users", "result": ""}],
        layout_metrics={"layout_confidence_avg": 0.88, "skip_page_count": 2},
    )

    # GraphManager / CoverageAnalyzer 都在 persist_mapping_result 里 lazy-import，
    # 所以需要 patch 它们的源模块属性。
    with patch(
        "graph_agent.neo4j_client.manager.GraphManager",
        lambda: _FakeManager(captured),
    ), patch(
        "graph_agent.coverage.analyzer.CoverageAnalyzer",
        analyzer_factory,
    ):
        await persistence.persist_mapping_result(
            app_id="app:demo:202605",
            app_name="demo",
            session_id="session:demo:1",
            resolved_url="https://demo/users",
            current_url="https://demo/users",
            inventory=[],
            initial_actions_log=[],
            result=result,
        )

    # — L1: CoverageSnapshot 写入 + 双向链接 —
    assert len(captured.coverage_snapshots) == 1
    snap = captured.coverage_snapshots[0]
    assert snap.app_id == "app:demo:202605"
    assert snap.session_id == "session:demo:1"
    assert snap.menu_coverage == pytest.approx(0.6)
    assert snap.zone_coverage == pytest.approx(0.8)
    assert snap.state_coverage == pytest.approx(0.4)
    assert snap.overall_completeness == pytest.approx(0.65)
    assert snap.transition_high == 3
    assert snap.transition_low == 1
    assert captured.session_cov_links == [("session:demo:1", snap.id)]
    assert len(captured.release_cov_links) == 1
    assert captured.release_cov_links[0][1] == snap.id

    # — L2: 每条 transition 写 Intent + REALIZES + zone-COVERS_INTENT —
    assert len(captured.intent_added) == 2
    assert all(i.id.startswith("intent:") for i in captured.intent_added)
    assert len(captured.realizes_links) == 2
    assert {tid for tid, _ in captured.realizes_links} == {"t:1", "t:2"}
    assert len(captured.zone_intent_links) == 2
    for link in captured.zone_intent_links:
        assert link["from_state_id"] == "s:from"
        assert link["selector"] == ".action-bar button.create"
        assert link["intent_id"].startswith("intent:")
        assert link["session_id"] == "session:demo:1"

    # — L4: TransitionEntity 升级到 with_session 接口 —
    assert len(captured.entity_session_calls) == 2
    assert all(sid == "session:demo:1" for _, sid in captured.entity_session_calls)

    # — Phase1: App-Session 关系与 State-Zone 回填 —
    assert captured.app_session_links == [("app:demo:202605", "session:demo:1")]
    assert len(captured.zone_batches) == 1
    assert captured.zone_batches[0][0]["state_ids"] == ["s:from"]
    assert len(captured.deactivated_release_calls) == 1
    assert captured.deactivated_release_calls[0][0] == "app:demo:202605"
    assert captured.deactivated_release_calls[0][1].startswith("release:app:demo:202605:")
    assert captured.session_stats_payloads, "expected update_session_stats call"
    session_stats = captured.session_stats_payloads[0]
    assert session_stats.get("layout_confidence_avg") == pytest.approx(0.88)
    assert session_stats.get("skip_page_count") == 2
    assert session_stats.get("semantic_stability_score") == pytest.approx(100.0)
    assert session_stats.get("semantic_stability_passed") is True
    assert session_stats.get("semantic_stability_mode") == "bootstrap"


@pytest.mark.asyncio
async def test_zone_hints_with_same_selector_are_scoped_by_source_page():
    from graph_agent.cartography import persistence
    from graph_agent.graph.merger import CartographyResult

    captured = _CapturedCalls(
        coverage_snapshots=[],
        session_cov_links=[],
        release_cov_links=[],
        intent_added=[],
        realizes_links=[],
        zone_intent_links=[],
        entity_session_calls=[],
        app_session_links=[],
        zone_batches=[],
        deactivated_release_calls=[],
        session_stats_payloads=[],
    )
    fake_report = CoverageReport(
        menu_coverage=0.2,
        zone_coverage=0.2,
        interaction_coverage=0.2,
        state_coverage=0.2,
        transition_confidence=TransitionConfidenceDistribution(
            high=0, medium=0, low=0
        ),
        overall_completeness=0.2,
        recommendation="needs_more",
    )
    analyzer_mock = MagicMock()
    analyzer_mock.compute = AsyncMock(return_value=fake_report)
    analyzer_factory = MagicMock(return_value=analyzer_mock)
    result = CartographyResult(
        states=[
            _make_state(sid="s:list", url="https://demo/users"),
            _make_state(sid="s:detail", url="https://demo/users/1"),
        ],
        transitions=[],
        zones=[],
        zone_hints=[
            {
                "zone_type": "table",
                "selector": ".data-table",
                "description": "User list table",
                "source_url": "https://demo/users",
                "exploration_status": "partial",
            },
            {
                "zone_type": "table",
                "selector": ".data-table",
                "description": "Related records table",
                "source_url": "https://demo/users/1",
                "exploration_status": "explored",
            },
        ],
        history=[{"url": "https://demo/users", "result": ""}],
    )

    with patch(
        "graph_agent.neo4j_client.manager.GraphManager",
        lambda: _FakeManager(captured),
    ), patch(
        "graph_agent.coverage.analyzer.CoverageAnalyzer",
        analyzer_factory,
    ):
        await persistence.persist_mapping_result(
            app_id="app:demo:zone-scope",
            app_name="demo",
            session_id="session:demo:zone-scope",
            resolved_url="https://demo/users",
            current_url="https://demo/users",
            inventory=[],
            initial_actions_log=[],
            result=result,
        )

    assert len(captured.zone_batches) == 1
    rows = captured.zone_batches[0]
    assert len(rows) == 2
    selector_rows = [row for row in rows if row["selector"] == ".data-table"]
    assert len(selector_rows) == 2
    state_id_sets = {tuple(row["state_ids"]) for row in selector_rows}
    assert state_id_sets == {("s:list",), ("s:detail",)}

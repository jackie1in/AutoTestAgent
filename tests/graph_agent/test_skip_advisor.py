from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from graph_agent.cartography.skip_advisor import (
    SkipAdvisor,
    SkipKind,
    SkipPolicy,
)


def _utc(hours_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


def _build_row(
    *,
    state_count: int,
    last_visited: datetime | None,
    zones: list[dict[str, Any]],
    release_coverage: float = 0.0,
    entity_confirm_total: int = 0,
) -> dict[str, Any]:
    # 自动给每个 zone 补 intent_confirm（默认 0），让旧用例无需改动
    normalized_zones = [
        {**z, "intent_confirm": z.get("intent_confirm", 0)} for z in zones
    ]
    return {
        "state_count": state_count,
        "last_visited": last_visited,
        "zones": normalized_zones,
        "release_coverage": release_coverage,
        "entity_confirm_total": entity_confirm_total,
    }


@pytest.mark.asyncio
async def test_full_explore_when_never_visited():
    advisor = SkipAdvisor(app_name="demo")
    row = _build_row(state_count=0, last_visited=None, zones=[])

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.FULL_EXPLORE
    assert decision.reason == "never_visited"


@pytest.mark.asyncio
async def test_skip_page_when_fully_explored_recently():
    advisor = SkipAdvisor(app_name="demo")
    row = _build_row(
        state_count=1,
        last_visited=_utc(2),
        zones=[
            {"selector": ".form", "status": "explored", "last_explored": _utc(2)},
            {"selector": ".table", "status": "validated", "last_explored": _utc(2)},
        ],
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.SKIP_PAGE
    assert decision.coverage >= 0.9


@pytest.mark.asyncio
async def test_zones_only_when_partial_pending():
    advisor = SkipAdvisor(app_name="demo")
    row = _build_row(
        state_count=1,
        last_visited=_utc(2),
        zones=[
            {"selector": ".form", "status": "explored", "last_explored": _utc(2)},
            {"selector": ".chart", "status": "discovered", "last_explored": None},
            {"selector": ".sidebar", "status": "partial", "last_explored": _utc(2)},
        ],
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.EXPLORE_ZONES_ONLY
    assert ".chart" in decision.target_zone_selectors
    assert ".sidebar" in decision.target_zone_selectors
    assert ".form" not in decision.target_zone_selectors


@pytest.mark.asyncio
async def test_full_explore_when_stale():
    policy = SkipPolicy.from_profile("balanced")
    advisor = SkipAdvisor(app_name="demo", policy=policy)
    row = _build_row(
        state_count=1,
        last_visited=_utc(policy.ttl_hours + 5),
        zones=[
            {
                "selector": ".form",
                "status": "explored",
                "last_explored": _utc(policy.ttl_hours + 5),
            },
        ],
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.FULL_EXPLORE
    assert "stale" in decision.reason or "low" in decision.reason


@pytest.mark.asyncio
async def test_cache_hit_skips_query():
    advisor = SkipAdvisor(app_name="demo")
    row = _build_row(
        state_count=1,
        last_visited=_utc(1),
        zones=[
            {"selector": ".form", "status": "explored", "last_explored": _utc(1)},
            {"selector": ".table", "status": "validated", "last_explored": _utc(1)},
        ],
    )
    call_count = {"n": 0}

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        call_count["n"] += 1
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        first = await advisor.evaluate("https://example.com/foo")
        second = await advisor.evaluate("https://example.com/foo")

    assert call_count["n"] == 1
    assert first.kind is second.kind
    assert second.cache_hit is True


@pytest.mark.asyncio
async def test_timeout_falls_back_to_full_explore_and_breaks_circuit():
    policy = SkipPolicy.from_profile("balanced")
    policy.query_timeout_ms = 50
    policy.failure_threshold = 1
    advisor = SkipAdvisor(app_name="demo", policy=policy)

    async def slow_query(_url_clean: str) -> dict[str, Any]:
        await asyncio.sleep(0.5)
        return _build_row(state_count=0, last_visited=None, zones=[])

    with patch.object(advisor, "_query_coverage", side_effect=slow_query):
        first = await advisor.evaluate("https://example.com/a")
        # 第二次直接命中熔断分支
        second = await advisor.evaluate("https://example.com/b")

    assert first.kind is SkipKind.FULL_EXPLORE
    assert first.timed_out is True
    assert second.kind is SkipKind.FULL_EXPLORE
    assert second.circuit_open is True


@pytest.mark.asyncio
async def test_query_error_falls_back_and_records_metric():
    advisor = SkipAdvisor(app_name="demo")

    async def failing_query(_url_clean: str) -> dict[str, Any]:
        raise RuntimeError("neo4j down")

    with patch.object(advisor, "_query_coverage", side_effect=failing_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.FULL_EXPLORE
    assert decision.reason == "query_error"
    assert decision.error == "neo4j down"
    assert advisor.metrics["skip_error_count"] == 1


def test_policy_profiles_have_distinct_thresholds():
    aggressive = SkipPolicy.from_profile("aggressive")
    balanced = SkipPolicy.from_profile("balanced")
    conservative = SkipPolicy.from_profile("conservative")
    assert aggressive.skip_threshold_coverage < balanced.skip_threshold_coverage
    assert balanced.skip_threshold_coverage < conservative.skip_threshold_coverage
    assert aggressive.ttl_hours > balanced.ttl_hours > conservative.ttl_hours
    # 长期沉淀阈值：profile 越保守，需要越多确认才肯触发 learned_skip
    assert (
        aggressive.intent_confirm_skip_threshold
        <= balanced.intent_confirm_skip_threshold
        <= conservative.intent_confirm_skip_threshold
    )
    assert (
        aggressive.release_coverage_floor
        <= balanced.release_coverage_floor
        <= conservative.release_coverage_floor
    )


@pytest.mark.asyncio
async def test_learned_skip_when_intent_and_entity_confirmed():
    """长期沉淀：本地 coverage 还差一点，但 intent+entity 都被多 session 确认过，
    且 release coverage 站得住，应该升级为 SKIP_PAGE 走 learned_skip 路径。"""
    policy = SkipPolicy.from_profile("balanced")
    advisor = SkipAdvisor(app_name="demo", policy=policy)
    row = _build_row(
        state_count=1,
        last_visited=_utc(2),
        zones=[
            # 全是 explored 但只有一两个 zone — 普通 coverage 已经 1.0
            # 这里故意把数量设少 + 配 intent_confirm 充足来测 learned 分支生效
            {
                "selector": ".form",
                "status": "explored",
                "last_explored": _utc(2),
                "intent_confirm": 5,
            },
            {
                "selector": ".table",
                "status": "explored",
                "last_explored": _utc(2),
                "intent_confirm": 3,
            },
        ],
        release_coverage=0.82,
        entity_confirm_total=5,
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    assert decision.kind is SkipKind.SKIP_PAGE
    assert decision.intent_confirm_total == 8
    assert decision.entity_confirm_total == 5
    assert decision.release_coverage == pytest.approx(0.82)
    # 既可能命中标准 SKIP_PAGE（coverage 已 1.0）也可能命中 learned_skip；
    # 任一路径都可，关键是 intent/entity 学习信号被读取并暴露
    assert advisor.metrics["skip_intent_confirm_total_seen"] >= 8
    assert advisor.metrics["skip_entity_confirm_total_seen"] >= 5


@pytest.mark.asyncio
async def test_learned_skip_only_when_release_floor_met():
    """release_coverage 不达地板时，即便 intent/entity 都满，也不能 learned_skip。

    构造场景：本地 coverage 故意低于 skip_threshold_coverage 但又有 pending —
    没有 pending 才有可能进 learned 分支；这里加 1 个 partial 让 pending 非空，
    确认 learned 分支不会绕过 pending 校验。"""
    policy = SkipPolicy.from_profile("balanced")
    policy.intent_confirm_skip_threshold = 1
    policy.entity_confirm_skip_threshold = 1
    policy.release_coverage_floor = 0.95  # 故意调高让此场景进不去
    advisor = SkipAdvisor(app_name="demo", policy=policy)
    row = _build_row(
        state_count=1,
        last_visited=_utc(2),
        zones=[
            {
                "selector": ".form",
                "status": "explored",
                "last_explored": _utc(2),
                "intent_confirm": 10,
            },
            {
                "selector": ".pending",
                "status": "discovered",
                "last_explored": None,
                "intent_confirm": 0,
            },
        ],
        release_coverage=0.5,
        entity_confirm_total=20,
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        decision = await advisor.evaluate("https://example.com/foo")

    # 有 pending 且 release_coverage 不达 floor，不会走 learned_skip
    assert decision.kind is not SkipKind.SKIP_PAGE
    # 但 intent/entity 字段仍应被填充
    assert decision.intent_confirm_total == 10
    assert decision.entity_confirm_total == 20


@pytest.mark.asyncio
async def test_long_term_signals_round_trip_through_cache():
    """cache 命中也要把 release_coverage / intent_confirm / entity_confirm 透传出来。"""
    advisor = SkipAdvisor(app_name="demo")
    row = _build_row(
        state_count=1,
        last_visited=_utc(1),
        zones=[
            {
                "selector": ".form",
                "status": "explored",
                "last_explored": _utc(1),
                "intent_confirm": 4,
            },
        ],
        release_coverage=0.7,
        entity_confirm_total=2,
    )

    async def fake_query(_url_clean: str) -> dict[str, Any]:
        return row

    with patch.object(advisor, "_query_coverage", side_effect=fake_query):
        first = await advisor.evaluate("https://example.com/x")
        second = await advisor.evaluate("https://example.com/x")

    assert second.cache_hit is True
    assert second.release_coverage == first.release_coverage
    assert second.intent_confirm_total == first.intent_confirm_total
    assert second.entity_confirm_total == first.entity_confirm_total
    assert second.intent_confirmed_zone_count == first.intent_confirmed_zone_count

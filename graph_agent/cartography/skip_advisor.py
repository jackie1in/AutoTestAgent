"""SkipAdvisor — 跨 session 跳过已探索区域的决策中心。

设计要点：
- 复用现有 ``Zone.exploration_status`` 与 ``State.last_visited`` 字段，不引入新的图节点。
- 三态决策：``SKIP_PAGE`` / ``EXPLORE_ZONES_ONLY`` / ``FULL_EXPLORE``。
- 内置 LRU 缓存、超时、熔断；任何异常路径都退化为 ``FULL_EXPLORE``，确保主流程不被阻断。

调用方：``mapping_pipeline._enqueue_page`` 入队前；主循环 pop 出 url 后再判定一次。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from graph_agent.cartography.config import clean_url
from graph_agent.neo4j_client.manager import GraphManager


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _parse_iso_or_neo_datetime(value: object) -> datetime | None:
    """把 Neo4j 返回的 datetime / ISO string / None 统一成 timezone-aware datetime。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    iso_method = getattr(value, "iso_format", None)
    if callable(iso_method):
        try:
            return _parse_iso_or_neo_datetime(iso_method())
        except Exception:
            return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _hours_since(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    now = datetime.now(timezone.utc)
    delta = now - ts
    return max(0.0, delta.total_seconds() / 3600.0)


class SkipKind(str, Enum):
    SKIP_PAGE = "skip_page"
    EXPLORE_ZONES_ONLY = "explore_zones_only"
    FULL_EXPLORE = "full_explore"


@dataclass(slots=True)
class SkipDecision:
    kind: SkipKind = SkipKind.FULL_EXPLORE
    reason: str = ""
    confidence: float = 0.0
    target_zone_selectors: list[str] = field(default_factory=list)
    coverage: float = 0.0
    state_count: int = 0
    last_visited_age_h: float | None = None
    last_explored_age_h: float | None = None
    # 长期沉淀信号 —— 来自 GraphRelease/CoverageSnapshot 与 zone-intent 反向边
    release_coverage: float = 0.0
    intent_confirm_total: int = 0
    entity_confirm_total: int = 0
    intent_confirmed_zone_count: int = 0
    cache_hit: bool = False
    timed_out: bool = False
    circuit_open: bool = False
    error: str = ""
    query_latency_ms: float = 0.0


@dataclass(slots=True)
class SkipPolicy:
    profile: str = "balanced"
    skip_threshold_coverage: float = 0.9
    zones_only_threshold: float = 0.4
    ttl_hours: float = 24.0
    query_timeout_ms: int = 300
    cache_ttl_sec: float = 60.0
    failure_threshold: int = 3
    cooldown_sec: float = 60.0
    # 长期沉淀触发的"业务确认 skip" —— 当 zone 上的 intent_confirm_total
    # 超过该阈值时，即便本地 coverage 略低于 skip_threshold_coverage，
    # 也允许提升为 SKIP_PAGE（这是"越用越聪明"的核心）
    intent_confirm_skip_threshold: int = 3
    entity_confirm_skip_threshold: int = 3
    release_coverage_floor: float = 0.7

    @classmethod
    def from_profile(cls, profile: str) -> "SkipPolicy":
        p = (profile or "balanced").strip().lower()
        if p == "aggressive":
            return cls(
                profile="aggressive",
                skip_threshold_coverage=0.7,
                zones_only_threshold=0.3,
                ttl_hours=72.0,
                intent_confirm_skip_threshold=2,
                entity_confirm_skip_threshold=2,
                release_coverage_floor=0.55,
            )
        if p == "conservative":
            return cls(
                profile="conservative",
                skip_threshold_coverage=0.95,
                zones_only_threshold=0.6,
                ttl_hours=8.0,
                intent_confirm_skip_threshold=5,
                entity_confirm_skip_threshold=5,
                release_coverage_floor=0.85,
            )
        return cls(profile="balanced")


class SkipAdvisor:
    """决策器：对单个 URL 给出 SkipDecision。"""

    # 当 zone 总数为 0 但 state 已访问过时使用的 fallback coverage。
    _NO_ZONE_BASE_COVERAGE = 0.5

    def __init__(
        self,
        *,
        app_id: str = "",
        app_name: str,
        primary_origin_url: str = "",
        policy: SkipPolicy | None = None,
    ) -> None:
        self._app_id = (app_id or "").strip()
        self._app_name = (app_name or "").strip()
        self._primary_origin_url = (primary_origin_url or "").strip()
        self._policy = policy or SkipPolicy()
        self._cache: dict[str, tuple[float, SkipDecision]] = {}
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._eval_count = 0
        self._cache_hit_count = 0
        self._timeout_count = 0
        self._error_count = 0
        self._circuit_open_event_count = 0
        self._latency_sum_ms = 0.0
        self._skip_page_count = 0
        self._zones_only_count = 0
        self._full_explore_count = 0
        self._learned_skip_count = 0
        self._intent_confirm_total_seen = 0
        self._entity_confirm_total_seen = 0

    @property
    def policy(self) -> SkipPolicy:
        return self._policy

    @property
    def metrics(self) -> dict[str, object]:
        avg_latency = (
            self._latency_sum_ms / self._eval_count if self._eval_count > 0 else 0.0
        )
        return {
            "skip_evaluations": self._eval_count,
            "skip_cache_hit_count": self._cache_hit_count,
            "skip_timeout_count": self._timeout_count,
            "skip_error_count": self._error_count,
            "skip_circuit_open_count": self._circuit_open_event_count,
            "skip_query_avg_ms": round(avg_latency, 3),
            "skip_page_count": self._skip_page_count,
            "skip_zones_only_count": self._zones_only_count,
            "skip_full_explore_count": self._full_explore_count,
            "skip_learned_skip_count": self._learned_skip_count,
            "skip_intent_confirm_total_seen": self._intent_confirm_total_seen,
            "skip_entity_confirm_total_seen": self._entity_confirm_total_seen,
            "skip_policy_profile": self._policy.profile,
        }

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._policy.failure_threshold:
            self._circuit_open_until = (
                time.monotonic() + self._policy.cooldown_sec
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _account_decision(self, decision: SkipDecision) -> None:
        if decision.kind is SkipKind.SKIP_PAGE:
            self._skip_page_count += 1
            if decision.reason.startswith("learned_skip"):
                self._learned_skip_count += 1
        elif decision.kind is SkipKind.EXPLORE_ZONES_ONLY:
            self._zones_only_count += 1
        else:
            self._full_explore_count += 1
        self._intent_confirm_total_seen += decision.intent_confirm_total
        self._entity_confirm_total_seen += decision.entity_confirm_total

    async def evaluate(self, url: str) -> SkipDecision:
        """返回该 url 的 SkipDecision；任何异常都退化为 FULL_EXPLORE。"""
        started = time.monotonic()
        self._eval_count += 1
        url_clean = clean_url(url or "")
        if (not self._app_id and not self._app_name) or not url_clean:
            decision = SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason="missing_app_identity_or_url",
            )
            self._account_decision(decision)
            return decision

        # circuit breaker
        if self._is_circuit_open():
            decision = SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason="circuit_open",
                circuit_open=True,
            )
            self._circuit_open_event_count += 1
            self._account_decision(decision)
            return decision

        # cache
        now = time.monotonic()
        cached = self._cache.get(url_clean)
        if cached and cached[0] > now:
            cached_decision = cached[1]
            decision = SkipDecision(
                kind=cached_decision.kind,
                reason=cached_decision.reason,
                confidence=cached_decision.confidence,
                target_zone_selectors=list(cached_decision.target_zone_selectors),
                coverage=cached_decision.coverage,
                state_count=cached_decision.state_count,
                last_visited_age_h=cached_decision.last_visited_age_h,
                last_explored_age_h=cached_decision.last_explored_age_h,
                release_coverage=cached_decision.release_coverage,
                intent_confirm_total=cached_decision.intent_confirm_total,
                entity_confirm_total=cached_decision.entity_confirm_total,
                intent_confirmed_zone_count=cached_decision.intent_confirmed_zone_count,
                cache_hit=True,
                query_latency_ms=(time.monotonic() - started) * 1000,
            )
            self._cache_hit_count += 1
            self._account_decision(decision)
            return decision

        # query Neo4j
        try:
            row = await asyncio.wait_for(
                self._query_coverage(url_clean),
                timeout=max(0.1, self._policy.query_timeout_ms / 1000.0),
            )
            decision = self._build_decision(row)
            decision.query_latency_ms = (time.monotonic() - started) * 1000
            self._latency_sum_ms += decision.query_latency_ms
            self._record_success()
            self._cache[url_clean] = (
                time.monotonic() + self._policy.cache_ttl_sec,
                decision,
            )
            self._account_decision(decision)
            return decision
        except (TimeoutError, asyncio.TimeoutError):
            self._timeout_count += 1
            self._record_failure()
            decision = SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason="query_timeout",
                timed_out=True,
                query_latency_ms=(time.monotonic() - started) * 1000,
            )
            self._account_decision(decision)
            return decision
        except Exception as e:
            self._error_count += 1
            self._record_failure()
            decision = SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason="query_error",
                error=str(e),
                query_latency_ms=(time.monotonic() - started) * 1000,
            )
            self._account_decision(decision)
            return decision

    async def _query_coverage(self, url_clean: str) -> dict[str, Any]:
        """聚合本 URL 的 zone 状态 + release coverage + 长期确认计数。

        长期沉淀字段：
        - ``release_coverage``：来自 ``GraphRelease.coverage_overall``（由 CoverageSnapshot 同步），
          反映整个 app 在最近 release 的整体覆盖度。
        - 每个 zone 的 ``intent_confirm`` 来自 ``COVERS_INTENT.observed_count`` 累加，
          ``_build_decision`` 据此聚合出 ``intent_confirm_total``。
        - ``entity_confirm_total``：本页 from_state 上挂的 ``TransitionEntity.confirmed_session_count``
          中超过 1 的部分总和（衡量"已被多个独立 session 复现"的强度）。
        """
        async with GraphManager() as manager:
            row = await manager.get_skip_advisor_coverage(
                url_clean=url_clean,
                app_id=self._app_id,
                app_name=self._app_name,
            )
            if not row:
                return {
                    "state_count": 0,
                    "last_visited": None,
                    "zones": [],
                    "release_coverage": 0.0,
                    "entity_confirm_total": 0,
                }
            return row

    def _build_decision(self, row: dict[str, Any]) -> SkipDecision:
        state_count = int(row.get("state_count") or 0)
        last_visited = _parse_iso_or_neo_datetime(row.get("last_visited"))
        last_visited_age_h = _hours_since(last_visited)
        release_coverage = _as_float(row.get("release_coverage"), 0.0)
        entity_confirm_total = int(row.get("entity_confirm_total") or 0)
        raw_zones = row.get("zones") or []
        zones: list[dict[str, Any]] = []
        for z in raw_zones:
            if not isinstance(z, dict):
                continue
            selector = str(z.get("selector") or "").strip()
            if not selector:
                continue
            zones.append(
                {
                    "selector": selector,
                    "status": str(z.get("status") or "undiscovered").strip().lower(),
                    "last_explored": _parse_iso_or_neo_datetime(z.get("last_explored")),
                    "intent_confirm": int(z.get("intent_confirm") or 0),
                }
            )

        intent_confirm_total = sum(z["intent_confirm"] for z in zones)
        intent_confirmed_zone_count = sum(
            1 for z in zones if z["intent_confirm"] > 0
        )

        # 没探索过 → 全探索
        if state_count == 0 and not zones:
            return SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason="never_visited",
                state_count=0,
                last_visited_age_h=last_visited_age_h,
                release_coverage=release_coverage,
            )

        # 计算 coverage
        last_explored_max: datetime | None = None
        if zones:
            explored = [
                z for z in zones if z["status"] in ("explored", "validated")
            ]
            partial = [z for z in zones if z["status"] == "partial"]
            pending = [
                z
                for z in zones
                if z["status"] in ("undiscovered", "discovered", "stale")
            ]
            total = len(zones)
            coverage = (len(explored) + 0.5 * len(partial)) / max(1, total)
            for z in zones:
                if z["last_explored"] is not None:
                    if last_explored_max is None or z["last_explored"] > last_explored_max:
                        last_explored_max = z["last_explored"]
            pending_selectors = [z["selector"] for z in pending + partial]
        else:
            # 有 state 没 zone：fallback
            coverage = self._NO_ZONE_BASE_COVERAGE
            pending_selectors = []

        last_explored_age_h = _hours_since(last_explored_max)

        # TTL 校验
        ttl = self._policy.ttl_hours
        # 优先用 last_explored，其次用 last_visited
        ref_age = (
            last_explored_age_h
            if last_explored_age_h is not None
            else last_visited_age_h
        )
        within_ttl = ref_age is not None and ref_age <= ttl

        def _decorate(decision: SkipDecision) -> SkipDecision:
            decision.release_coverage = release_coverage
            decision.intent_confirm_total = intent_confirm_total
            decision.intent_confirmed_zone_count = intent_confirmed_zone_count
            decision.entity_confirm_total = entity_confirm_total
            return decision

        # 决策 1: 标准 SKIP_PAGE（覆盖率 + TTL 双达标，且没有 pending）
        if (
            coverage >= self._policy.skip_threshold_coverage
            and within_ttl
            and not pending_selectors
        ):
            return _decorate(
                SkipDecision(
                    kind=SkipKind.SKIP_PAGE,
                    reason=(
                        f"coverage={coverage:.2f}>=thr={self._policy.skip_threshold_coverage:.2f} "
                        f"and age={ref_age:.1f}h<=ttl={ttl:.1f}h"
                    ),
                    confidence=min(1.0, coverage),
                    coverage=coverage,
                    state_count=state_count,
                    last_visited_age_h=last_visited_age_h,
                    last_explored_age_h=last_explored_age_h,
                )
            )

        # 决策 2: "学习沉淀"驱动的 SKIP_PAGE —— 不依赖本地 coverage 严格达标，
        # 而是看跨 session 的强信号：intent 与 entity 都已多次确认，
        # 且 release 整体 coverage 也站得住，就允许 SKIP。这是"越用越聪明"的核心。
        if (
            within_ttl
            and not pending_selectors
            and intent_confirm_total >= self._policy.intent_confirm_skip_threshold
            and entity_confirm_total >= self._policy.entity_confirm_skip_threshold
            and release_coverage >= self._policy.release_coverage_floor
        ):
            return _decorate(
                SkipDecision(
                    kind=SkipKind.SKIP_PAGE,
                    reason=(
                        f"learned_skip: intent_confirm={intent_confirm_total} "
                        f"(>= {self._policy.intent_confirm_skip_threshold}), "
                        f"entity_confirm={entity_confirm_total} "
                        f"(>= {self._policy.entity_confirm_skip_threshold}), "
                        f"release_cov={release_coverage:.2f}"
                    ),
                    confidence=min(1.0, max(coverage, release_coverage)),
                    coverage=coverage,
                    state_count=state_count,
                    last_visited_age_h=last_visited_age_h,
                    last_explored_age_h=last_explored_age_h,
                )
            )

        if (
            self._policy.zones_only_threshold <= coverage
            and pending_selectors
            and within_ttl
        ):
            return _decorate(
                SkipDecision(
                    kind=SkipKind.EXPLORE_ZONES_ONLY,
                    reason=(
                        f"coverage={coverage:.2f}, "
                        f"pending_zones={len(pending_selectors)}, age={ref_age:.1f}h, "
                        f"intent_confirm={intent_confirm_total}"
                    ),
                    confidence=coverage,
                    target_zone_selectors=pending_selectors,
                    coverage=coverage,
                    state_count=state_count,
                    last_visited_age_h=last_visited_age_h,
                    last_explored_age_h=last_explored_age_h,
                )
            )

        return _decorate(
            SkipDecision(
                kind=SkipKind.FULL_EXPLORE,
                reason=(
                    "low_coverage_or_stale"
                    if coverage < self._policy.zones_only_threshold or not within_ttl
                    else "default_full_explore"
                ),
                coverage=coverage,
                state_count=state_count,
                last_visited_age_h=last_visited_age_h,
                last_explored_age_h=last_explored_age_h,
                target_zone_selectors=pending_selectors,
            )
        )

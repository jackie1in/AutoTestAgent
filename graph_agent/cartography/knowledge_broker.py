from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from graph_agent.neo4j_client.manager import GraphManager


def _as_str(value: object) -> str:
    return str(value or "")


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _extract_app_name_from_app_id(app_id: str) -> str:
    # app:<name>:<timestamp>
    parts = app_id.split(":")
    if len(parts) >= 3 and parts[0] == "app":
        return parts[1]
    return app_id


@dataclass(slots=True)
class KnowledgeQueryInput:
    app_id: str
    session_id: str
    current_url: str
    page_type: str
    layout_fingerprint: str = ""
    recent_selector: str = ""
    recent_action: str = ""
    release_id: str = ""
    signals: dict[str, float] = field(default_factory=dict)
    top_k: int = 5


@dataclass(slots=True)
class KnowledgeQueryMeta:
    source: str = "none"
    cache_hit: bool = False
    circuit_open: bool = False
    timed_out: bool = False
    query_latency_ms: float = 0.0
    error: str = ""


@dataclass(slots=True)
class KnowledgeQueryResult:
    summary: str = ""
    transition_hints: list[dict[str, object]] = field(default_factory=list)
    state_hints: list[dict[str, object]] = field(default_factory=list)
    intent_hints: list[dict[str, object]] = field(default_factory=list)
    meta: KnowledgeQueryMeta = field(default_factory=KnowledgeQueryMeta)


class KnowledgeBroker:
    """On-demand graph knowledge retriever with cache and circuit-breaker."""

    def __init__(
        self,
        *,
        cache_ttl_sec: float = 30.0,
        failure_threshold: int = 3,
        cooldown_sec: float = 60.0,
    ) -> None:
        self._cache_ttl_sec = max(1.0, cache_ttl_sec)
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_sec = max(1.0, cooldown_sec)
        self._cache: dict[str, tuple[float, KnowledgeQueryResult]] = {}
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _cache_key(self, query: KnowledgeQueryInput) -> str:
        return "|".join(
            [
                _extract_app_name_from_app_id(query.app_id),
                query.current_url.strip(),
                query.page_type.strip(),
                query.layout_fingerprint.strip(),
                query.recent_selector.strip(),
                query.recent_action.strip(),
                query.release_id.strip(),
                str(query.top_k),
            ]
        )

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._circuit_open_until = time.monotonic() + self._cooldown_sec

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _score_row(self, row: dict[str, Any], query: KnowledgeQueryInput) -> float:
        score = _as_float(row.get("confidence"), 0.0)
        source_url = _as_str(row.get("source_url")).strip()
        target_url = _as_str(row.get("target_url")).strip()
        selector = _as_str(row.get("selector")).strip().lower()
        action = _as_str(row.get("action")).strip().lower()
        current_url = query.current_url.strip()
        recent_selector = query.recent_selector.strip().lower()
        recent_action = query.recent_action.strip().lower()
        if current_url and (source_url == current_url or target_url == current_url):
            score += 1.0
        if recent_selector and recent_selector in selector:
            score += 0.8
        if recent_action and recent_action == action:
            score += 0.5
        intent = row.get("intent")
        if isinstance(intent, dict) and _as_str(intent.get("key")):
            score += 0.2
        return score

    def _build_result(
        self,
        rows: list[dict[str, Any]],
        query: KnowledgeQueryInput,
        source: str,
    ) -> KnowledgeQueryResult:
        scored = sorted(
            rows,
            key=lambda item: self._score_row(item, query),
            reverse=True,
        )
        top_rows = scored[: max(1, query.top_k)]
        transition_hints: list[dict[str, object]] = []
        state_hints: list[dict[str, object]] = []
        intent_hints: list[dict[str, object]] = []
        seen_states: set[str] = set()
        seen_intents: set[str] = set()
        for row in top_rows:
            selector = _as_str(row.get("selector"))
            action = _as_str(row.get("action"))
            confidence = _as_float(row.get("confidence"), 0.0)
            transition_hints.append(
                {
                    "transition_id": _as_str(row.get("id")),
                    "selector": selector,
                    "action": action,
                    "confidence": confidence,
                    "source_url": _as_str(row.get("source_url")),
                    "target_url": _as_str(row.get("target_url")),
                    "source": source,
                }
            )
            source_url = _as_str(row.get("source_url")).strip()
            target_url = _as_str(row.get("target_url")).strip()
            for state_url in (source_url, target_url):
                if not state_url or state_url in seen_states:
                    continue
                seen_states.add(state_url)
                state_hints.append({"url": state_url, "source": source})
            intent = row.get("intent")
            if isinstance(intent, dict):
                intent_key = _as_str(intent.get("key")).strip()
                if intent_key and intent_key not in seen_intents:
                    seen_intents.add(intent_key)
                    intent_hints.append(
                        {
                            "key": intent_key,
                            "summary": _as_str(intent.get("summary")),
                            "confidence": _as_float(intent.get("confidence"), 0.0),
                            "source": source,
                        }
                    )
        summary_parts: list[str] = []
        if transition_hints:
            top = transition_hints[0]
            summary_parts.append(
                f"historical {top.get('action')} on {top.get('selector')} (conf={_as_float(top.get('confidence'), 0.0):.2f})"
            )
        if intent_hints:
            summary_parts.append(f"intent hint: {intent_hints[0].get('key')}")
        if state_hints:
            summary_parts.append(f"state hint: {state_hints[0].get('url')}")
        return KnowledgeQueryResult(
            summary="; ".join(summary_parts),
            transition_hints=transition_hints,
            state_hints=state_hints,
            intent_hints=intent_hints,
            meta=KnowledgeQueryMeta(source=source),
        )

    async def _query_release_rows(
        self,
        manager: GraphManager,
        app_id: str,
        app_name: str,
        release_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not release_id:
            return []
        return await manager.get_knowledge_release_rows(
            app_id=app_id,
            app_name=app_name,
            release_id=release_id,
            limit=limit,
        )

    async def _query_legacy_rows(
        self, manager: GraphManager, app_id: str, app_name: str, limit: int
    ) -> list[dict[str, Any]]:
        return await manager.get_knowledge_legacy_rows(
            app_id=app_id,
            app_name=app_name,
            limit=limit,
        )

    async def query(
        self,
        query: KnowledgeQueryInput,
        *,
        timeout_ms: int = 1200,
    ) -> KnowledgeQueryResult:
        started = time.monotonic()
        if self._is_circuit_open():
            result = KnowledgeQueryResult()
            result.meta.circuit_open = True
            result.meta.source = "circuit_open"
            return result

        key = self._cache_key(query)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            cached_result = cached[1]
            out = KnowledgeQueryResult(
                summary=cached_result.summary,
                transition_hints=list(cached_result.transition_hints),
                state_hints=list(cached_result.state_hints),
                intent_hints=list(cached_result.intent_hints),
                meta=KnowledgeQueryMeta(
                    source=cached_result.meta.source,
                    cache_hit=True,
                    query_latency_ms=(time.monotonic() - started) * 1000,
                ),
            )
            return out

        app_id = (query.app_id or "").strip()
        app_name = _extract_app_name_from_app_id(app_id)
        limit = max(10, query.top_k * 4)
        try:
            async with GraphManager() as manager:
                rows: list[dict[str, Any]] = []
                source = "none"
                if query.release_id:
                    rows = await asyncio.wait_for(
                        self._query_release_rows(
                            manager, app_id, app_name, query.release_id, limit
                        ),
                        timeout=max(0.1, timeout_ms / 1000.0),
                    )
                    if rows:
                        source = "release"
                if not rows:
                    rows = await asyncio.wait_for(
                        self._query_legacy_rows(manager, app_id, app_name, limit),
                        timeout=max(0.1, timeout_ms / 1000.0),
                    )
                    if rows:
                        source = "legacy"
            result = self._build_result(rows, query, source=source)
            result.meta.query_latency_ms = (time.monotonic() - started) * 1000
            self._record_success()
            self._cache[key] = (time.monotonic() + self._cache_ttl_sec, result)
            return result
        except TimeoutError:
            self._record_failure()
            result = KnowledgeQueryResult()
            result.meta.timed_out = True
            result.meta.source = "timeout"
            result.meta.query_latency_ms = (time.monotonic() - started) * 1000
            return result
        except Exception as e:
            self._record_failure()
            result = KnowledgeQueryResult()
            result.meta.error = str(e)
            result.meta.source = "error"
            result.meta.query_latency_ms = (time.monotonic() - started) * 1000
            return result

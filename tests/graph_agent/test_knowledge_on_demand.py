from __future__ import annotations

import asyncio

import pytest

from graph_agent.cartography.config import resolve_knowledge_trigger_profile
from graph_agent.cartography.knowledge_broker import KnowledgeBroker, KnowledgeQueryInput
from graph_agent.cartography.mapping_pipeline import (
    build_knowledge_hint_text,
    compute_knowledge_trigger_score,
    should_query_knowledge,
)


def test_trigger_score_and_gate():
    score = compute_knowledge_trigger_score(
        low_layout_confidence_hits=3,
        failed_action_count=2,
        semantic_conflict_count=1,
        stuck_steps=2,
    )
    assert score >= 3.0
    assert should_query_knowledge(
        enabled=True,
        now_ts=100.0,
        last_query_ts=70.0,
        min_interval_sec=15.0,
        score=score,
        threshold=2.0,
    )
    assert not should_query_knowledge(
        enabled=True,
        now_ts=80.0,
        last_query_ts=70.0,
        min_interval_sec=15.0,
        score=score,
        threshold=2.0,
    )


def test_trigger_score_profiles():
    conservative = compute_knowledge_trigger_score(
        low_layout_confidence_hits=3,
        failed_action_count=3,
        semantic_conflict_count=2,
        stuck_steps=3,
        profile="conservative",
    )
    balanced = compute_knowledge_trigger_score(
        low_layout_confidence_hits=3,
        failed_action_count=3,
        semantic_conflict_count=2,
        stuck_steps=3,
        profile="balanced",
    )
    aggressive = compute_knowledge_trigger_score(
        low_layout_confidence_hits=3,
        failed_action_count=3,
        semantic_conflict_count=2,
        stuck_steps=3,
        profile="aggressive",
    )
    assert conservative < balanced < aggressive


def test_resolve_knowledge_trigger_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CARTOGRAPHY_KNOWLEDGE_TRIGGER_PROFILE", "aggressive")
    assert resolve_knowledge_trigger_profile() == "aggressive"
    monkeypatch.setenv("CARTOGRAPHY_KNOWLEDGE_TRIGGER_PROFILE", "invalid")
    assert resolve_knowledge_trigger_profile() == "balanced"


def test_build_knowledge_hint_text():
    text = build_knowledge_hint_text(
        "historical click",
        [
            {"action": "click", "selector": ".menu-orders", "confidence": 0.9},
        ],
    )
    assert "HISTORICAL_HINTS" in text
    assert ".menu-orders" in text


class _FakeManager:
    release_rows: list[dict[str, object]] = []
    legacy_rows: list[dict[str, object]] = []
    calls: list[str] = []
    kwargs_calls: list[dict[str, object]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get_knowledge_release_rows(self, **kwargs):
        _FakeManager.calls.append("release")
        _FakeManager.kwargs_calls.append(kwargs)
        return list(_FakeManager.release_rows)

    async def get_knowledge_legacy_rows(self, **kwargs):
        _FakeManager.calls.append("legacy")
        _FakeManager.kwargs_calls.append(kwargs)
        return list(_FakeManager.legacy_rows)


@pytest.mark.asyncio
async def test_broker_release_first_and_cache(monkeypatch: pytest.MonkeyPatch):
    _FakeManager.calls = []
    _FakeManager.kwargs_calls = []
    _FakeManager.release_rows = [
        {
            "id": "t:1",
            "confidence": 0.91,
            "selector": ".menu-orders",
            "action": "click",
            "source_url": "https://app/home",
            "target_url": "https://app/orders",
            "intent": {"key": "order.list.open", "summary": "open order list"},
        }
    ]
    _FakeManager.legacy_rows = []
    monkeypatch.setattr("graph_agent.cartography.knowledge_broker.GraphManager", _FakeManager)
    broker = KnowledgeBroker(cache_ttl_sec=60.0)
    query = KnowledgeQueryInput(
        app_id="app:demo:1",
        session_id="session:1",
        current_url="https://app/home",
        page_type="dashboard",
        release_id="release:1",
    )
    first = await broker.query(query, timeout_ms=500)
    second = await broker.query(query, timeout_ms=500)
    assert first.meta.source == "release"
    assert first.transition_hints
    assert second.meta.cache_hit is True
    assert len(_FakeManager.calls) == 1


@pytest.mark.asyncio
async def test_broker_fallback_to_legacy(monkeypatch: pytest.MonkeyPatch):
    _FakeManager.calls = []
    _FakeManager.kwargs_calls = []
    _FakeManager.release_rows = []
    _FakeManager.legacy_rows = [
        {
            "id": "t:2",
            "confidence": 0.75,
            "selector": ".list-row",
            "action": "click",
            "source_url": "https://app/orders",
            "target_url": "https://app/orders/1",
            "intent": {"key": "order.detail.open", "summary": "open detail"},
        }
    ]
    monkeypatch.setattr("graph_agent.cartography.knowledge_broker.GraphManager", _FakeManager)
    broker = KnowledgeBroker(cache_ttl_sec=1.0)
    result = await broker.query(
        KnowledgeQueryInput(
            app_id="app:demo:2",
            session_id="session:2",
            current_url="https://app/orders",
            page_type="list",
            release_id="release:missing",
        ),
        timeout_ms=500,
    )
    assert result.meta.source == "legacy"
    assert result.transition_hints
    assert _FakeManager.kwargs_calls, "expected broker query calls"
    first_kwargs = _FakeManager.kwargs_calls[0]
    assert first_kwargs.get("app_id") == "app:demo:2"


@pytest.mark.asyncio
async def test_broker_timeout_and_circuit_open(monkeypatch: pytest.MonkeyPatch):
    async def _slow_legacy(*args, **kwargs):
        await asyncio.sleep(0.2)
        return []

    monkeypatch.setattr("graph_agent.cartography.knowledge_broker.GraphManager", _FakeManager)
    broker = KnowledgeBroker(failure_threshold=1, cooldown_sec=60.0)
    monkeypatch.setattr(broker, "_query_legacy_rows", _slow_legacy)
    first = await broker.query(
        KnowledgeQueryInput(
            app_id="app:demo:3",
            session_id="session:3",
            current_url="https://app/x",
            page_type="detail",
        ),
        timeout_ms=1,
    )
    second = await broker.query(
        KnowledgeQueryInput(
            app_id="app:demo:3",
            session_id="session:3",
            current_url="https://app/x",
            page_type="detail",
        ),
        timeout_ms=1,
    )
    assert first.meta.timed_out is True
    assert second.meta.circuit_open is True

from __future__ import annotations

from graph_agent.cartography.mapping_pipeline import (
    _classify_pipeline_exception,
    _deserialize_skip_decision,
    _looks_rate_limited,
    _serialize_skip_decision,
)
from graph_agent.cartography.skip_advisor import SkipDecision, SkipKind


def test_skip_decision_round_trip_serialization():
    decision = SkipDecision(
        kind=SkipKind.EXPLORE_ZONES_ONLY,
        reason="test",
        confidence=0.8,
        target_zone_selectors=[".zone-a"],
        coverage=0.66,
        state_count=3,
        release_coverage=0.7,
        intent_confirm_total=4,
        entity_confirm_total=2,
        intent_confirmed_zone_count=1,
    )
    raw = _serialize_skip_decision(decision)
    restored = _deserialize_skip_decision(raw)
    assert restored is not None
    assert restored.kind is SkipKind.EXPLORE_ZONES_ONLY
    assert restored.target_zone_selectors == [".zone-a"]
    assert restored.intent_confirm_total == 4


def test_classify_pipeline_exception():
    assert _classify_pipeline_exception(Exception("HTTP 429 Too Many Requests")) == "rate_limited"
    assert _classify_pipeline_exception(Exception("403 forbidden")) == "unauthorized"
    assert _classify_pipeline_exception(Exception("navigation timeout")) == "timeout"
    assert _classify_pipeline_exception(Exception("other")) == "unknown"


def test_looks_rate_limited_signal():
    assert _looks_rate_limited("Too many requests, please retry", "https://demo")
    assert _looks_rate_limited("", "https://demo/path?error=429")
    assert not _looks_rate_limited("normal dashboard page", "https://demo")

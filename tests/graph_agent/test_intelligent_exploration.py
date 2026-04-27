from __future__ import annotations

from graph_agent.cartography.react_explorer import _build_state_identity
from graph_agent.cartography.runner import rank_warm_start_candidates
from graph_agent.playback.engine import summarize_transition_evidence


def test_build_state_identity_is_stable_for_same_route_and_view():
    state_id_a = _build_state_identity(
        url="https://example.com/app#/orders/list?tenant=A",
        spa_route="/orders/list",
        view_fingerprint="vf-001",
    )
    state_id_b = _build_state_identity(
        url="https://example.com/app#/orders/list?tenant=B",
        spa_route="/orders/list",
        view_fingerprint="vf-001",
    )
    assert state_id_a == state_id_b


def test_rank_warm_start_candidates_prefers_high_confidence_and_unexplored():
    candidates = [
        {"transition_id": "t-low", "confidence": 0.6, "zone_unexplored": False},
        {"transition_id": "t-high", "confidence": 0.9, "zone_unexplored": False},
        {"transition_id": "t-mid-unexplored", "confidence": 0.7, "zone_unexplored": True},
    ]
    ranked = rank_warm_start_candidates(candidates)
    assert [item["transition_id"] for item in ranked] == [
        "t-mid-unexplored",
        "t-high",
        "t-low",
    ]


def test_summarize_transition_evidence_compacts_types_and_counts():
    summary = summarize_transition_evidence(
        [
            {"type": "dom_diff"},
            {"type": "network"},
            {"type": "network"},
            {"type": "console"},
        ]
    )
    assert "dom_diff:1" in summary
    assert "network:2" in summary
    assert "console:1" in summary

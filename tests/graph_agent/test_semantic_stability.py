from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph_agent.cartography import persistence


def _result_fixture() -> object:
    states = [
        SimpleNamespace(
            id="s:users",
            url="https://demo.local/users#top",
            title="Users",
            spa_route="/users",
        ),
        SimpleNamespace(
            id="s:new-user",
            url="https://demo.local/users/new",
            title="Create User",
            spa_route="/users/new",
        ),
    ]
    transitions = [
        SimpleNamespace(
            from_state_id="s:users",
            to_state_id="s:new-user",
            action="click",
            semantic_action_key="open_create_form",
            selector=".action-bar .create",
            intent=SimpleNamespace(key="create_user"),
        )
    ]
    return SimpleNamespace(states=states, transitions=transitions)


def test_calculate_semantic_stability_bootstrap_mode():
    metrics = persistence._build_semantic_metrics(_result_fixture())
    result = persistence._calculate_semantic_stability(
        semantic_metrics=metrics,
        baseline_row={},
    )
    assert result["semantic_stability_mode"] == "bootstrap"
    assert result["semantic_stability_score"] == pytest.approx(100.0)
    assert result["semantic_stability_passed"] is True


def test_calculate_semantic_stability_compare_mode():
    metrics = persistence._build_semantic_metrics(_result_fixture())
    baseline = {
        "session_id": "session:baseline:1",
        "semantic_state_keys_json": '["route:/users","route:/users/edit"]',
        "semantic_transition_keys_json": (
            '["route:/users|route:/users/edit|click|open_edit_form"]'
        ),
        "semantic_intent_keys_json": '["create_user"]',
    }
    result = persistence._calculate_semantic_stability(
        semantic_metrics=metrics,
        baseline_row=baseline,
        threshold=90.0,
    )
    assert result["semantic_stability_mode"] == "compare"
    assert result["semantic_baseline_session_id"] == "session:baseline:1"
    assert result["semantic_stability_score"] == pytest.approx(33.33)
    assert result["semantic_stability_passed"] is False

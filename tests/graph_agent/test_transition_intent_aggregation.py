"""Tests for Transition intent-level aggregation, id format, and param_name inference."""

from __future__ import annotations

import json

import pytest

from graph_agent.cartography.react_explorer import (
    _build_state_identity,
    _extract_spa_route,
    _infer_param_name_from_snapshot,
    _should_commit_transition,
)
from graph_agent.models import ActionType, Transition, TransitionStep


# ---------------------------------------------------------------------------
# _should_commit_transition
# ---------------------------------------------------------------------------


class TestShouldCommitTransition:
    def test_url_change_commits(self) -> None:
        assert _should_commit_transition(
            "https://a.com/login", "https://a.com/home", "click"
        ) is True

    def test_same_url_click_commits(self) -> None:
        assert _should_commit_transition(
            "https://a.com/login", "https://a.com/login", "click"
        ) is True

    def test_same_url_input_does_not_commit(self) -> None:
        assert _should_commit_transition(
            "https://a.com/login", "https://a.com/login", "input"
        ) is False

    def test_same_url_select_does_not_commit(self) -> None:
        assert _should_commit_transition(
            "https://a.com/login", "https://a.com/login", "select_dropdown"
        ) is False

    def test_spa_route_change_commits(self) -> None:
        assert _should_commit_transition(
            "https://a.com/#/login", "https://a.com/#/home", "input"
        ) is True


# ---------------------------------------------------------------------------
# _infer_param_name_from_snapshot
# ---------------------------------------------------------------------------


class TestInferParamNameFromSnapshot:
    def test_name_attribute(self) -> None:
        snap = json.dumps({"name": "username", "id": "user-input"})
        assert _infer_param_name_from_snapshot(snap) == "username"

    def test_id_attribute(self) -> None:
        snap = json.dumps({"id": "password-field"})
        assert _infer_param_name_from_snapshot(snap) == "password-field"

    def test_placeholder_attribute(self) -> None:
        snap = json.dumps({"placeholder": "Enter your email"})
        assert _infer_param_name_from_snapshot(snap) == "Enter your email"

    def test_nested_attributes_type_password(self) -> None:
        snap = json.dumps({"attributes": {"type": "password"}})
        assert _infer_param_name_from_snapshot(snap) == "password"

    def test_nested_attributes_name_over_id(self) -> None:
        snap = json.dumps({"attributes": {"name": "email", "id": "input-0"}})
        assert _infer_param_name_from_snapshot(snap) == "email"

    def test_none_input(self) -> None:
        assert _infer_param_name_from_snapshot(None) is None

    def test_empty_string(self) -> None:
        assert _infer_param_name_from_snapshot("") is None

    def test_invalid_json(self) -> None:
        assert _infer_param_name_from_snapshot("not json") is None

    def test_no_relevant_attributes(self) -> None:
        snap = json.dumps({"attributes": {"class": "form-control"}})
        assert _infer_param_name_from_snapshot(snap) is None

    def test_type_email(self) -> None:
        snap = json.dumps({"attributes": {"type": "email"}})
        assert _infer_param_name_from_snapshot(snap) == "email"


# ---------------------------------------------------------------------------
# Transition id format
# ---------------------------------------------------------------------------


class TestTransitionIdFormat:
    def test_state_id_uses_digest_not_url(self) -> None:
        state_id = _build_state_identity(
            "http://172.20.22.202:2209/mis-ui/login.html",
            _extract_spa_route("http://172.20.22.202:2209/mis-ui/login.html"),
            "fake-fp",
        )
        assert state_id.startswith("state:")
        # Should NOT contain the raw URL
        assert "172.20.22.202" not in state_id
        assert "mis-ui" not in state_id

    def test_state_id_deterministic(self) -> None:
        url = "http://172.20.22.202:2209/mis-ui/login.html"
        route = _extract_spa_route(url)
        fp = "abc123"
        id1 = _build_state_identity(url, route, fp)
        id2 = _build_state_identity(url, route, fp)
        assert id1 == id2

    def test_different_url_different_id(self) -> None:
        id1 = _build_state_identity("http://a.com/login", "/login", "fp")
        id2 = _build_state_identity("http://a.com/home", "/home", "fp")
        assert id1 != id2


# ---------------------------------------------------------------------------
# TransitionStep model
# ---------------------------------------------------------------------------


class TestTransitionStepModel:
    def test_default_values(self) -> None:
        step = TransitionStep()
        assert step.action == ActionType.CLICK
        assert step.selector == ""
        assert step.selector_chain == []
        assert step.param_name is None
        assert step.action_value is None
        assert step.element_snapshot is None
        assert step.thought is None
        assert step.step_index is None
        assert step.semantic_action_key is None

    def test_full_construction(self) -> None:
        step = TransitionStep(
            action=ActionType.FILL,
            selector="#username",
            selector_chain=["#username", "[name='username']"],
            param_name="username",
            action_value="admin",
            element_snapshot='{"name": "username"}',
            thought="Fill username field",
            step_index=0,
            semantic_action_key="fill:#username",
        )
        assert step.action == ActionType.FILL
        assert step.param_name == "username"
        assert step.action_value == "admin"


# ---------------------------------------------------------------------------
# Transition with steps field
# ---------------------------------------------------------------------------


class TestTransitionWithSteps:
    def test_default_steps_empty(self) -> None:
        t = Transition(id="t:abc:click-0")
        assert t.steps == []

    def test_steps_populated(self) -> None:
        steps = [
            TransitionStep(action=ActionType.FILL, selector="#user", step_index=0),
            TransitionStep(action=ActionType.FILL, selector="#pass", step_index=1),
            TransitionStep(action=ActionType.CLICK, selector="#login", step_index=2),
        ]
        t = Transition(
            id="t:abc:auth.login",
            steps=steps,
            from_state_id="state:login",
            to_state_id="state:home",
        )
        assert len(t.steps) == 3
        assert t.steps[0].action == ActionType.FILL
        assert t.steps[2].action == ActionType.CLICK

    def test_steps_serialization(self) -> None:
        steps = [
            TransitionStep(action=ActionType.FILL, selector="#user", param_name="username"),
        ]
        t = Transition(id="t:abc:test", steps=steps)
        data = t.model_dump(mode="json")
        assert "steps" in data
        assert len(data["steps"]) == 1
        assert data["steps"][0]["action"] == "fill"
        assert data["steps"][0]["param_name"] == "username"

    def test_backward_compat_no_steps(self) -> None:
        t = Transition(id="t:old:click-0", action=ActionType.CLICK)
        data = t.model_dump(mode="json")
        assert data["steps"] == []


# ---------------------------------------------------------------------------
# _extract_spa_route
# ---------------------------------------------------------------------------


class TestExtractSpaRoute:
    def test_hash_route(self) -> None:
        assert _extract_spa_route("https://a.com/#/login") == "/login"

    def test_no_hash(self) -> None:
        assert _extract_spa_route("https://a.com/login") == "/login"

    def test_root(self) -> None:
        result = _extract_spa_route("https://a.com/")
        assert result == "/" or result == ""

    def test_empty_string(self) -> None:
        result = _extract_spa_route("")
        assert result == "/" or result == ""

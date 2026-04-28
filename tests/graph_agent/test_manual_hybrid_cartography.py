from __future__ import annotations

import pytest

from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    SemanticTransitionPatch,
    SemanticTransitionResult,
    infer_transition_semantics,
)
from graph_agent.cartography.intervention_queue import evaluate_intervention_need
import graph_agent.cartography.manual_capture as manual_capture_module
from graph_agent.cartography.manual_capture import ManualCaptureSession
from graph_agent.cartography.transition_normalizer import (
    NormalizeDefaults,
    candidate_to_transition,
    normalize_candidates,
)
from graph_agent.models import (
    ActionType,
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointOrigin,
    CheckpointTiming,
    Severity,
    TransitionSourceType,
)


def test_normalize_candidates_apply_defaults():
    normalized = normalize_candidates(
        [
            {
                "action": "click",
                "selector": "button.save",
                "url_before": "https://a/before",
                "url_after": "https://a/after",
            }
        ],
        defaults=NormalizeDefaults(
            source_type=TransitionSourceType.MANUAL_RAW,
            operator_id="human:qa",
            session_id="session:manual:1",
            trace_id="trace-1",
        ),
    )
    item = normalized[0]
    assert item["source_type"] == "manual_raw"
    assert item["operator_id"] == "human:qa"
    assert item["session_id"] == "session:manual:1"
    assert item["trace_id"] == "trace-1"


def test_candidate_to_transition_uses_source_and_operator():
    transition = candidate_to_transition(
        {
            "source_type": "manual_graph_assisted",
            "operator_id": "human:reviewer",
            "session_id": "session:x",
            "trace_id": "trace:x",
            "step_index": 1,
            "action": "fill",
            "selector": "input[name='name']",
            "url_before": "https://app/a",
            "url_after": "https://app/b",
        },
        fallback_session_id="session:x",
    )
    assert transition.action == ActionType.FILL
    assert transition.source_type == TransitionSourceType.MANUAL_GRAPH_ASSISTED
    assert transition.operator_id == "human:reviewer"


@pytest.mark.asyncio
async def test_manual_capture_session_builds_manual_result(monkeypatch: pytest.MonkeyPatch):
    async def _fake_infer(_input: SemanticInferenceInput) -> SemanticTransitionResult:
        return SemanticTransitionResult(
            transition_patch=SemanticTransitionPatch(
                intent=None,
                intent_failure_reason=None,
                selector_chain=[".menu-orders"],
                semantic_action_key="click:.menu-orders",
                confidence_hint=0.8,
            ),
            checkpoints=[
                Checkpoint(
                    id="cp:semantic:1",
                    layer=CheckpointLayer.SEMANTIC,
                    timing=CheckpointTiming.AFTER,
                    expect=CheckpointExpect.SHOULD_PASS,
                    severity=Severity.INFO,
                    rule_type="semantic_consistency",
                    description="semantic ok",
                    origin_type=CheckpointOrigin.MANUAL,
                )
            ],
            conflict_flags=[],
            debug_reason="ok",
        )

    monkeypatch.setattr(manual_capture_module, "infer_transition_semantics", _fake_infer)

    session = ManualCaptureSession.from_raw(
        session_id="session:manual:test",
        operator_id="human:qa",
        start_url="https://app/start",
    )
    session.record_action(
        action="click",
        selector=".menu-orders",
        url_before="https://app/start",
        url_after="https://app/orders",
        thought="Open orders menu",
    )
    result = await session.to_cartography_result()
    assert len(result.transitions) == 1
    assert len(result.states) == 2
    assert result.transitions[0].source_type == TransitionSourceType.MANUAL_RAW
    assert result.checkpoints[0].origin_type.value == "manual"


def test_intervention_trigger_generates_tasks():
    tasks = evaluate_intervention_need(
        session_id="session:1",
        source_url="https://app",
        page_type="dashboard",
        low_layout_confidence_hits=3,
        failed_action_count=4,
        semantic_conflict_count=2,
        has_cross_origin=True,
        has_iframe=True,
        has_captcha=True,
    )
    reasons = {task.reason for task in tasks}
    assert "low_layout_confidence" in reasons
    assert "high_action_failure_rate" in reasons
    assert "semantic_conflict" in reasons


@pytest.mark.asyncio
async def test_inference_core_consistent_semantic_key_for_auto_and_manual(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_distill(**kwargs) -> str:
        return str(kwargs.get("thought_text") or "")

    async def _fake_infer(
        **_kwargs,
    ):
        return None, "low_confidence:0.10", "full"

    monkeypatch.setattr("graph_agent.cartography.inference_core.distill_ui_thought", _fake_distill)
    monkeypatch.setattr("graph_agent.cartography.inference_core.infer_intent_progressive", _fake_infer)

    common = dict(
        action=ActionType.CLICK,
        selector=".btn-save",
        selector_chain_hint=[".btn-save"],
        source_url="https://app/a",
        target_url="https://app/b",
        step_index=1,
        transition_id="t:1",
    )
    auto_result = await infer_transition_semantics(
        SemanticInferenceInput(source_type="auto", operator_id="agent", **common)
    )
    manual_result = await infer_transition_semantics(
        SemanticInferenceInput(
            source_type="manual_raw",
            operator_id="human:qa",
            **common,
        )
    )
    assert (
        auto_result.transition_patch.semantic_action_key
        == manual_result.transition_patch.semantic_action_key
    )


@pytest.mark.asyncio
async def test_inference_core_emits_conflict_flags(monkeypatch: pytest.MonkeyPatch):
    async def _fake_distill(**kwargs) -> str:
        return str(kwargs.get("thought_text") or "")

    async def _fake_infer(
        **_kwargs,
    ):
        return None, "semantic_conflict:action_intent_mismatch", "full"

    monkeypatch.setattr("graph_agent.cartography.inference_core.distill_ui_thought", _fake_distill)
    monkeypatch.setattr("graph_agent.cartography.inference_core.infer_intent_progressive", _fake_infer)

    result = await infer_transition_semantics(
        SemanticInferenceInput(
            source_type="manual_graph_assisted",
            operator_id="human:qa",
            action=ActionType.CLICK,
            selector=".menu-item",
            selector_chain_hint=[".menu-item"],
            source_url="https://app/home",
            target_url="https://app/orders",
            transition_id="t:manual:1",
            step_index=3,
            existing_semantic_keys={"click:.menu-item"},
        )
    )
    assert "action_intent_mismatch" in result.conflict_flags
    assert "duplicate_semantic_key" in result.conflict_flags

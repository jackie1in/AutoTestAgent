from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from graph_agent.intent.parser import distill_ui_thought, infer_intent_progressive
from graph_agent.models import (
    ActionType,
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointOrigin,
    CheckpointTiming,
    Intent,
    Severity,
)


SemanticConflictFlag = Literal[
    "action_intent_mismatch",
    "low_confidence",
    "duplicate_semantic_key",
]


@dataclass(slots=True)
class SemanticInferenceInput:
    source_type: str
    operator_id: str
    action: ActionType | str
    selector: str
    selector_chain_hint: list[str] = field(default_factory=list)
    source_url: str = ""
    target_url: str = ""
    param_name: str | None = None
    thought_text: str = ""
    neighbor_steps: list[dict[str, str]] | None = None
    page_signals: dict[str, str] | None = None
    evidence_bundle: list[dict[str, object]] = field(default_factory=list)
    from_state_id: str = ""
    to_state_id: str = ""
    step_index: int = 0
    transition_id: str = ""
    existing_semantic_keys: set[str] | None = None


@dataclass(slots=True)
class SemanticTransitionPatch:
    intent: Intent | None
    intent_failure_reason: str | None
    selector_chain: list[str]
    semantic_action_key: str
    confidence_hint: float


@dataclass(slots=True)
class SemanticTransitionResult:
    transition_patch: SemanticTransitionPatch
    checkpoints: list[Checkpoint]
    conflict_flags: list[SemanticConflictFlag]
    debug_reason: str = ""


def _normalize_action(action: ActionType | str) -> ActionType:
    if isinstance(action, ActionType):
        return action
    raw = (action or "").strip().lower()
    if raw == ActionType.CLICK.value:
        return ActionType.CLICK
    if raw == ActionType.FILL.value:
        return ActionType.FILL
    if raw == ActionType.SELECT.value:
        return ActionType.SELECT
    if raw == ActionType.RICH_TEXT.value:
        return ActionType.RICH_TEXT
    if raw == ActionType.NAVIGATE.value:
        return ActionType.NAVIGATE
    return ActionType.UNKNOWN


def _resolve_checkpoint_origin(source_type: str) -> CheckpointOrigin:
    if (source_type or "").strip().lower().startswith("manual"):
        return CheckpointOrigin.MANUAL
    return CheckpointOrigin.INFERRED


def _resolve_conflict_flags(reason: str | None) -> list[SemanticConflictFlag]:
    if not reason:
        return []
    raw = reason.lower()
    flags: list[SemanticConflictFlag] = []
    if "semantic_conflict" in raw:
        flags.append("action_intent_mismatch")
    if "low_confidence" in raw or "intent_inference_failed" in raw:
        flags.append("low_confidence")
    return flags


async def infer_transition_semantics(
    input: SemanticInferenceInput,
) -> SemanticTransitionResult:
    action = _normalize_action(input.action)
    selector_chain = list(input.selector_chain_hint or [])
    selector = (input.selector or "").strip()
    if not selector_chain:
        selector_chain = [selector or "[unknown]"]
    semantic_action_key = f"{action.value}:{selector_chain[0]}"

    distilled_thought = await distill_ui_thought(
        thought_text=input.thought_text or "",
        action=action,
        selector=selector_chain[0],
        source_url=input.source_url,
        target_url=input.target_url,
    )
    intent, reason, _ = await infer_intent_progressive(
        action=action,
        selector=selector_chain[0],
        source_url=input.source_url,
        target_url=input.target_url,
        param_name=input.param_name,
        thought_text=distilled_thought,
        neighbor_steps=input.neighbor_steps,
        page_signals=input.page_signals,
    )
    conflict_flags = _resolve_conflict_flags(reason)
    if input.existing_semantic_keys and semantic_action_key in input.existing_semantic_keys:
        conflict_flags.append("duplicate_semantic_key")

    confidence_hint = 0.5
    if intent is not None and intent.confidence is not None:
        confidence_hint = max(0.1, min(1.0, float(intent.confidence)))
    elif "low_confidence" in conflict_flags:
        confidence_hint = 0.3

    transition_patch = SemanticTransitionPatch(
        intent=intent,
        intent_failure_reason=reason,
        selector_chain=selector_chain,
        semantic_action_key=semantic_action_key,
        confidence_hint=confidence_hint,
    )
    cp_origin = _resolve_checkpoint_origin(input.source_type)
    cp_rule_type = "semantic_consistency" if not conflict_flags else "semantic_conflict"
    cp_desc = (
        f"Semantic inference for {action.value} on {selector_chain[0]}"
        if not conflict_flags
        else f"Semantic conflict detected ({','.join(conflict_flags)}) for {action.value} on {selector_chain[0]}"
    )
    transition_ref = input.transition_id or f"transition:{input.step_index}:{semantic_action_key}"
    checkpoint = Checkpoint(
        id=f"cp:{transition_ref}:semantic",
        layer=CheckpointLayer.SEMANTIC,
        timing=CheckpointTiming.AFTER,
        expect=CheckpointExpect.SHOULD_PASS,
        severity=Severity.MAJOR if conflict_flags else Severity.INFO,
        rule_type=cp_rule_type,
        description=cp_desc,
        origin_type=cp_origin,
    )
    return SemanticTransitionResult(
        transition_patch=transition_patch,
        checkpoints=[checkpoint],
        conflict_flags=conflict_flags,
        debug_reason=reason or "ok",
    )

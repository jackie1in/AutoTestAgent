from __future__ import annotations

import hashlib
from dataclasses import dataclass

from graph_agent.cartography.types import TransitionCandidate
from graph_agent.models import ActionType, Transition, TransitionSourceType


@dataclass(slots=True)
class NormalizeDefaults:
    source_type: TransitionSourceType = TransitionSourceType.AUTO
    operator_id: str = "agent"
    session_id: str = ""
    trace_id: str = ""


def _normalize_action(action: str) -> ActionType:
    raw = (action or "").strip().lower()
    if raw == ActionType.CLICK.value:
        return ActionType.CLICK
    if raw == ActionType.FILL.value:
        return ActionType.FILL
    if raw == ActionType.SELECT.value:
        return ActionType.SELECT
    if raw == ActionType.NAVIGATE.value:
        return ActionType.NAVIGATE
    if raw == ActionType.RICH_TEXT.value:
        return ActionType.RICH_TEXT
    return ActionType.UNKNOWN


def _normalize_source_type(source_type: str, default: TransitionSourceType) -> TransitionSourceType:
    raw = (source_type or "").strip().lower()
    if raw == TransitionSourceType.MANUAL_GRAPH_ASSISTED.value:
        return TransitionSourceType.MANUAL_GRAPH_ASSISTED
    if raw == TransitionSourceType.MANUAL_RAW.value:
        return TransitionSourceType.MANUAL_RAW
    if raw == TransitionSourceType.AUTO.value:
        return TransitionSourceType.AUTO
    return default


def normalize_candidates(
    candidates: list[TransitionCandidate],
    *,
    defaults: NormalizeDefaults | None = None,
) -> list[TransitionCandidate]:
    applied = defaults or NormalizeDefaults()
    normalized: list[TransitionCandidate] = []
    for idx, item in enumerate(candidates):
        source = _normalize_source_type(
            str(item.get("source_type") or ""),
            applied.source_type,
        )
        step_index = int(item.get("step_index") or idx)
        normalized.append(
            {
                **item,
                "source_type": source.value,
                "operator_id": str(item.get("operator_id") or applied.operator_id),
                "session_id": str(item.get("session_id") or applied.session_id),
                "trace_id": str(item.get("trace_id") or applied.trace_id),
                "step_index": step_index,
                "action": str(item.get("action") or ActionType.UNKNOWN.value).strip().lower(),
                "selector": str(item.get("selector") or "").strip(),
                "url_before": str(item.get("url_before") or "").strip(),
                "url_after": str(item.get("url_after") or "").strip(),
                "confidence_hint": float(item.get("confidence_hint") or 0.5),
                "evidence_bundle": list(item.get("evidence_bundle") or []),
            }
        )
    return normalized


def candidate_to_transition(
    candidate: TransitionCandidate,
    *,
    fallback_session_id: str,
) -> Transition:
    source_type = _normalize_source_type(
        str(candidate.get("source_type") or ""),
        TransitionSourceType.AUTO,
    )
    from_hint = str(candidate.get("from_state_hint") or candidate.get("url_before") or "")
    to_hint = str(candidate.get("to_state_hint") or candidate.get("url_after") or "")
    selector = str(candidate.get("selector") or "[manual]")
    action = _normalize_action(str(candidate.get("action") or "unknown"))
    step_index = int(candidate.get("step_index") or 0)
    trace_id = str(candidate.get("trace_id") or fallback_session_id or "manual")
    transition_seed = f"{trace_id}|{step_index}|{action.value}|{selector}|{from_hint}|{to_hint}"
    transition_id = f"t:normalized:{hashlib.md5(transition_seed.encode(), usedforsecurity=False).hexdigest()[:16]}"
    confidence = float(candidate.get("confidence_hint") or 0.5)
    return Transition(
        id=transition_id,
        selector=selector,
        action=action,
        action_value=str(candidate.get("action_value") or "") or None,
        param_name=str(candidate.get("param_name") or "") or None,
        thought=str(candidate.get("thought") or "") or None,
        confidence=max(0.1, min(1.0, confidence)),
        session_id=str(candidate.get("session_id") or fallback_session_id),
        step_index=step_index,
        from_state_id=from_hint or None,
        to_state_id=to_hint or None,
        source_type=source_type,
        operator_id=str(candidate.get("operator_id") or "agent"),
    )

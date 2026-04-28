from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from graph_agent.cartography.transition_normalizer import (
    NormalizeDefaults,
    candidate_to_transition,
    normalize_candidates,
)
from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    infer_transition_semantics,
)
from graph_agent.cartography.types import (
    EvidenceBundleItem,
    TransitionCandidate,
    TransitionSourceType as CaptureTransitionSourceType,
)
from graph_agent.graph.merger import CartographyResult
from graph_agent.models import (
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointOrigin,
    CheckpointTiming,
    Severity,
    State,
    TransitionSourceType,
)


def _state_id_from_hint(value: str) -> str:
    cleaned = (value or "").strip()
    if cleaned.startswith("state:"):
        return cleaned
    digest = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:16]
    return f"state:manual:{digest}"


def _state_title(url_or_hint: str) -> str:
    raw = (url_or_hint or "").strip()
    if not raw:
        return "Manual state"
    return raw[:120]


@dataclass(slots=True)
class ManualCaptureSession:
    session_id: str
    source_type: CaptureTransitionSourceType
    operator_id: str
    trace_id: str = ""
    app_id: str = ""
    start_state_hint: str = ""
    start_url: str = ""
    _events: list[TransitionCandidate] = field(default_factory=list)

    @classmethod
    def from_graph_context(
        cls,
        *,
        session_id: str,
        operator_id: str,
        app_id: str,
        start_state_hint: str,
        start_url: str = "",
        trace_id: str = "",
    ) -> "ManualCaptureSession":
        return cls(
            session_id=session_id,
            source_type="manual_graph_assisted",
            operator_id=operator_id,
            app_id=app_id,
            start_state_hint=start_state_hint,
            start_url=start_url,
            trace_id=trace_id,
        )

    @classmethod
    def from_raw(
        cls,
        *,
        session_id: str,
        operator_id: str,
        start_url: str = "",
        trace_id: str = "",
    ) -> "ManualCaptureSession":
        return cls(
            session_id=session_id,
            source_type="manual_raw",
            operator_id=operator_id,
            start_url=start_url,
            trace_id=trace_id,
        )

    def record_action(
        self,
        *,
        action: str,
        selector: str,
        url_before: str,
        url_after: str,
        thought: str = "",
        action_value: str = "",
        param_name: str = "",
        confidence_hint: float = 0.9,
        evidence_bundle: list[EvidenceBundleItem] | None = None,
        from_state_hint: str = "",
        to_state_hint: str = "",
    ) -> None:
        step_index = len(self._events)
        evidence_items: list[EvidenceBundleItem] = evidence_bundle or []
        candidate: TransitionCandidate = {
            "source_type": self.source_type,
            "operator_id": self.operator_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id or self.session_id,
            "step_index": step_index,
            "action": action,
            "selector": selector,
            "action_value": action_value,
            "param_name": param_name,
            "url_before": url_before,
            "url_after": url_after,
            "from_state_hint": from_state_hint or self.start_state_hint or url_before,
            "to_state_hint": to_state_hint or url_after,
            "thought": thought,
            "confidence_hint": confidence_hint,
            "evidence_bundle": evidence_items,
        }
        self._events.append(candidate)

    def export_candidates(self) -> list[TransitionCandidate]:
        source = TransitionSourceType.AUTO
        if self.source_type == TransitionSourceType.MANUAL_GRAPH_ASSISTED.value:
            source = TransitionSourceType.MANUAL_GRAPH_ASSISTED
        elif self.source_type == TransitionSourceType.MANUAL_RAW.value:
            source = TransitionSourceType.MANUAL_RAW
        return normalize_candidates(
            self._events,
            defaults=NormalizeDefaults(
                source_type=source,
                operator_id=self.operator_id,
                session_id=self.session_id,
                trace_id=self.trace_id or self.session_id,
            ),
        )

    async def to_cartography_result(self) -> CartographyResult:
        normalized = self.export_candidates()
        result = CartographyResult()
        seen_states: set[str] = set()
        semantic_keys: set[str] = set()
        semantic_conflicts = 0
        for candidate in normalized:
            transition = candidate_to_transition(
                candidate,
                fallback_session_id=self.session_id,
            )
            from_hint = str(candidate.get("from_state_hint") or candidate.get("url_before") or "")
            to_hint = str(candidate.get("to_state_hint") or candidate.get("url_after") or "")
            from_state = State(
                id=_state_id_from_hint(from_hint),
                url=str(candidate.get("url_before") or from_hint),
                title=_state_title(from_hint),
            )
            to_state = State(
                id=_state_id_from_hint(to_hint),
                url=str(candidate.get("url_after") or to_hint),
                title=_state_title(to_hint),
            )
            transition.from_state_id = from_state.id
            transition.to_state_id = to_state.id
            if from_state.id not in seen_states:
                result.states.append(from_state)
                seen_states.add(from_state.id)
            if to_state.id not in seen_states:
                result.states.append(to_state)
                seen_states.add(to_state.id)
            semantic = await infer_transition_semantics(
                SemanticInferenceInput(
                    source_type=self.source_type,
                    operator_id=self.operator_id,
                    action=transition.action,
                    selector=transition.selector,
                    selector_chain_hint=transition.selector_chain,
                    source_url=from_state.url,
                    target_url=to_state.url,
                    param_name=transition.param_name,
                    thought_text=transition.thought or "",
                    from_state_id=from_state.id,
                    to_state_id=to_state.id,
                    step_index=int(transition.step_index or 0),
                    transition_id=transition.id,
                    existing_semantic_keys=semantic_keys,
                )
            )
            transition.intent = semantic.transition_patch.intent
            transition.intent_failure_reason = semantic.transition_patch.intent_failure_reason
            transition.selector_chain = semantic.transition_patch.selector_chain
            transition.semantic_action_key = semantic.transition_patch.semantic_action_key
            transition.confidence = semantic.transition_patch.confidence_hint
            semantic_keys.add(transition.semantic_action_key or "")
            if semantic.conflict_flags:
                semantic_conflicts += 1
            result.transitions.append(transition)
            cp = Checkpoint(
                id=f"cp:{transition.id}:manual-after",
                layer=CheckpointLayer.SEMANTIC,
                timing=CheckpointTiming.AFTER,
                expect=CheckpointExpect.SHOULD_PASS,
                severity=Severity.MAJOR,
                rule_type="manual_action_observed",
                description=(
                    "Manual recorded transition "
                    f"{transition.action.value} on "
                    f"{transition.selector}"
                ),
                origin_type=CheckpointOrigin.MANUAL,
                session_id=self.session_id,
            )
            result.checkpoints.append(cp)
            result.checkpoints.extend(semantic.checkpoints)
            result.checkpoint_transition_map[cp.id] = transition.id
            for extra in semantic.checkpoints:
                result.checkpoint_transition_map[extra.id] = transition.id
        result.semantic_conflict_count = semantic_conflicts
        return result

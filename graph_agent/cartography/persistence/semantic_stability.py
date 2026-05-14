from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from graph_agent.cartography.config import clean_url

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _empty_stats() -> dict[str, int]:
    return {
        "states_added": 0,
        "transitions_added": 0,
        "filtered_non_ui_edges": 0,
        "semantic_mismatch_warnings": 0,
        "url_discontinuity_warnings": 0,
        "frame_context_transition_warnings": 0,
        "manual_transition_count": 0,
        "auto_transition_count": 0,
    }


def _as_str(value: object) -> str:
    return str(value or "")


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _source_priority(source_type: object) -> int:
    value = str(source_type or "auto").strip().lower()
    if value == "manual_graph_assisted":
        return 3
    if value == "manual_raw":
        return 2
    return 1


def _transition_stable_key(transition: object) -> str:
    from_id = _as_str(getattr(transition, "from_state_id", "")).strip()
    to_id = _as_str(getattr(transition, "to_state_id", "")).strip()
    action = _as_str(getattr(transition, "action", "")).strip().lower()
    semantic = _as_str(getattr(transition, "semantic_action_key", "")).strip()
    if not semantic:
        semantic = _as_str(getattr(transition, "selector", "")).strip()
    # Include steps digest for intent-level transitions
    steps = getattr(transition, "steps", None)
    steps_digest = ""
    if steps:
        steps_summary = "|".join(
            f"{getattr(s, 'action', '')}:{getattr(s, 'selector', '')}"
            for s in steps
        )
        steps_digest = hashlib.md5(steps_summary.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{from_id}|{to_id}|{action}|{semantic}|{steps_digest}"


def _state_semantic_key(state: object) -> str:
    route = _as_str(getattr(state, "spa_route", "")).strip().lower()
    cleaned = clean_url(_as_str(getattr(state, "url", "")).strip())
    url = _as_str(cleaned or "").split("#", 1)[0].strip().lower()
    title = _as_str(getattr(state, "title", "")).strip().lower()
    if route:
        return f"route:{route}"
    if url:
        return f"url:{url}"
    if title:
        return f"title:{title}"
    return _as_str(getattr(state, "id", "")).strip().lower()


def _stable_signature(items: set[str]) -> str:
    if not items:
        return ""
    payload = "\n".join(sorted(item for item in items if item))
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


SEMANTIC_STABILITY_THRESHOLD = 90.0


def _json_list_payload(items: set[str]) -> str:
    return json.dumps(sorted(item for item in items if item), ensure_ascii=False)


def _parse_json_list(raw: object) -> set[str]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(item).strip() for item in parsed if str(item).strip()}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def _calculate_semantic_stability(
    *,
    semantic_metrics: dict[str, object],
    baseline_row: dict[str, object] | None,
    threshold: float = SEMANTIC_STABILITY_THRESHOLD,
) -> dict[str, object]:
    baseline = baseline_row or {}
    baseline_session_id = _as_str(baseline.get("session_id")).strip()
    current_state = _parse_json_list(semantic_metrics.get("semantic_state_keys_json"))
    current_transition = _parse_json_list(
        semantic_metrics.get("semantic_transition_keys_json")
    )
    current_intent = _parse_json_list(semantic_metrics.get("semantic_intent_keys_json"))
    baseline_state = _parse_json_list(baseline.get("semantic_state_keys_json"))
    baseline_transition = _parse_json_list(baseline.get("semantic_transition_keys_json"))
    baseline_intent = _parse_json_list(baseline.get("semantic_intent_keys_json"))

    if not baseline_session_id:
        return {
            "semantic_baseline_session_id": "",
            "semantic_stability_state_similarity": 1.0,
            "semantic_stability_transition_similarity": 1.0,
            "semantic_stability_intent_similarity": 1.0,
            "semantic_stability_score": 100.0,
            "semantic_stability_threshold": float(threshold),
            "semantic_stability_passed": True,
            "semantic_stability_mode": "bootstrap",
        }

    state_similarity = _jaccard_similarity(current_state, baseline_state)
    transition_similarity = _jaccard_similarity(current_transition, baseline_transition)
    intent_similarity = _jaccard_similarity(current_intent, baseline_intent)
    score = (
        0.4 * state_similarity
        + 0.4 * transition_similarity
        + 0.2 * intent_similarity
    ) * 100.0
    score = round(score, 2)
    threshold_value = float(threshold)
    return {
        "semantic_baseline_session_id": baseline_session_id,
        "semantic_stability_state_similarity": round(state_similarity, 4),
        "semantic_stability_transition_similarity": round(transition_similarity, 4),
        "semantic_stability_intent_similarity": round(intent_similarity, 4),
        "semantic_stability_score": score,
        "semantic_stability_threshold": threshold_value,
        "semantic_stability_passed": score >= threshold_value,
        "semantic_stability_mode": "compare",
    }


def _build_semantic_metrics(result: object) -> dict[str, object]:
    states = list(getattr(result, "states", []) or [])
    transitions = list(getattr(result, "transitions", []) or [])
    state_key_by_id: dict[str, str] = {}
    state_keys: set[str] = set()
    for state in states:
        key = _state_semantic_key(state)
        sid = _as_str(getattr(state, "id", "")).strip()
        if sid:
            state_key_by_id[sid] = key
        if key:
            state_keys.add(key)

    transition_keys: set[str] = set()
    intent_keys: set[str] = set()
    for transition in transitions:
        from_id = _as_str(getattr(transition, "from_state_id", "")).strip()
        to_id = _as_str(getattr(transition, "to_state_id", "")).strip()
        from_key = state_key_by_id.get(from_id, from_id)
        to_key = state_key_by_id.get(to_id, to_id)
        action = _as_str(getattr(transition, "action", "")).strip().lower()
        semantic = _as_str(getattr(transition, "semantic_action_key", "")).strip()
        if not semantic:
            semantic = _as_str(getattr(transition, "selector", "")).strip()
        transition_keys.add(f"{from_key}|{to_key}|{action}|{semantic}")

        intent = getattr(transition, "intent", None)
        intent_key = ""
        if intent is not None:
            intent_key = _as_str(getattr(intent, "key", "")).strip()
        if not intent_key:
            intent_key = semantic
        if intent_key:
            intent_keys.add(intent_key)

    return {
        "semantic_state_count": len(state_keys),
        "semantic_transition_count": len(transition_keys),
        "semantic_intent_count": len(intent_keys),
        "semantic_state_signature": _stable_signature(state_keys),
        "semantic_transition_signature": _stable_signature(transition_keys),
        "semantic_intent_signature": _stable_signature(intent_keys),
        "semantic_state_keys_json": _json_list_payload(state_keys),
        "semantic_transition_keys_json": _json_list_payload(transition_keys),
        "semantic_intent_keys_json": _json_list_payload(intent_keys),
    }


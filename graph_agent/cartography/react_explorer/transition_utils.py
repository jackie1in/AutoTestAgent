from __future__ import annotations

import json
from typing import TYPE_CHECKING

from graph_agent.cartography.react_explorer.url_utils import _extract_spa_route

if TYPE_CHECKING:
    from graph_agent.models import Transition


def _transition_dedupe_key(transition: Transition) -> str:
    return "|".join(
        [
            str(transition.from_state_id or "").strip(),
            str(transition.to_state_id or "").strip(),
            str(
                transition.action.value
                if hasattr(transition.action, "value")
                else transition.action
            )
            .strip()
            .lower(),
            str(transition.semantic_action_key or transition.selector or "").strip(),
        ]
    )


def _should_commit_transition(
    url_before: str,
    url_after: str,
    action_type: str,
) -> bool:
    """Determine whether pending steps should be committed as a Transition.

    Commit happens when:
    - URL or SPA route changes (real page navigation)
    - A click triggers a structural state change (e.g. modal close, tab switch)

    fill/select on the same page are accumulated, not committed.
    """
    if url_before != url_after:
        return True
    route_before = _extract_spa_route(url_before)
    route_after = _extract_spa_route(url_after)
    if route_before != route_after:
        return True
    if action_type == "click":
        return True
    return False


def _infer_param_name_from_snapshot(snapshot_json: str | None) -> str | None:
    """Infer param_name from element snapshot JSON (name/id/placeholder/type)."""
    if not snapshot_json:
        return None
    try:
        snap = json.loads(snapshot_json)
    except (json.JSONDecodeError, TypeError):
        return None
    attrs = snap if isinstance(snap, dict) else {}
    # Direct attribute fields first
    for key in ("name", "id", "placeholder"):
        val = attrs.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    # Nested attributes dict
    nested = attrs.get("attributes")
    if isinstance(nested, dict):
        for key in ("name", "id", "placeholder"):
            val = nested.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        # Fallback: use input type as param hint (e.g. "password")
        input_type = nested.get("type")
        if (
            input_type
            and isinstance(input_type, str)
            and input_type.strip()
            in (
                "password",
                "email",
                "tel",
                "search",
                "url",
                "number",
            )
        ):
            return input_type.strip()
    return None



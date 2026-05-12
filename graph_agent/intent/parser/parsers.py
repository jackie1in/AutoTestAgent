from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any


from graph_agent.models import (
    ActionType,
    ElementConstraints,
    ElementSnapshot,
    FrameLocatorSnapshot,
    GraphEdge,
    TabActionType,
    TabSnapshot,
)
from graph_agent.intent.parser.base import (
    ACTION_MAPPING,
    _get_next_goal,
    distill_ui_thought,
    infer_intent_progressive,
)

logger = logging.getLogger(__name__)



def _infer_constraints(
    param_name: str | None, element: ElementSnapshot | None = None
) -> ElementConstraints | None:
    """Infer loose constraints from param_name and element metadata."""
    if not param_name and element is None:
        return None

    key = (param_name or "").lower()
    input_type = element.type.lower() if element and element.type else ""
    constraints = ElementConstraints()
    if key == "email" or input_type == "email":
        constraints.format = "email"
    elif "password" in key or input_type == "password":
        constraints.format = "password"
        constraints.masked = True
    elif "phone" in key or input_type == "tel":
        constraints.format = "phone"

    if constraints.format is None and not constraints.masked:
        return None
    return constraints


def _selector_from_element(interacted: dict | list | object) -> str:
    """Extract selector from interacted element dict (or list of dicts)."""
    if not interacted:
        return ""

    # Handle list case (browser-use might return a list of elements)
    element = interacted
    if isinstance(interacted, list):
        if not interacted:
            return ""
        element = interacted[0]

    # If element is not a dict (e.g. DOMInteractedElement object), try to convert or access attributes
    if not isinstance(element, dict):
        # Try to access attributes directly if it's an object
        xpath = getattr(element, "xpath", None)
        if xpath:
            return f"xpath={xpath}"
        x_path = getattr(element, "x_path", None)
        if x_path:
            return f"xpath={x_path}"

        css = getattr(element, "css_selector", None)
        if css:
            return css

        # If it has a to_dict method (Pydantic model or similar)
        if hasattr(element, "to_dict"):
            element = element.to_dict()
        elif hasattr(element, "dict"):
            element = element.dict()
        else:
            # Last resort: try __dict__
            try:
                element = element.__dict__
            except AttributeError:
                return ""

    if not isinstance(element, Mapping):
        return ""

    # Now treat as mapping
    # Try xpath first
    xpath = element.get("xpath")
    if xpath:
        return f"xpath={xpath}"
    x_path = element.get("x_path")
    if x_path:
        return f"xpath={x_path}"

    # Try CSS selector
    css = element.get("css_selector")
    if css:
        return css

    # Try attributes
    attrs_raw = element.get("attributes", {})
    attrs: Mapping[str, Any] = attrs_raw if isinstance(attrs_raw, Mapping) else {}
    if attrs.get("id"):
        return f"#{attrs['id']}"
    if attrs.get("name"):
        return f"[name='{attrs['name']}']"
    if attrs.get("class"):
        return f".{attrs['class'].replace(' ', '.')}"

    return ""


def _normalize_interacted_element(
    interacted: dict | list | object,
) -> Mapping[str, Any]:
    """Normalize browser-use interacted element into a mapping."""
    if not interacted:
        return {}
    element = (
        interacted[0] if isinstance(interacted, list) and interacted else interacted
    )
    if isinstance(element, Mapping):
        return element
    if hasattr(element, "to_dict"):
        element = element.to_dict()
    elif hasattr(element, "dict"):
        element = element.dict()
    elif hasattr(element, "__dict__"):
        element = element.__dict__
    return element if isinstance(element, Mapping) else {}


def _normalize_mapping(value: Any) -> Mapping[str, Any]:
    """Normalize dict-like values into a mapping."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    elif hasattr(value, "__dict__"):
        value = value.__dict__
    return value if isinstance(value, Mapping) else {}


def _extract_tab_snapshot(action: Mapping[str, Any]) -> TabSnapshot | None:
    raw_tab = action.get("tab")
    tab_data = _normalize_mapping(raw_tab)
    if not tab_data:
        return None
    tab_id = tab_data.get("tab_id")
    if not tab_id:
        return None
    return TabSnapshot(
        tab_id=str(tab_id),
        opener_tab_id=(
            str(tab_data.get("opener_tab_id"))
            if tab_data.get("opener_tab_id") is not None
            else None
        ),
        url=str(tab_data.get("url")) if tab_data.get("url") is not None else None,
        title=str(tab_data.get("title")) if tab_data.get("title") is not None else None,
    )


def _extract_tab_action(action: Mapping[str, Any]) -> TabActionType | None:
    raw_tab_action = action.get("tab_action")
    if raw_tab_action is None:
        return None
    if isinstance(raw_tab_action, TabActionType):
        return raw_tab_action
    try:
        return TabActionType(str(raw_tab_action))
    except ValueError:
        return None


def _frame_locator_snapshot_from_interacted(
    interacted: Mapping[str, Any],
) -> FrameLocatorSnapshot | None:
    """Build a frame locator snapshot from one raw iframe mapping."""
    selector = _selector_from_element(interacted)
    if not selector:
        return None
    attrs_raw = interacted.get("attributes", {})
    attrs = dict(attrs_raw) if isinstance(attrs_raw, Mapping) else {}
    return FrameLocatorSnapshot(
        selector=selector,
        xpath=str(interacted.get("xpath")) if interacted.get("xpath") else None,
        x_path=str(interacted.get("x_path")) if interacted.get("x_path") else None,
        css_selector=str(interacted.get("css_selector"))
        if interacted.get("css_selector")
        else None,
        name=str(attrs.get("name")) if attrs.get("name") is not None else None,
        id=str(attrs.get("id")) if attrs.get("id") is not None else None,
        attributes=attrs,
    )


def _extract_frame_path_from_interacted(
    interacted: dict | list | object,
) -> list[FrameLocatorSnapshot]:
    """Extract nested iframe locator chain from interacted element metadata."""
    element = _normalize_interacted_element(interacted)
    raw_frame_path = element.get("frame_path", [])
    if not isinstance(raw_frame_path, list):
        return []

    frame_path: list[FrameLocatorSnapshot] = []
    for raw_frame in raw_frame_path:
        normalized = _normalize_interacted_element(raw_frame)
        if not normalized:
            continue
        snapshot = _frame_locator_snapshot_from_interacted(normalized)
        if snapshot is not None:
            frame_path.append(snapshot)
    return frame_path


def _element_snapshot_from_interacted(
    interacted: dict | list | object,
) -> ElementSnapshot | None:
    """Build an element snapshot from browser-use interacted_element."""
    element = _normalize_interacted_element(interacted)
    if not element:
        return None

    attrs_raw = element.get("attributes", {})
    attrs = dict(attrs_raw) if isinstance(attrs_raw, Mapping) else {}
    selector = _selector_from_element(element)
    
    # Extract text content (try multiple sources)
    text_content = None
    for key in ["text_content", "textContent", "inner_text", "innerText", "text"]:
        if element.get(key):
            text_content = str(element.get(key)).strip()
            if text_content:
                break
    
    return ElementSnapshot(
        selector=selector,
        xpath=str(element.get("xpath")) if element.get("xpath") else None,
        x_path=str(element.get("x_path")) if element.get("x_path") else None,
        css_selector=str(element.get("css_selector"))
        if element.get("css_selector")
        else None,
        name=str(attrs.get("name")) if attrs.get("name") is not None else None,
        id=str(attrs.get("id")) if attrs.get("id") is not None else None,
        class_name=str(attrs.get("class")) if attrs.get("class") is not None else None,
        type=str(attrs.get("type")) if attrs.get("type") is not None else None,
        tag_name=str(element.get("tag_name")) if element.get("tag_name") else None,
        text_content=text_content if text_content else None,
        inner_text=str(element.get("inner_text")) if element.get("inner_text") else text_content,
        placeholder=str(attrs.get("placeholder")) if attrs.get("placeholder") else None,
        aria_label=str(attrs.get("aria-label")) if attrs.get("aria-label") else None,
        value=str(attrs.get("value")) if attrs.get("value") else None,
        href=str(attrs.get("href")) if attrs.get("href") else None,
        title=str(attrs.get("title")) if attrs.get("title") else None,
        attributes=attrs,
        frame_path=_extract_frame_path_from_interacted(interacted),
    )


def _is_contenteditable(element: ElementSnapshot | None) -> bool:
    """Detect whether the interacted element is a contenteditable rich text target."""
    if element is None:
        return False
    attrs = element.attributes or {}
    if "contenteditable" in attrs and str(attrs["contenteditable"]).lower() in (
        "true",
        "",
    ):
        return True
    tag = str(element.type or "").lower()
    if tag in ("input", "textarea", "select"):
        return False
    sel = (element.selector or "").lower()
    rich_hints = ("tinymce", "ckeditor", "ql-editor", "prosemirror", "contenteditable")
    if any(h in sel for h in rich_hints):
        return True
    for v in attrs.values():
        if isinstance(v, str) and any(h in v.lower() for h in rich_hints):
            return True
    return False


async def _infer_param_name(
    element: ElementSnapshot | None, is_fill: bool
) -> str | None:
    """Infer param_name from real element attributes for fill actions."""
    if not is_fill or element is None:
        return None
    if element.name:
        return element.name
    if element.id:
        return element.id
    return None


async def _extract_action_value(
    action: Mapping[str, Any], play_action: ActionType
) -> str | None:
    """Extract raw recorded value from browser-use action payload."""
    if play_action not in (ActionType.FILL, ActionType.RICH_TEXT, ActionType.SELECT):
        return None

    for action_key in ("input_text", "input", "send_keys", "select_dropdown"):
        payload = action.get(action_key)
        if not isinstance(payload, Mapping):
            continue
        for candidate_key in ("text", "value", "selected", "option", "label"):
            value = payload.get(candidate_key)
            if value is None:
                continue
            return str(value)
    return None


async def parse_browser_use_step(
    action: dict,
    thought: dict | object,
    source_url: str,
    target_url: str,
    neighbor_steps: list[dict[str, str]] | None = None,
    page_signals: dict[str, str] | None = None,
) -> GraphEdge:
    """Parse one browser-use step to GraphEdge.

    Args:
        action: The action dictionary from browser-use.
        thought: The thought object or dict from browser-use.
        source_url: The URL before the action.
        target_url: The URL after the action.
    """
    # Ensure action is a dict
    if not isinstance(action, dict):
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        elif hasattr(action, "dict"):
            action = action.dict()
        elif hasattr(action, "__dict__"):
            action = action.__dict__
        else:
            # Fallback
            logger.warning(
                "action is not a dict and cannot be converted: %s", type(action)
            )
            action = {}

    interacted = action.get("interacted_element") or {}
    element = _element_snapshot_from_interacted(interacted)
    selector = (
        element.selector if element is not None else _selector_from_element(interacted)
    )

    # Extract action type (click/fill) using mapping
    play_action = ActionType.UNKNOWN
    for key, action_type in ACTION_MAPPING.items():
        if key in action:
            play_action = action_type
            break

    if play_action == ActionType.UNKNOWN:
        # Only print if it's truly an unknown/unmapped action key (not just one we explicitly mapped to UNKNOWN)
        known_keys = set(ACTION_MAPPING.keys())
        if not any(k in action for k in known_keys):
            logger.warning("Unknown action in parse_browser_use_step: %s", action)

    raw_thought = _get_next_goal(thought)
    distilled_thought = await distill_ui_thought(
        thought_text=raw_thought,
        action=play_action,
        selector=selector,
        source_url=source_url,
        target_url=target_url,
    )
    if play_action == ActionType.FILL and _is_contenteditable(element):
        play_action = ActionType.RICH_TEXT

    is_fill = play_action in (ActionType.FILL, ActionType.RICH_TEXT, ActionType.SELECT)
    param_name = await _infer_param_name(element, is_fill)
    action_value = await _extract_action_value(action, play_action)
    intent, intent_failure_reason, context_level_used = await infer_intent_progressive(
        action=play_action,
        selector=selector,
        source_url=source_url,
        target_url=target_url,
        param_name=param_name,
        thought_text=distilled_thought,
        neighbor_steps=neighbor_steps,
        page_signals=page_signals,
    )
    if intent is None:
        logger.warning(
            "Intent inference failed: %s",
            {
                "action": play_action.value,
                "selector": selector,
                "source": source_url,
                "target": target_url,
                "reason": intent_failure_reason or "unknown",
            },
        )
    constraints = _infer_constraints(param_name, element)
    tab_id = str(action.get("tab_id") or "tab-0")
    target_tab_id = (
        str(action.get("target_tab_id"))
        if action.get("target_tab_id") is not None
        else None
    )
    tab_action = _extract_tab_action(action)
    tab = _extract_tab_snapshot(action)

    return GraphEdge(
        source=source_url,
        target=target_url,
        selector=selector,
        action=play_action,
        tab_id=tab_id,
        target_tab_id=target_tab_id,
        tab_action=tab_action,
        tab=tab,
        frame_path=element.frame_path if element is not None else [],
        intent=intent,
        context_level_used=context_level_used,
        intent_failure_reason=intent_failure_reason,
        param_name=param_name,
        action_value=action_value,
        element=element,
        constraints=constraints,
    )


def parse_browser_use_step_lite(
    action: dict,
    thought: dict | object,
    source_url: str,
    target_url: str,
) -> GraphEdge:
    """No-LLM parse: extracts action/selector/element only. Intent is left as None/pending."""
    if not isinstance(action, dict):
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        elif hasattr(action, "dict"):
            action = action.dict()
        elif hasattr(action, "__dict__"):
            action = action.__dict__
        else:
            action = {}

    interacted = action.get("interacted_element") or {}
    element = _element_snapshot_from_interacted(interacted)
    selector = (
        element.selector if element is not None else _selector_from_element(interacted)
    )

    play_action = ActionType.UNKNOWN
    for key, action_type in ACTION_MAPPING.items():
        if key in action:
            play_action = action_type
            break

    if play_action == ActionType.FILL and _is_contenteditable(element):
        play_action = ActionType.RICH_TEXT

    is_fill = play_action in (ActionType.FILL, ActionType.RICH_TEXT, ActionType.SELECT)
    param_name: str | None = None
    if is_fill and element is not None:
        param_name = element.name or element.id or None

    action_value: str | None = None
    if play_action in (ActionType.FILL, ActionType.RICH_TEXT, ActionType.SELECT):
        for action_key in ("input_text", "input", "send_keys", "select_dropdown"):
            payload = action.get(action_key)
            if not isinstance(payload, Mapping):
                continue
            for candidate_key in ("text", "value", "selected", "option", "label"):
                value = payload.get(candidate_key)
                if value is not None:
                    action_value = str(value)
                    break
            if action_value is not None:
                break

    constraints = _infer_constraints(param_name, element)
    tab_id = str(action.get("tab_id") or "tab-0")
    target_tab_id = (
        str(action.get("target_tab_id"))
        if action.get("target_tab_id") is not None
        else None
    )
    tab_action = _extract_tab_action(action)
    tab = _extract_tab_snapshot(action)

    return GraphEdge(
        source=source_url,
        target=target_url,
        selector=selector,
        action=play_action,
        tab_id=tab_id,
        target_tab_id=target_tab_id,
        tab_action=tab_action,
        tab=tab,
        frame_path=element.frame_path if element is not None else [],
        intent=None,
        context_level_used=None,
        intent_failure_reason="pending",
        param_name=param_name,
        action_value=action_value,
        element=element,
        constraints=constraints,
    )

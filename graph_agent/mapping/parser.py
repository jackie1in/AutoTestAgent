"""Parse browser-use action/thought into structured GraphEdge."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from graph_agent.models import ActionType, ElementConstraints, GraphEdge, Intent
from graph_agent.llm import get_llm
from langchain_core.messages import HumanMessage

# Map semantic_label keywords (CN/EN) to data_key for fill actions.
# Deprecated: Logic moved to _infer_data_key
SEMANTIC_LABEL_TO_DATA_KEY: list[tuple[list[str], str]] = []


# Map action keys to ActionType
ACTION_MAPPING = {
    "click_element": ActionType.CLICK,
    "input_text": ActionType.FILL,
    "navigate_browser": ActionType.NAVIGATE,
    "click": ActionType.CLICK,
    "input": ActionType.FILL,
    "navigate": ActionType.NAVIGATE,
    "scroll": ActionType.UNKNOWN,
    "done": ActionType.UNKNOWN,
    "write_file": ActionType.UNKNOWN,
    "read_file": ActionType.UNKNOWN,
}

_NOISE_GOAL_PATTERNS = (
    r"^the task is complete",
    r"^i will now output the final result",
    r"^task complete",
    r"^document .*csv file",
    r"^write .*csv file",
    r"^stopped:",
    r"^任务已完成",
    r"^我将输出最终结果",
    r"^将.*记录到.*csv",
)

MIN_INTENT_CONFIDENCE = 0.4


class DataKeyExtractor:
    """Uses LLM to infer data_key from semantic label and context."""
    
    def __init__(self):
        self.llm = get_llm()
        
    async def extract(self, semantic_label: str) -> str | None:
        """Infer data_key using LLM. Returns None if no clear data key."""
        if not semantic_label:
            return None
            
        prompt = f"""
        Analyze the following user intent/thought and determine if it refers to filling a standard form field.
        If it does, return the standard data key (e.g., 'username', 'password', 'email', 'phone', 'otp', 'search_query').
        If it's not a fill action or the field is ambiguous/custom, return 'null'.
        
        User Intent: "{semantic_label}"
        
        Output ONLY the data key or 'null'. No markdown, no explanation.
        """
        
        try:
            # Asynchronous invocation for browser-use LLM compatibility
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Handle browser-use ChatInvokeCompletion or standard LangChain AIMessage
            if hasattr(response, "completion"):
                content = str(response.completion)
            else:
                content = getattr(response, "content", str(response))
            
            content = content.strip().lower()
            
            if content == "null" or "null" in content:
                return None
            return content
        except Exception as e:
            print(f"Warning: DataKeyExtractor LLM failed: {e}")
            return None


def _get_next_goal(thought: dict | object) -> str:
    """Extract and clean next_goal from thought."""
    raw_goal = ""
    if isinstance(thought, dict):
        raw_goal = thought.get("next_goal", "") or ""
    else:
        raw_goal = getattr(thought, "next_goal", "") or ""

    goal = raw_goal.strip()
    if not goal:
        return ""

    normalized = re.sub(r"\s+", " ", goal).strip().lower()
    for pattern in _NOISE_GOAL_PATTERNS:
        if re.search(pattern, normalized):
            return ""
    return goal


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from text and parse it."""
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _response_to_text(response: Any) -> str:
    """Normalize llm response into plain text."""
    if hasattr(response, "completion"):
        return str(response.completion)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(getattr(c, "text", getattr(c, "content", c))) for c in content]
        return " ".join(parts)
    return str(response)


async def infer_intent_for_context(
    action: ActionType,
    selector: str,
    source_url: str,
    target_url: str,
    data_key: str | None,
    thought_text: str,
) -> tuple[Intent | None, str | None]:
    """Infer intent from context using AI only."""
    llm = get_llm()
    prompt = f"""
You are an intent normalizer for browser automation steps.
Given one interaction step context, infer business intent.

Output STRICT JSON object with fields:
- key: short dot-separated intent key (example: auth.fill.username, auth.submit.login, generic.click.button)
- confidence: number between 0 and 1
- summary: concise natural language summary
- verb: action verb
- object: target object

Context:
- action: {action.value}
- selector: {selector}
- source_url: {source_url}
- target_url: {target_url}
- data_key: {data_key or ""}
- thought: {thought_text or ""}

Rules:
- Return JSON only.
- If uncertain, still provide best guess with low confidence.
"""
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001
        return None, f"llm_error:{exc}"

    payload = _extract_json_object(_response_to_text(response))
    if payload is None:
        return None, "parse_error:invalid_json"

    key = str(payload.get("key", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    verb = str(payload.get("verb", "")).strip() or "Interact"
    obj = str(payload.get("object", "")).strip() or "Element"
    conf_raw = payload.get("confidence")
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        return None, "parse_error:invalid_confidence"

    if not key or not summary:
        return None, "parse_error:missing_key_or_summary"
    if confidence < MIN_INTENT_CONFIDENCE:
        return None, f"low_confidence:{confidence:.2f}"
    confidence = max(0.0, min(1.0, confidence))

    intent = Intent(
        raw=thought_text or summary,
        verb=verb,
        object=obj,
        summary=summary,
        key=key,
        confidence=confidence,
    )
    return intent, None


def _infer_constraints(data_key: str | None) -> ElementConstraints | None:
    """Infer constraints based on data_key."""
    if not data_key:
        return None
    
    constraints = ElementConstraints()
    if data_key == "email":
        constraints.format = "email"
    elif data_key == "password":
        constraints.format = "password"
        constraints.masked = True
    elif data_key == "phone":
        constraints.format = "phone"
    
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


async def _infer_data_key(semantic_label: str, is_fill: bool) -> str | None:
    """Infer data_key from semantic_label for fill actions."""
    if not is_fill or not semantic_label:
        return None
    
    # Use AI inference exclusively
    try:
        extractor = DataKeyExtractor()
        return await extractor.extract(semantic_label)
    except Exception:
        return None


async def parse_browser_use_step(action: dict, thought: dict | object, source_url: str, target_url: str) -> GraphEdge:
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
            print(f"Warning: action is not a dict and cannot be converted: {type(action)}")
            action = {}

    interacted = action.get("interacted_element") or {}
    selector = _selector_from_element(interacted)
    
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
            print(f"Unknown action: {action} in parse_browser_use_step")
    
    raw_thought = _get_next_goal(thought)
    is_fill = play_action == ActionType.FILL
    data_key = await _infer_data_key(raw_thought, is_fill)
    intent, intent_failure_reason = await infer_intent_for_context(
        action=play_action,
        selector=selector,
        source_url=source_url,
        target_url=target_url,
        data_key=data_key,
        thought_text=raw_thought,
    )
    if intent is None:
        print(
            "Intent inference failed:",
            {
                "action": play_action.value,
                "selector": selector,
                "source": source_url,
                "target": target_url,
                "reason": intent_failure_reason or "unknown",
            },
        )
    constraints = _infer_constraints(data_key)

    return GraphEdge(
        source=source_url,
        target=target_url,
        selector=selector,
        action=play_action,
        intent=intent,
        intent_failure_reason=intent_failure_reason,
        data_key=data_key,
        constraints=constraints,
    )

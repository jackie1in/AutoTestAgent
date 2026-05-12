"""Parse browser-use action/thought into structured GraphEdge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from graph_agent.models import (
    ActionType,
    Intent,
)
from graph_agent.llm import get_llm, ainvoke_structured

logger = logging.getLogger(__name__)

# Matches browser-use index selectors like [17531] or [12345]
INDEX_SELECTOR_RE = re.compile(r"^\[\d+\]$")


class IntentInferenceResult(BaseModel):
    """Result of intent inference from browser automation step."""
    key: str = Field(description="Short dot-separated intent key, e.g. auth.fill.username")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    summary: str = Field(description="Concise natural language summary")
    verb: str = Field(default="Interact", description="Action verb")
    object: str = Field(default="Element", description="Target object")


class KeyRefinementResult(BaseModel):
    """Result of refining an intent key."""
    key: str = Field(description="Dot-separated lowercase intent key")


class UIDistillationResult(BaseModel):
    """Result of distilling UI thought from raw thought text."""
    ui_thought: str = Field(default="", description="Cleaned UI interaction thought")

# Map action keys to ActionType
ACTION_MAPPING = {
    "click_element": ActionType.CLICK,
    "input_text": ActionType.FILL,
    "navigate_browser": ActionType.NAVIGATE,
    "click": ActionType.CLICK,
    "input": ActionType.FILL,
    "navigate": ActionType.NAVIGATE,
    "go_back": ActionType.NAVIGATE,
    "select_dropdown": ActionType.SELECT,
    "send_keys": ActionType.FILL,
    "evaluate": ActionType.UNKNOWN,
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

MIN_INTENT_CONFIDENCE = 0.2
_THOUGHT_NOISE_HINTS = (
    "write_file",
    "read_file",
    "todo",
    "log",
    "csv",
    "report",
    "final result",
    "output file",
    "document",
)

# ---------------------------------------------------------------------------
# Phase 1 optimizations: skip switches + intent cache
# ---------------------------------------------------------------------------

_intent_cache: dict[str, tuple[Intent, str]] = {}
_intent_schema_echo_short_circuit_until: float = 0.0


def _heuristic_intent_fallback(
    *,
    action: ActionType,
    selector: str,
    source_url: str,
    target_url: str,
    param_name: str | None,
    thought_text: str,
) -> Intent:
    selector_l = (selector or "").lower()
    thought_l = (thought_text or "").lower()
    src = (source_url or "").lower()
    tgt = (target_url or "").lower()
    login_context = any(token in f"{src} {tgt} {selector_l} {thought_l}" for token in ("login", "signin", "auth", "密码", "验证码"))

    if action in (ActionType.FILL, ActionType.RICH_TEXT, ActionType.SELECT):
        field_raw = (param_name or "").strip() or "field"
        field = re.sub(r"[^a-z0-9_]+", "_", field_raw.lower()).strip("_") or "field"
        key = f"{'auth' if login_context else 'form'}.fill.{field}"
        summary = f"Fill {field} input"
        return Intent(raw=thought_text or summary, verb="Fill", object=field, summary=summary, key=key, confidence=0.34)

    if action == ActionType.CLICK:
        if login_context and any(token in selector_l for token in ("submit", "login", "signin", "btn")):
            key = "auth.click.submit"
            summary = "Click login submit button"
            obj = "Login Button"
        else:
            key = "elements.click.control"
            summary = "Click page control"
            obj = "Control"
        return Intent(raw=thought_text or summary, verb="Click", object=obj, summary=summary, key=key, confidence=0.31)

    if action == ActionType.NAVIGATE:
        summary = "Navigate to page"
        return Intent(raw=thought_text or summary, verb="Navigate", object="Page", summary=summary, key="navigation.page.change", confidence=0.33)

    summary = "Interact with page element"
    return Intent(raw=thought_text or summary, verb="Interact", object="Element", summary=summary, key="elements.interact.unknown", confidence=0.3)


def _should_skip_refine() -> bool:
    return (os.getenv("MAPPING_SKIP_REFINE") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _should_skip_distill() -> bool:
    return (os.getenv("MAPPING_SKIP_DISTILL") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _make_intent_cache_key(
    action: ActionType,
    selector: str,
    source_url: str,
    target_url: str,
    param_name: str | None,
) -> str:
    src_path = urlparse(source_url).path if source_url else ""
    tgt_path = urlparse(target_url).path if target_url else ""
    raw = f"{action.value}|{selector}|{src_path}|{tgt_path}|{param_name or ''}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def clear_intent_cache() -> None:
    """Reset the in-memory intent cache (useful in tests)."""
    _intent_cache.clear()


def _schema_echo_cooldown_seconds() -> float:
    raw = (os.getenv("MAPPING_INTENT_SCHEMA_ECHO_COOLDOWN_SEC") or "").strip()
    try:
        value = float(raw)
    except Exception:
        value = 300.0
    return max(30.0, value)


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
    param_name: str | None,
    thought_text: str,
    neighbor_steps: list[dict[str, str]] | None = None,
    page_signals: dict[str, str] | None = None,
    context_level: str = "L0",
    playback_error_hint: str | None = None,
) -> tuple[Intent | None, str | None]:
    """Infer intent from context using AI only."""
    global _intent_schema_echo_short_circuit_until
    # Cache lookup (skip when doing repair inference with playback hints)
    if not playback_error_hint:
        cache_key = _make_intent_cache_key(action, selector, source_url, target_url, param_name)
        cached = _intent_cache.get(cache_key)
        if cached is not None:
            return cached[0], None
    else:
        cache_key = ""

    now = time.time()
    if now < _intent_schema_echo_short_circuit_until:
        fallback = _heuristic_intent_fallback(
            action=action,
            selector=selector,
            source_url=source_url,
            target_url=target_url,
            param_name=param_name,
            thought_text=thought_text,
        )
        if cache_key:
            _intent_cache[cache_key] = (fallback, context_level)
        return fallback, "fallback:structured_schema_echo_short_circuit"

    llm = get_llm()
    neighbor_section = ""
    if neighbor_steps:
        pairs = []
        for item in neighbor_steps:
            pairs.append(
                f"- action:{item.get('action', '')} selector:{item.get('selector', '')} "
                f"source:{item.get('source_url', '')} target:{item.get('target_url', '')} thought:{item.get('thought', '')}"
            )
        neighbor_section = "Neighbor steps:\n" + "\n".join(pairs) + "\n"

    page_signal_section = ""
    if page_signals:
        page_signal_section = (
            "Page signals:\n"
            + "\n".join(f"- {k}: {v}" for k, v in page_signals.items() if v)
            + "\n"
        )
    playback_section = ""
    if playback_error_hint:
        playback_section = f"Playback failure hint:\n- error: {playback_error_hint}\n"

    system_prompt = """You are an intent normalizer for browser automation steps.
Given one interaction step context, infer business intent."""
    
    user_prompt = f"""Infer the business intent for this browser automation step:

Context:
- context_level: {context_level}
- action: {action.value}
- selector: {selector}
- source_url: {source_url}
- target_url: {target_url}
- param_name: {param_name or ""}
- thought: {thought_text or ""}
{neighbor_section}{page_signal_section}{playback_section}

Rules:
- Prefer domain/business keys over generic/navigation when action is click/fill.
- Avoid broad keys like navigation.click.link unless step is only pure page jump.
- Use selector/param_name/url/thought cues to produce specific key (e.g., auth.*, form.*, elements.*).
- If uncertain, still provide best guess with low confidence."""

    try:
        result = await ainvoke_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_format=IntentInferenceResult,
        )
        if isinstance(result, IntentInferenceResult):
            parsed_result = result
        else:
            raw_text = _response_to_text(result)
            parsed_obj = _extract_json_object(raw_text)
            if parsed_obj is None:
                return None, "parse_error:invalid_json"
            try:
                parsed_result = IntentInferenceResult.model_validate(parsed_obj)
            except Exception:
                return None, "parse_error:invalid_json"

        key = parsed_result.key
        confidence = parsed_result.confidence
        summary = parsed_result.summary
        verb = parsed_result.verb
        obj = parsed_result.object
    except asyncio.TimeoutError:
        timeout_ms = os.getenv("MAPPING_INTENT_TIMEOUT_MS", "?")
        return None, f"llm_error:timeout({timeout_ms}ms)"
    except Exception as exc:  # noqa: BLE001
        err_text = str(exc)
        schema_echo_like = "SCHEMA_ECHO_STRUCTURED_OUTPUT" in err_text
        if (
            schema_echo_like
            or
            "validation errors for IntentInferenceResult" in err_text
            or (
                "Failed to parse structured output" in err_text
                and "Field required" in err_text
                and "IntentInferenceResult" in err_text
            )
        ):
            fallback = _heuristic_intent_fallback(
                action=action,
                selector=selector,
                source_url=source_url,
                target_url=target_url,
                param_name=param_name,
                thought_text=thought_text,
            )
            if schema_echo_like:
                _intent_schema_echo_short_circuit_until = time.time() + _schema_echo_cooldown_seconds()
            if cache_key:
                _intent_cache[cache_key] = (fallback, context_level)
            return fallback, "fallback:structured_parse_failed"
        return None, f"llm_error:{exc}"

    if not key or not summary:
        logger.warning(
            "[Intent] parse_error: missing key or summary. key=%r summary=%r",
            key,
            summary,
        )
        return None, "parse_error:missing_key_or_summary"
    if confidence <= MIN_INTENT_CONFIDENCE:
        logger.info(
            "[Intent] low_confidence: %.2f < %.2f. key=%r summary=%r",
            confidence,
            MIN_INTENT_CONFIDENCE,
            key,
            summary,
        )
        return None, f"low_confidence:{confidence:.2f}"
    confidence = max(0.0, min(1.0, confidence))

    key = await _refine_non_business_key_if_needed(
        llm=llm,
        action=action,
        key=key,
        selector=selector,
        source_url=source_url,
        target_url=target_url,
        param_name=param_name,
        thought_text=thought_text,
        summary=summary,
    )

    intent = Intent(
        raw=thought_text or summary,
        verb=verb,
        object=obj,
        summary=summary,
        key=key,
        confidence=confidence,
    )
    if cache_key:
        _intent_cache[cache_key] = (intent, context_level)
    return intent, None


async def _refine_non_business_key_if_needed(
    llm: Any,
    action: ActionType,
    key: str,
    selector: str,
    source_url: str,
    target_url: str,
    param_name: str | None,
    thought_text: str,
    summary: str,
) -> str:
    """AI-only refinement to reduce over-broad navigation keys on click/fill."""
    raw_key = (key or "").strip()
    if _should_skip_refine():
        return raw_key
    if action not in (ActionType.CLICK, ActionType.FILL):
        return raw_key
    if not raw_key.startswith("navigation."):
        return raw_key

    system_prompt = "You refine an existing intent key for browser automation. Current key may be too broad."
    
    user_prompt = f"""Refine this intent key:

Context:
- action: {action.value}
- current_key: {raw_key}
- selector: {selector}
- source_url: {source_url}
- target_url: {target_url}
- param_name: {param_name or ""}
- thought: {thought_text or ""}
- summary: {summary}

Constraints:
- Keep dot-separated lowercase key.
- If action is click/fill, prefer business/domain key when possible.
- Avoid navigation.* when there is a concrete business operation.
- If truly only navigation, keep current key unchanged."""

    try:
        result = await ainvoke_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_format=KeyRefinementResult,
        )
    except Exception:
        return raw_key
    if isinstance(result, KeyRefinementResult):
        refined = result.key.strip().lower()
    else:
        raw_text = _response_to_text(result)
        parsed_obj = _extract_json_object(raw_text)
        if not parsed_obj:
            return raw_key
        refined_raw = str(parsed_obj.get("key") or "").strip().lower()
        if not refined_raw:
            return raw_key
        refined = refined_raw
    if not refined or "." not in refined:
        return raw_key
    return refined


def _action_intent_conflict(
    action: ActionType,
    intent: Intent,
    selector: str = "",
    source_url: str = "",
    target_url: str = "",
) -> bool:
    """Heuristic semantic conflict detector for progressive escalation."""
    key = (intent.key or "").lower()
    summary = (intent.summary or "").lower()
    sel = (selector or "").lower()
    text = f"{key} {summary}"
    if action == ActionType.SELECT:
        return False
    if action == ActionType.RICH_TEXT:
        return False
    if action == ActionType.FILL:
        # If selector strongly indicates a fillable control, always accept.
        if any(
            token in sel
            for token in (
                "input",
                "textarea",
                "select",
                "password",
                "username",
                "email",
                "text",
                "#user",
                "#pwd",
            )
        ):
            return False
        # Fallback: require at least one fill-related keyword in intent text
        return all(
            token not in text
            for token in (
                "fill",
                "input",
                "type",
                "enter",
                "select",
                "choose",
                "toggle",
            )
        )
    if action == ActionType.CLICK:
        # Clicking links/buttons/checkboxes is always valid click intent.
        if any(
            token in sel
            for token in ("a[", "/a", "href", "link", "button", "btn", "submit", "input[type=submit", "checkbox", "radio", "label", "span", "div", "li")
        ):
            return False
        # Index selectors like [17531] are valid click targets (browser-use internal refs).
        if INDEX_SELECTOR_RE.match(sel):
            return False
        # Fallback: require at least one click-related keyword in intent text
        return all(
            token not in text
            for token in (
                "click",
                "submit",
                "press",
                "tap",
                "toggle",
                "check",
                "open",
                "navigate",
                "visit",
                "go",
                "close",
                "dismiss",
                "remove",
                "delete",
                "hide",
                "select",
                "choose",
                "switch",
            )
        )
    if action == ActionType.NAVIGATE:
        src = (source_url or "").strip()
        tgt = (target_url or "").strip()
        # Real page transition is acceptable navigate intent even if wording is business-like.
        if src and tgt and src != tgt:
            return False
        return all(token not in text for token in ("navigate", "open", "visit", "go"))
    return False


async def infer_intent_progressive(
    action: ActionType,
    selector: str,
    source_url: str,
    target_url: str,
    param_name: str | None,
    thought_text: str,
    neighbor_steps: list[dict[str, str]] | None = None,
    page_signals: dict[str, str] | None = None,
) -> tuple[Intent | None, str | None, str]:
    """Single-pass inference with all context; retry once on low confidence or conflict."""
    intent, reason = await infer_intent_for_context(
        action=action,
        selector=selector,
        source_url=source_url,
        target_url=target_url,
        param_name=param_name,
        thought_text=thought_text,
        neighbor_steps=neighbor_steps,
        page_signals=page_signals,
        context_level="full",
    )
    if intent is not None:
        if not _action_intent_conflict(
            action, intent, selector=selector,
            source_url=source_url, target_url=target_url,
        ):
            logger.info("[Intent] OK full: %r %r", intent.key, intent.summary)
            return intent, None, "full"
        logger.info(
            "[Intent] conflict full: action=%s intent=%r selector=%r",
            action.value,
            intent.key,
            selector,
        )
        reason = "semantic_conflict:action_intent_mismatch"

    if reason and ("low_confidence" in reason or "semantic_conflict" in reason):
        intent, reason = await infer_intent_for_context(
            action=action,
            selector=selector,
            source_url=source_url,
            target_url=target_url,
            param_name=param_name,
            thought_text=thought_text,
            neighbor_steps=neighbor_steps,
            page_signals=page_signals,
            context_level="full_retry",
        )
        if intent is not None and not _action_intent_conflict(
            action, intent, selector=selector,
            source_url=source_url, target_url=target_url,
        ):
            logger.info("[Intent] OK retry: %r %r", intent.key, intent.summary)
            return intent, None, "full_retry"

    logger.warning(
        "[Intent] FAILED: action=%s sel=%r reason=%r", action.value, selector, reason
    )
    return None, (reason or "intent_inference_failed"), "full"


async def distill_ui_thought(
    thought_text: str,
    action: ActionType,
    selector: str,
    source_url: str,
    target_url: str,
) -> str:
    """Optional intermediate AI step: remove non-UI planning noise from thought.

    Trigger only when raw thought includes file/log/report-like hints, to keep
    overhead low and avoid unnecessary extra LLM calls.
    """
    raw = (thought_text or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if not any(hint in lowered for hint in _THOUGHT_NOISE_HINTS):
        return raw
    if _should_skip_distill():
        return raw

    system_prompt = "You clean browser-agent thought text for UI intent inference. Keep only user-visible UI interaction intent."
    
    user_prompt = f"""Clean this thought text:

Context:
- action: {action.value}
- selector: {selector}
- source_url: {source_url}
- target_url: {target_url}
- raw_thought: {raw}

Keep only user-visible UI interaction intent (click/fill/select/navigate).
Remove file operations, notes, logs, todo, report, summary, and meta planning.
If no UI intent remains, return empty ui_thought."""

    try:
        llm = get_llm()
        result = await ainvoke_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_format=UIDistillationResult,
        )
        return result.ui_thought.strip() or raw
    except Exception:
        return raw


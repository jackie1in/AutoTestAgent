"""Tests for parser AI-only intent inference (T2: no rule fallback)."""

import pytest
from unittest.mock import AsyncMock, patch

from graph_agent.models import ActionType, Intent
from graph_agent.mapping.parser import (
    infer_intent_for_context,
    parse_browser_use_step,
)


def _mock_llm_response(content: str):
    """Create response object with .content only (no .completion to avoid _response_to_text using wrong attr)."""
    return type("Response", (), {"content": content})()


@pytest.mark.asyncio
async def test_infer_intent_success():
    """AI returns valid JSON -> intent is returned, no failure reason."""
    mock_response = _mock_llm_response(
        '{"key":"auth.fill.username","confidence":0.9,"summary":"Fill username","verb":"Fill","object":"Username field"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.FILL,
            selector="#username",
            source_url="https://a.com",
            target_url="https://a.com",
            data_key="username",
            thought_text="Type user into username",
        )

    assert intent is not None
    assert intent.key == "auth.fill.username"
    assert intent.confidence == 0.9
    assert intent.summary == "Fill username"
    assert intent.verb == "Fill"
    assert intent.object == "Username field"
    assert reason is None


@pytest.mark.asyncio
async def test_infer_intent_failure_llm_error():
    """LLM raises -> returns (None, llm_error:...)."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API timeout"))

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="#btn",
            source_url="https://a.com",
            target_url="https://b.com",
            data_key=None,
            thought_text="Click button",
        )

    assert intent is None
    assert reason is not None
    assert "llm_error" in reason
    assert "API timeout" in reason


@pytest.mark.asyncio
async def test_infer_intent_failure_invalid_json():
    """LLM returns non-JSON -> returns (None, parse_error:invalid_json)."""
    mock_response = _mock_llm_response("This is not JSON at all")

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="#btn",
            source_url="https://a.com",
            target_url="https://b.com",
            data_key=None,
            thought_text="Click",
        )

    assert intent is None
    assert reason == "parse_error:invalid_json"


@pytest.mark.asyncio
async def test_infer_intent_failure_low_confidence():
    """LLM returns low confidence -> returns (None, low_confidence:...)."""
    mock_response = _mock_llm_response(
        '{"key":"generic.click","confidence":0.2,"summary":"Click","verb":"Click","object":"Button"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="#btn",
            source_url="https://a.com",
            target_url="https://b.com",
            data_key=None,
            thought_text="Click",
        )

    assert intent is None
    assert reason is not None
    assert "low_confidence" in reason


@pytest.mark.asyncio
async def test_infer_intent_failure_missing_key_or_summary():
    """LLM returns empty key/summary -> returns (None, parse_error:missing_key_or_summary)."""
    mock_response = _mock_llm_response(
        '{"key":"","confidence":0.8,"summary":"","verb":"Click","object":"Button"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="#btn",
            source_url="https://a.com",
            target_url="https://b.com",
            data_key=None,
            thought_text="Click",
        )

    assert intent is None
    assert reason == "parse_error:missing_key_or_summary"


@pytest.mark.asyncio
async def test_parse_browser_use_step_ai_failure_no_fake_intent():
    """When AI fails, edge has intent=None and intent_failure_reason; no rule-generated pseudo intent."""
    action = {
        "click": {"element": "button"},
        "interacted_element": {"xpath": "//button[@id='login']"},
    }
    thought = {"next_goal": "Click login button"}

    with patch(
        "graph_agent.mapping.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(None, "parse_error:invalid_json"),
    ), patch(
        "graph_agent.mapping.parser._infer_data_key",
        new_callable=AsyncMock,
        return_value=None,
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://a.com/login",
            target_url="https://a.com/dashboard",
        )

    assert edge.intent is None
    assert edge.intent_failure_reason == "parse_error:invalid_json"
    assert edge.source == "https://a.com/login"
    assert edge.target == "https://a.com/dashboard"
    assert edge.selector == "xpath=//button[@id='login']"
    assert edge.action == ActionType.CLICK


@pytest.mark.asyncio
async def test_parse_browser_use_step_ai_success():
    """When AI succeeds, edge has intent with key/confidence/summary/verb/object."""
    action = {
        "click": {"element": "button"},
        "interacted_element": {"xpath": "//button[@id='submit']"},
    }
    thought = {"next_goal": "Submit the form"}

    expected_intent = Intent(
        raw="Submit the form",
        verb="Submit",
        object="Form",
        summary="Submit login form",
        key="auth.submit.login",
        confidence=0.85,
    )

    with patch(
        "graph_agent.mapping.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ), patch(
        "graph_agent.mapping.parser._infer_data_key",
        new_callable=AsyncMock,
        return_value=None,
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://a.com",
            target_url="https://a.com/done",
        )

    assert edge.intent is expected_intent
    assert edge.intent_failure_reason is None
    assert edge.intent.key == "auth.submit.login"
    assert edge.intent.confidence == 0.85

"""Tests for parser AI-only intent inference (T2: no rule fallback)."""

import pytest
from unittest.mock import AsyncMock, patch

from graph_agent.models import ActionType, Intent, TabActionType
from graph_agent.mapping.parser import (
    _action_intent_conflict,
    infer_intent_for_context,
    infer_intent_progressive,
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
            param_name="username",
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
            param_name=None,
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
            param_name=None,
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
            param_name=None,
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
            param_name=None,
            thought_text="Click",
        )

    assert intent is None
    assert reason == "parse_error:missing_key_or_summary"


@pytest.mark.asyncio
async def test_infer_intent_refines_navigation_key_for_click():
    """Click intent with navigation.* key should be refined by AI second-pass."""
    first_pass = _mock_llm_response(
        '{"key":"navigation.click.link","confidence":0.91,"summary":"Click login link","verb":"Click","object":"Login link"}'
    )
    refined_pass = _mock_llm_response('{"key":"auth.navigate.login"}')
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[first_pass, refined_pass])

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="a[href='/login']",
            source_url="https://a.com",
            target_url="https://a.com/login",
            param_name=None,
            thought_text="go to login page",
        )

    assert reason is None
    assert intent is not None
    assert intent.key == "auth.navigate.login"
    assert mock_llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_infer_intent_skip_refine_for_navigate_action():
    """Navigate action should not trigger second-pass key refinement."""
    mock_response = _mock_llm_response(
        '{"key":"navigation.open.page","confidence":0.9,"summary":"Open page","verb":"Navigate","object":"Page"}'
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.mapping.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.NAVIGATE,
            selector="",
            source_url="https://a.com",
            target_url="https://a.com/b",
            param_name=None,
            thought_text="open page",
        )

    assert reason is None
    assert intent is not None
    assert intent.key == "navigation.open.page"
    assert mock_llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_parse_browser_use_step_ai_failure_no_fake_intent():
    """When AI fails, edge has intent=None and intent_failure_reason; no rule-generated pseudo intent."""
    action = {
        "click": {"element": "button"},
        "interacted_element": {"xpath": "//button[@id='login']"},
    }
    thought = {"next_goal": "Click login button"}

    with (
        patch(
            "graph_agent.mapping.parser.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(None, "parse_error:invalid_json"),
        ),
        patch(
            "graph_agent.mapping.parser._infer_param_name",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.mapping.parser._extract_action_value",
            new_callable=AsyncMock,
            return_value=None,
        ),
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

    with (
        patch(
            "graph_agent.mapping.parser.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(expected_intent, None),
        ),
        patch(
            "graph_agent.mapping.parser._infer_param_name",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.mapping.parser._extract_action_value",
            new_callable=AsyncMock,
            return_value=None,
        ),
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


@pytest.mark.asyncio
async def test_infer_intent_progressive_escalates_to_l2():
    """Progressive disclosure should escalate L0/L1 and accept at L2."""
    low_confidence = (
        None,
        "low_confidence:0.22",
    )
    high_confidence_intent = Intent(
        raw="submit login",
        verb="Submit",
        object="Login form",
        summary="Submit login form",
        key="auth.submit.login",
        confidence=0.91,
    )

    with patch(
        "graph_agent.mapping.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        side_effect=[low_confidence, low_confidence, (high_confidence_intent, None)],
    ) as mock_infer:
        intent, reason, level = await infer_intent_progressive(
            action=ActionType.CLICK,
            selector="#submit",
            source_url="https://a.com/login",
            target_url="https://a.com/secure",
            param_name=None,
            thought_text="submit login form",
            neighbor_steps=[{"action": "fill", "selector": "#username"}],
            page_signals={"source_path": "/login", "target_path": "/secure"},
        )

    assert intent is not None
    assert reason is None
    assert level == "L2"
    assert mock_infer.await_count == 3


def test_action_intent_conflict_allows_click_navigation_link():
    intent = Intent(
        raw="go login",
        verb="Navigate",
        object="Login page",
        summary="Navigate to login page",
        key="auth.navigate.login",
        confidence=0.91,
    )
    assert (
        _action_intent_conflict(ActionType.CLICK, intent, selector="a[href='/login']")
        is False
    )


def test_action_intent_conflict_allows_fill_on_input_selector():
    intent = Intent(
        raw="credentials",
        verb="Provide",
        object="Credentials",
        summary="Provide credentials",
        key="auth.credentials.password",
        confidence=0.82,
    )
    assert (
        _action_intent_conflict(ActionType.FILL, intent, selector="input#password")
        is False
    )


def test_action_intent_conflict_allows_navigate_when_url_changes():
    intent = Intent(
        raw="submit form",
        verb="Submit",
        object="Form",
        summary="Submit login form",
        key="auth.submit.login",
        confidence=0.87,
    )
    assert (
        _action_intent_conflict(
            ActionType.NAVIGATE,
            intent,
            selector="",
            source_url="https://a.com/login",
            target_url="https://a.com/secure",
        )
        is False
    )


def test_action_intent_conflict_keeps_navigate_conflict_without_transition():
    intent = Intent(
        raw="toggle checkbox",
        verb="Toggle",
        object="Checkbox",
        summary="Toggle checkbox",
        key="form.toggle.checkbox",
        confidence=0.87,
    )
    assert (
        _action_intent_conflict(
            ActionType.NAVIGATE,
            intent,
            selector="",
            source_url="https://a.com/checkboxes",
            target_url="https://a.com/checkboxes",
        )
        is True
    )


@pytest.mark.asyncio
async def test_parse_browser_use_step_fill_extracts_param_name_element_and_action_value():
    """Fill action should persist param_name, raw action value and element attrs."""
    action = {
        "input_text": {"text": "tomsmith"},
        "interacted_element": {
            "css_selector": "input[name='username']",
            "attributes": {"name": "username", "id": "user", "type": "text"},
        },
    }
    thought = {"next_goal": "Fill username"}
    expected_intent = Intent(
        raw="Fill username",
        verb="Fill",
        object="Username field",
        summary="Fill username",
        key="auth.fill.username",
        confidence=0.92,
    )

    with patch(
        "graph_agent.mapping.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(expected_intent, None),
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://a.com/login",
            target_url="https://a.com/login",
        )

    assert edge.selector == "input[name='username']"
    assert edge.param_name == "username"
    assert edge.action_value == "tomsmith"
    assert edge.element is not None
    assert edge.element.name == "username"
    assert edge.element.id == "user"
    assert edge.element.type == "text"
    assert edge.element.attributes["name"] == "username"


@pytest.mark.asyncio
async def test_parse_browser_use_step_extracts_nested_frame_path():
    """Nested iframe metadata should be copied onto both edge and element snapshot."""
    action = {
        "click": {"element": "button"},
        "interacted_element": {
            "css_selector": "#submit",
            "attributes": {"id": "submit"},
            "frame_path": [
                {
                    "css_selector": "iframe[name='outer']",
                    "attributes": {"name": "outer"},
                },
                {
                    "css_selector": "iframe[name='inner']",
                    "attributes": {"name": "inner"},
                },
            ],
        },
    }
    thought = {"next_goal": "Submit form"}
    expected_intent = Intent(
        raw="Submit form",
        verb="Submit",
        object="Form",
        summary="Submit form",
        key="form.submit",
        confidence=0.88,
    )

    with patch(
        "graph_agent.mapping.parser.infer_intent_progressive",
        new_callable=AsyncMock,
        return_value=(expected_intent, None, "L0"),
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://a.com/form",
            target_url="https://a.com/done",
        )

    assert [frame.selector for frame in edge.frame_path] == [
        "iframe[name='outer']",
        "iframe[name='inner']",
    ]
    assert edge.element is not None
    assert [frame.name for frame in edge.element.frame_path] == ["outer", "inner"]


@pytest.mark.asyncio
async def test_parse_browser_use_step_extracts_tab_context():
    action = {
        "click": {"element": "button"},
        "interacted_element": {"css_selector": "#quality"},
        "tab_id": "tab-0",
        "target_tab_id": "tab-1",
        "tab_action": "open",
        "tab": {
            "tab_id": "tab-1",
            "opener_tab_id": "tab-0",
            "url": "https://example.com/quality",
            "title": "Quality",
        },
    }
    thought = {"next_goal": "Open quality page in a new tab"}
    expected_intent = Intent(
        raw="Open quality page in a new tab",
        verb="Open",
        object="Quality page",
        summary="Open quality page",
        key="quality.open.page",
        confidence=0.9,
    )

    with patch(
        "graph_agent.mapping.parser.infer_intent_progressive",
        new_callable=AsyncMock,
        return_value=(expected_intent, None, "L0"),
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://example.com/home",
            target_url="https://example.com/quality",
        )

    assert edge.tab_id == "tab-0"
    assert edge.target_tab_id == "tab-1"
    assert edge.tab_action == TabActionType.OPEN
    assert edge.tab is not None
    assert edge.tab.tab_id == "tab-1"
    assert edge.tab.opener_tab_id == "tab-0"
    assert edge.tab.url == "https://example.com/quality"
    assert edge.tab.title == "Quality"

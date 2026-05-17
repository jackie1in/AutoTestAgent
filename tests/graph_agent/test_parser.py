"""Tests for parser AI-only intent inference (T2: no rule fallback)."""

import pytest
from unittest.mock import AsyncMock, patch

from graph_agent.models import ActionType, Intent, TabActionType
from graph_agent.intent.parser import (
    _action_intent_conflict,
    clear_intent_cache,
    infer_intent_for_context,
    infer_intent_progressive,
    parse_browser_use_step,
    parse_browser_use_step_lite,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset intent cache between tests to avoid cross-test pollution."""
    clear_intent_cache()
    yield
    clear_intent_cache()


def _mock_llm_response(content: str):
    """Create response object with .content only (no .completion to avoid _response_to_text using wrong attr)."""
    return type("Response", (), {"content": content})()


@pytest.mark.asyncio
async def test_infer_intent_success():
    """AI returns valid JSON -> intent is returned, no failure reason. Key is empty (set by persistence)."""
    mock_response = _mock_llm_response(
        '{"confidence":0.9,"summary":"Fill username","verb":"Fill","object":"Username field"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.FILL,
            selector="#username",
            source_url="https://a.com",
            target_url="https://a.com",
            param_name="username",
            thought_text="Type user into username",
        )

    assert intent is not None
    assert intent.key == ""  # key is now structural, set later by persistence
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

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
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

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
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
        '{"confidence":0.2,"summary":"Click","verb":"Click","object":"Button"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
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
async def test_infer_intent_failure_missing_summary():
    """LLM returns empty summary -> returns (None, parse_error:missing_summary)."""
    mock_response = _mock_llm_response(
        '{"confidence":0.8,"summary":"","verb":"Click","object":"Button"}'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
        intent, reason = await infer_intent_for_context(
            action=ActionType.CLICK,
            selector="#btn",
            source_url="https://a.com",
            target_url="https://b.com",
            param_name=None,
            thought_text="Click",
        )

    assert intent is None
    assert reason == "parse_error:missing_summary"


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
            "graph_agent.intent.parser.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(None, "parse_error:invalid_json"),
        ),
        patch(
            "graph_agent.intent.parser._infer_param_name",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.intent.parser._extract_action_value",
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
        key="",  # key is structural, set later by persistence
        confidence=0.85,
    )

    with (
        patch(
            "graph_agent.intent.parser.infer_intent_for_context",
            new_callable=AsyncMock,
            return_value=(expected_intent, None),
        ),
        patch(
            "graph_agent.intent.parser._infer_param_name",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "graph_agent.intent.parser._extract_action_value",
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
    assert edge.intent.key == ""
    assert edge.intent.confidence == 0.85


@pytest.mark.asyncio
async def test_infer_intent_progressive_retries_on_low_confidence():
    """Single-pass inference should retry once on low confidence and succeed."""
    low_confidence = (
        None,
        "low_confidence:0.22",
    )
    high_confidence_intent = Intent(
        raw="submit login",
        verb="Submit",
        object="Login form",
        summary="Submit login form",
        key="",
        confidence=0.91,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        side_effect=[low_confidence, (high_confidence_intent, None)],
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
    assert level == "full_retry"
    assert mock_infer.await_count == 2


@pytest.mark.asyncio
async def test_infer_intent_progressive_succeeds_on_first_pass():
    """Single-pass inference should succeed on first call without retry."""
    intent_ok = Intent(
        raw="click login",
        verb="Click",
        object="Login button",
        summary="Click login button",
        key="",
        confidence=0.95,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_for_context",
        new_callable=AsyncMock,
        return_value=(intent_ok, None),
    ) as mock_infer:
        intent, reason, level = await infer_intent_progressive(
            action=ActionType.CLICK,
            selector="#login",
            source_url="https://a.com",
            target_url="https://a.com/secure",
            param_name=None,
            thought_text="click login",
        )

    assert intent is not None
    assert reason is None
    assert level == "full"
    assert mock_infer.await_count == 1


def test_action_intent_conflict_allows_click_navigation_link():
    intent = Intent(
        raw="go login",
        verb="Navigate",
        object="Login page",
        summary="Navigate to login page",
        key="",
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
        key="",
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
        key="",
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
        key="",
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
        key="",
        confidence=0.92,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_for_context",
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
        key="",
        confidence=0.88,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_progressive",
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
        key="",
        confidence=0.9,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_progressive",
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


@pytest.mark.asyncio
async def test_parse_browser_use_step_detects_contenteditable_as_rich_text():
    """input_text on contenteditable element should be parsed as RICH_TEXT."""
    action = {
        "input_text": {"text": "Hello World"},
        "interacted_element": {
            "xpath": "//body",
            "attributes": {"contenteditable": "true"},
        },
    }
    thought = {"next_goal": "Type content in the rich text editor"}

    expected_intent = Intent(
        raw="Type content",
        verb="Type",
        object="Rich text editor",
        summary="Type content in the rich text editor",
        key="",
        confidence=0.9,
    )

    with patch(
        "graph_agent.intent.parser.infer_intent_progressive",
        new_callable=AsyncMock,
        return_value=(expected_intent, None, "L0"),
    ):
        edge = await parse_browser_use_step(
            action=action,
            thought=thought,
            source_url="https://example.com/editor",
            target_url="https://example.com/editor",
        )

    assert edge.action == ActionType.RICH_TEXT
    assert edge.action_value == "Hello World"


def test_is_contenteditable_true_for_contenteditable_attr():
    from graph_agent.intent.parser import _is_contenteditable
    from graph_agent.models import ElementSnapshot

    elem = ElementSnapshot(
        selector="#editor",
        attributes={"contenteditable": "true"},
    )
    assert _is_contenteditable(elem) is True


def test_is_contenteditable_false_for_input():
    from graph_agent.intent.parser import _is_contenteditable
    from graph_agent.models import ElementSnapshot

    elem = ElementSnapshot(
        selector="#username",
        type="input",
        attributes={},
    )
    assert _is_contenteditable(elem) is False


def test_is_contenteditable_true_for_tinymce_selector():
    from graph_agent.intent.parser import _is_contenteditable
    from graph_agent.models import ElementSnapshot

    elem = ElementSnapshot(
        selector="#tinymce",
        attributes={},
    )
    assert _is_contenteditable(elem) is True


def test_is_contenteditable_true_for_ql_editor_class():
    from graph_agent.intent.parser import _is_contenteditable
    from graph_agent.models import ElementSnapshot

    elem = ElementSnapshot(
        selector=".ql-editor",
        attributes={"class": "ql-editor"},
    )
    assert _is_contenteditable(elem) is True


# ---------------------------------------------------------------------------
# Phase 1/2 optimization tests
# ---------------------------------------------------------------------------


def test_parse_browser_use_step_lite_no_llm():
    """Lite parser should extract action/selector/element without LLM, intent=None."""
    action = {
        "click": {"element": "button"},
        "interacted_element": {"xpath": "//button[@id='submit']"},
    }
    thought = {"next_goal": "Submit the form"}

    edge = parse_browser_use_step_lite(
        action=action,
        thought=thought,
        source_url="https://a.com/form",
        target_url="https://a.com/done",
    )

    assert edge.intent is None
    assert edge.intent_failure_reason == "pending"
    assert edge.selector == "xpath=//button[@id='submit']"
    assert edge.action == ActionType.CLICK
    assert edge.source == "https://a.com/form"
    assert edge.target == "https://a.com/done"


def test_parse_browser_use_step_lite_fill_extracts_param_and_value():
    """Lite parser should extract param_name and action_value for fill actions."""
    action = {
        "input_text": {"text": "admin"},
        "interacted_element": {
            "css_selector": "input[name='user']",
            "attributes": {"name": "user", "type": "text"},
        },
    }
    thought = {"next_goal": "Fill username"}

    edge = parse_browser_use_step_lite(
        action=action,
        thought=thought,
        source_url="https://a.com/login",
        target_url="https://a.com/login",
    )

    assert edge.action == ActionType.FILL
    assert edge.param_name == "user"
    assert edge.action_value == "admin"
    assert edge.intent is None


@pytest.mark.asyncio
async def test_intent_cache_hit():
    """Second call with same inputs should return cached intent without LLM call."""
    mock_response = _mock_llm_response(
        '{"confidence":0.9,"summary":"Fill user","verb":"Fill","object":"User"}'
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("graph_agent.intent.parser.get_llm", return_value=mock_llm):
        intent1, _ = await infer_intent_for_context(
            action=ActionType.FILL, selector="#user",
            source_url="https://a.com/login", target_url="https://a.com/login",
            param_name="user", thought_text="fill user",
        )
        intent2, _ = await infer_intent_for_context(
            action=ActionType.FILL, selector="#user",
            source_url="https://a.com/login", target_url="https://a.com/login",
            param_name="user", thought_text="fill user",
        )

    assert intent1 is not None
    assert intent2 is intent1
    assert mock_llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_skip_distill_env(monkeypatch):
    """MAPPING_SKIP_DISTILL=true should skip the thought distillation LLM call."""
    monkeypatch.setenv("MAPPING_SKIP_DISTILL", "true")
    from graph_agent.intent.parser import distill_ui_thought

    result = await distill_ui_thought(
        thought_text="Write the report to csv file and then fill username",
        action=ActionType.FILL,
        selector="#user",
        source_url="https://a.com",
        target_url="https://a.com",
    )

    assert "csv" in result
    assert "fill" in result.lower()

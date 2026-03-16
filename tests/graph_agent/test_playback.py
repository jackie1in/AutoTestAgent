"""Tests for playback engine (T7: null intent compatibility)."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_agent.models import (
    ActionType,
    FrameLocatorSnapshot,
    GraphEdge,
    Intent,
    TabActionType,
)
from graph_agent.playback.engine import (
    _extract_login_error_message,
    _extract_login_error_message_from_page_text,
    _format_replay_error,
    _is_closed_context_error,
    _is_transient_error,
    run_playback,
)


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


# data URL with inline HTML - loads without network, has #btn and #user elements
_DATA_HTML = (
    "data:text/html,"
    "<html><body>"
    "<button id='btn'>click</button>"
    "<input id='user' type='text' placeholder='username'/>"
    "</body></html>"
)


def test_extract_login_error_message_from_failed_login_payload():
    """Login error parser should return the first backend error message when login fails."""
    payload = (
        '{"code":200,"data":{"passwordDefaultNum":1},'
        '"errors":[{"errorCode":"401","msg":"用户帐号已过期 "}],'
        '"serviceSuccess":false}'
    )

    assert _extract_login_error_message(payload) == "用户帐号已过期"


def test_extract_login_error_message_returns_none_for_success_payload():
    """Login error parser should ignore successful login payloads."""
    payload = '{"code":200,"data":{"token":"abc"},"errors":[],"serviceSuccess":true}'

    assert _extract_login_error_message(payload) is None


def test_extract_login_error_message_from_page_text():
    """Rendered page messages should expose login failure reason."""
    page_text = "系统登录\n立即登录\n用户帐号已过期\nTa+3 404开发框架"

    assert _extract_login_error_message_from_page_text(page_text) == "用户帐号已过期"


class _FakeRequest:
    def __init__(self, url: str, resource_type: str = "xhr") -> None:
        self.url = url
        self.resource_type = resource_type


class _FakeLocator:
    def __init__(self, page: "_FakePage", selector: str) -> None:
        self.page = page
        self.selector = selector

    async def click(self) -> None:
        self.page.events.append(("click", self.selector))
        hook = self.page.click_hooks.get(self.selector)
        if hook is not None:
            await hook(self.page)

    async def fill(self, value: str) -> None:
        self.page.events.append(("fill", self.selector, value))

    async def wait_for(self, *, state: str = "attached", timeout: int = 30_000) -> None:
        self.page.events.append(("wait-locator", self.selector, state))


class _FakeFrameContext:
    def __init__(self, page: "_FakePage") -> None:
        self.page = page

    def frame_locator(self, selector: str) -> "_FakeFrameContext":
        if selector in self.page.missing_frames:
            raise RuntimeError(f"missing frame: {selector}")
        self.page.events.append(("frame", selector))
        return _FakeFrameContext(self.page)

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self.page, selector)


class _FakeBrowserContext:
    def __init__(self, page: "_FakePage") -> None:
        self._page = page

    @property
    def pages(self) -> list["_FakePage"]:
        if self._page.closed:
            return []
        return [self._page]


class _FakePage:
    def __init__(self, missing_frames: set[str] | None = None) -> None:
        self.url = ""
        self.events: list[tuple[str, ...]] = []
        self.goto_hooks: dict[str, object] = {}
        self.click_hooks: dict[str, object] = {}
        self.popup_page: _FakePage | None = None
        self.closed = False
        self.missing_frames = missing_frames or set()
        self._handlers: dict[str, list[object]] = {
            "request": [],
            "requestfinished": [],
            "requestfailed": [],
        }
        self.context = _FakeBrowserContext(self)

    def set_default_timeout(self, timeout_ms: int) -> None:
        self.events.append(("set-timeout", str(timeout_ms)))

    async def goto(self, url: str) -> None:
        self.url = url
        self.events.append(("goto", url))
        hook = self.goto_hooks.get(url)
        if hook is not None:
            await hook(self)

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    def frame_locator(self, selector: str) -> _FakeFrameContext:
        if selector in self.missing_frames:
            raise RuntimeError(f"missing frame: {selector}")
        self.events.append(("frame", selector))
        return _FakeFrameContext(self)

    def on(self, event: str, handler: object) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: object) -> None:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, request: _FakeRequest) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(request)

    def expect_popup(self) -> "_FakeExpectPopup":
        return _FakeExpectPopup(self)

    async def wait_for_selector(
        self, selector: str, *, state: str = "attached", timeout: int = 30_000
    ) -> None:
        self.events.append(("wait-selector", selector, state))

    async def wait_for_load_state(
        self, state: str = "load", timeout: int = 30_000
    ) -> None:
        self.events.append(("wait-load", state))

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True
        self.events.append(("close",))


class _FakePopupInfo:
    def __init__(self, page: _FakePage | None) -> None:
        self.value = asyncio.Future()
        if page is not None:
            self.value.set_result(page)
        else:
            self.value.set_exception(RuntimeError("popup was not opened"))


class _FakeExpectPopup:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.info: _FakePopupInfo | None = None

    async def __aenter__(self) -> _FakePopupInfo:
        self.info = _FakePopupInfo(self.page.popup_page)
        return self.info

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeExpectation:
    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def to_have_url(self, expected_url: str) -> None:
        if self.page.url != expected_url:
            raise AssertionError(f"expected {expected_url}, got {self.page.url}")


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def new_page(self) -> _FakePage:
        return self.page

    async def close(self) -> None:
        self.page.events.append(("browser-close",))


class _FakeAsyncPlaywright:
    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(
            chromium=SimpleNamespace(
                launch=AsyncMock(return_value=_FakeBrowser(self.page))
            )
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _install_fake_playwright(monkeypatch: pytest.MonkeyPatch, page: _FakePage) -> None:
    monkeypatch.setattr(
        "graph_agent.playback.engine.async_playwright",
        lambda: _FakeAsyncPlaywright(page),
    )


def _install_fake_expect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "graph_agent.playback.engine.expect",
        lambda page: _FakeExpectation(page),
    )


@pytest.mark.asyncio
async def test_playback_null_intent_log_not_crash():
    """含空意图边的路径回放不因日志字段报错 (T7)."""
    # Ensure headless for CI
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        edge_list = [
            GraphEdge(
                source="a",
                target="b",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="parse failed",
            ),
        ]
        logs = []

        async def capture_log(entry):
            logs.append(entry)

        result = await run_playback(
            edge_list,
            test_data={},
            start_url=_DATA_HTML,
            log_callback=capture_log,
        )

        assert result["success"] is True
        assert len(logs) == 1
        assert logs[0]["intent"] == ""
        assert logs[0]["selector"] == "#btn"
        assert logs[0]["action"] == ActionType.CLICK
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_playback_mixed_intents_log_correct():
    """混合空意图与有效意图边时，日志字段正确 (T7)."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        intent = _make_intent("Fill username", key="fill_username")
        edge_list = [
            GraphEdge(
                source="a",
                target="b",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="x",
            ),
            GraphEdge(
                source="b",
                target="c",
                selector="#user",
                action=ActionType.FILL,
                intent=intent,
                param_name="username",
                action_value="recorded-user",
            ),
        ]
        logs = []

        async def capture_log(entry):
            logs.append(entry)

        result = await run_playback(
            edge_list,
            test_data={"username": "testuser"},
            start_url=_DATA_HTML,
            log_callback=capture_log,
        )

        assert result["success"] is True
        assert len(logs) == 2
        assert logs[0]["intent"] == ""
        assert logs[0]["selector"] == "#btn"
        assert logs[1]["intent"] == "Fill username"
        assert logs[1]["selector"] == "#user"
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_playback_falls_back_to_param_name_when_recorded_value_missing():
    """When action_value is missing, playback should use test_data[param_name]."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        edge_list = [
            GraphEdge(
                source="a",
                target="b",
                selector="#user",
                action=ActionType.FILL,
                intent=_make_intent("Fill username", key="fill_username"),
                param_name="username",
                action_value=None,
            ),
        ]

        result = await run_playback(
            edge_list,
            test_data={"username": "testuser"},
            start_url=_DATA_HTML,
        )

        assert result["success"] is True
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_playback_prefers_test_data_over_recorded_action_value(
    monkeypatch: pytest.MonkeyPatch,
):
    """Runtime test_data should override stale recorded action_value for fill steps."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
            action_value="recorded-user",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "testuser"},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("fill", "#user", "testuser") in page.events


@pytest.mark.asyncio
async def test_playback_click_uses_nested_frame_path(monkeypatch: pytest.MonkeyPatch):
    """Nested iframe steps should resolve frame context before clicking target element."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert page.events == [
        ("set-timeout", str(60_000)),
        ("goto", "https://a.com/start"),
        ("wait-selector", "iframe[name='outer']", "attached"),
        ("frame", "iframe[name='outer']"),
        ("wait-locator", ":root", "attached"),
        ("frame", "iframe[name='outer']"),
        ("frame", "iframe[name='inner']"),
        ("click", "#submit"),
        ("browser-close",),
    ]


@pytest.mark.asyncio
async def test_playback_reports_missing_iframe_level(monkeypatch: pytest.MonkeyPatch):
    """Missing iframe errors should report the failing nesting level."""
    page = _FakePage(missing_frames={"iframe[name='inner']"})
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "iframe level 2" in (result["error"] or "")


@pytest.mark.asyncio
async def test_playback_open_popup_tab_routes_followup_click_to_new_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """OPEN 新标签后，后续普通动作应在 target tab 对应的新 page 上执行。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _confirm_click(fake_page: _FakePage) -> None:
        fake_page.url = "https://a.com/popup/confirmed"

    popup.click_hooks["#confirm"] = _confirm_click
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#quality",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#quality") in page.events
    assert ("click", "#confirm") in popup.events
    assert ("click", "#confirm") not in page.events
    assert result["actual_url"] == "https://a.com/popup/confirmed"


@pytest.mark.asyncio
async def test_playback_expected_end_url_uses_popup_page_after_open(
    monkeypatch: pytest.MonkeyPatch,
):
    """expected_end_url 应对最后实际动作所在的 popup page 做断言。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _confirm_click(fake_page: _FakePage) -> None:
        fake_page.url = "https://a.com/popup/confirmed"

    popup.click_hooks["#confirm"] = _confirm_click
    _install_fake_playwright(monkeypatch, page)
    _install_fake_expect(monkeypatch)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#quality",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        expected_end_url="https://a.com/popup/confirmed",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert result["actual_url"] == "https://a.com/popup/confirmed"


@pytest.mark.asyncio
async def test_playback_switch_tab_updates_active_page_for_expected_end_url(
    monkeypatch: pytest.MonkeyPatch,
):
    """SWITCH 到已存在标签后，expected_end_url 应对切换后的 page 生效。"""
    page = _FakePage()
    popup = _FakePage()
    popup_url = "https://a.com/popup"
    popup.url = popup_url
    page.popup_page = popup

    async def _mutate_homepage(fake_page: _FakePage) -> None:
        fake_page.url = "https://a.com/home/changed"

    page.click_hooks["#change-home"] = _mutate_homepage
    _install_fake_playwright(monkeypatch, page)
    _install_fake_expect(monkeypatch)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#change-home",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="",
            action=ActionType.UNKNOWN,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.SWITCH,
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        expected_end_url=popup_url,
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("click", "#change-home") in page.events
    assert result["actual_url"] == popup_url


@pytest.mark.asyncio
async def test_playback_switch_tab_routes_followup_click_to_switched_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """SWITCH 后，后续仍带 source tab_id 的动作也应落到切换后的 page。"""
    page = _FakePage()
    popup = _FakePage()
    page.popup_page = popup
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="",
            action=ActionType.UNKNOWN,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.SWITCH,
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#confirm") in popup.events
    assert ("click", "#confirm") not in page.events


@pytest.mark.asyncio
async def test_playback_tab_and_nested_iframe_routes_followup_click_to_popup_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """tab + nested iframe 叠加时，frame 解析与点击都应落在 popup page 上。"""
    page = _FakePage(
        missing_frames={"iframe[name='popup-outer']", "iframe[name='popup-inner']"}
    )
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _confirm_click(fake_page: _FakePage) -> None:
        fake_page.url = "https://a.com/popup/frame-confirmed"

    popup.click_hooks["#confirm"] = _confirm_click
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='popup-outer']"),
                FrameLocatorSnapshot(selector="iframe[name='popup-inner']"),
            ],
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("frame", "iframe[name='popup-outer']") not in page.events
    assert ("frame", "iframe[name='popup-inner']") not in page.events
    assert ("click", "#confirm") not in page.events
    assert popup.events == [
        ("set-timeout", str(60_000)),
        ("wait-load", "domcontentloaded"),
        ("wait-selector", "iframe[name='popup-outer']", "attached"),
        ("frame", "iframe[name='popup-outer']"),
        ("wait-locator", ":root", "attached"),
        ("frame", "iframe[name='popup-outer']"),
        ("frame", "iframe[name='popup-inner']"),
        ("click", "#confirm"),
        ("wait-load", "networkidle"),
    ]
    assert result["actual_url"] == "https://a.com/popup/frame-confirmed"


def test_format_replay_error_tab():
    """Tab errors should be prefixed with 'tab:'."""
    err = ValueError("Target tab does not exist: tab-x")
    edge = GraphEdge(source="a", target="b", selector="#btn", action=ActionType.CLICK)
    assert (
        _format_replay_error(err, edge, "tab-x")
        == "tab: Target tab does not exist: tab-x"
    )


def test_format_replay_error_iframe():
    """Iframe errors should be prefixed with 'iframe:'."""
    err = ValueError("Failed to locate iframe level 2: iframe[name='inner']")
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
        frame_path=[
            FrameLocatorSnapshot(selector="iframe[name='outer']"),
            FrameLocatorSnapshot(selector="iframe[name='inner']"),
        ],
    )
    assert "iframe:" in _format_replay_error(err, edge, "tab-0")


def test_format_replay_error_selector_with_context():
    """Selector errors should include tab, frame_path, selector context."""
    err = RuntimeError("Locator timed out: waiting for selector '#missing'")
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#missing",
        action=ActionType.CLICK,
        tab_id="tab-1",
        frame_path=[FrameLocatorSnapshot(selector="iframe#f1")],
    )
    msg = _format_replay_error(err, edge, "tab-1")
    assert "selector:" in msg
    assert "tab=tab-1" in msg
    assert "frame_path=level1=iframe#f1" in msg
    assert "selector=#missing" in msg


@pytest.mark.asyncio
async def test_playback_reports_missing_target_tab(monkeypatch: pytest.MonkeyPatch):
    """引用不存在的 tab_id 时，应返回明确错误而不是裸 KeyError。"""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-x",
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert result["error"] == "tab: Target tab does not exist: tab-x"


@pytest.mark.asyncio
async def test_playback_reports_selector_not_found_with_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """Selector not found should return error with tab/frame_path/selector context."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _raise_selector_error(fake_page: _FakePage) -> None:
        raise RuntimeError("Locator timed out: waiting for selector '#nonexistent'")

    page.click_hooks["#nonexistent"] = _raise_selector_error

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#nonexistent",
            action=ActionType.CLICK,
            tab_id="tab-0",
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "selector:" in result["error"]
    assert "tab=tab-0" in result["error"]
    assert "selector=#nonexistent" in result["error"]


@pytest.mark.asyncio
async def test_playback_close_child_tab_restores_parent_for_followup_click(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLOSE 子标签后，后续普通动作应恢复到父标签页执行。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _parent_after_close(fake_page: _FakePage) -> None:
        if not popup.closed:
            raise RuntimeError("popup still open; close/recover logic missing")
        fake_page.url = "https://a.com/parent/after-close"

    page.click_hooks["#back-on-parent"] = _parent_after_close
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="",
            action=ActionType.CLICK,
            tab_id="tab-1",
            tab_action=TabActionType.CLOSE,
        ),
        GraphEdge(
            source="d",
            target="e",
            selector="#back-on-parent",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("click", "#confirm") in popup.events
    assert ("close",) in popup.events
    assert ("click", "#back-on-parent") in page.events
    assert result["actual_url"] == "https://a.com/parent/after-close"


@pytest.mark.asyncio
async def test_playback_close_child_tab_falls_back_to_parent_when_followup_still_uses_child_tab_id(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLOSE 子标签后，后续仍写 child tab_id 也应自动回落到父标签页执行。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _parent_after_close(fake_page: _FakePage) -> None:
        if not popup.closed:
            raise RuntimeError("popup still open; close/recover logic missing")
        fake_page.url = "https://a.com/parent/fallback-after-close"

    page.click_hooks["#back-on-parent"] = _parent_after_close
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="",
            action=ActionType.CLICK,
            tab_id="tab-1",
            tab_action=TabActionType.CLOSE,
        ),
        GraphEdge(
            source="d",
            target="e",
            selector="#back-on-parent",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("click", "#confirm") in popup.events
    assert ("close",) in popup.events
    assert ("click", "#back-on-parent") in page.events
    assert ("click", "#back-on-parent") not in popup.events
    assert result["actual_url"] == "https://a.com/parent/fallback-after-close"


@pytest.mark.asyncio
async def test_playback_fails_fast_when_login_submit_does_not_leave_login_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """When auth submit keeps user on login page, playback should fail before downstream steps."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="login",
            target="login-user",
            selector="#username",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
        GraphEdge(
            source="login-user",
            target="login-pass",
            selector="#password",
            action=ActionType.FILL,
            intent=_make_intent("Fill password", key="auth.fill.password"),
            param_name="password",
        ),
        GraphEdge(
            source="login-pass",
            target="secure",
            source_url="https://example.com/login",
            target_url="https://example.com/secure",
            selector="#submit",
            action=ActionType.CLICK,
            intent=_make_intent("Submit login", key="auth.submit.login"),
        ),
        GraphEdge(
            source="secure",
            target="dashboard",
            selector="#dashboard-entry",
            action=ActionType.CLICK,
            intent=_make_intent("Open dashboard", key="project.dashboard.open"),
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice", "password": "secret"},
        start_url="https://example.com/login",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "login" in (result["error"] or "").lower()
    # Guard should stop playback before first post-login business click.
    assert ("click", "#dashboard-entry") not in page.events


@pytest.mark.asyncio
async def test_playback_navigate_waits_for_http_requests_before_next_action(
    monkeypatch: pytest.MonkeyPatch,
):
    """NAVIGATE should wait for spawned HTTP requests before continuing."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _navigate_hook(fake_page: _FakePage) -> None:
        req = _FakeRequest("https://a.com/bootstrap")
        fake_page.events.append(("request-start", req.url))
        fake_page.emit("request", req)

        async def _finish() -> None:
            await asyncio.sleep(0.01)
            fake_page.events.append(("request-finish", req.url))
            fake_page.emit("requestfinished", req)

        asyncio.create_task(_finish())

    page.goto_hooks["https://a.com/secure"] = _navigate_hook

    edge_list = [
        GraphEdge(
            source="state-start",
            target="https://a.com/secure",
            selector="",
            action=ActionType.NAVIGATE,
            intent=_make_intent("Navigate to secure page", key="navigation.secure"),
        ),
        GraphEdge(
            source="state-secure",
            target="state-filled",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert ("goto", "https://a.com/secure") in page.events
    assert page.events.index(
        ("request-finish", "https://a.com/bootstrap")
    ) < page.events.index(("fill", "#user", "alice"))


@pytest.mark.asyncio
async def test_playback_click_waits_for_triggered_http_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLICK should wait for its HTTP request batch before the next step."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _click_hook(fake_page: _FakePage) -> None:
        req = _FakeRequest("https://a.com/api/projects")
        fake_page.events.append(("request-start", req.url))
        fake_page.emit("request", req)

        async def _finish() -> None:
            await asyncio.sleep(0.01)
            fake_page.events.append(("request-finish", req.url))
            fake_page.emit("requestfinished", req)

        asyncio.create_task(_finish())

    page.click_hooks["#btn"] = _click_hook

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=_make_intent("Open projects", key="project.open"),
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert page.events.index(
        ("request-finish", "https://a.com/api/projects")
    ) < page.events.index(("fill", "#user", "alice"))


@pytest.mark.asyncio
async def test_playback_no_http_request_continues_without_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """When an action triggers no HTTP request, playback should continue normally."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=_make_intent("Open modal", key="modal.open"),
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert page.events.index(("click", "#btn")) < page.events.index(
        ("fill", "#user", "alice")
    )


def test_is_transient_error_timeout():
    """TimeoutError should be classified as transient."""
    assert _is_transient_error(TimeoutError("timed out")) is True
    assert _is_transient_error(asyncio.TimeoutError()) is True


def test_is_transient_error_locator_message():
    """Locator timeout messages should be classified as transient."""
    assert (
        _is_transient_error(RuntimeError("Locator timed out: waiting for selector"))
        is True
    )
    assert _is_transient_error(RuntimeError("Timeout 30000ms exceeded")) is True


def test_is_transient_error_non_transient():
    """Non-timeout errors should not be classified as transient."""
    assert _is_transient_error(ValueError("Target tab does not exist")) is False
    assert _is_transient_error(RuntimeError("missing frame: iframe#x")) is False


def test_is_transient_error_excludes_closed_context():
    """Closed-context errors must NOT be transient — they need recovery, not retry."""
    assert (
        _is_transient_error(
            RuntimeError(
                "Locator.click: Target page, context or browser has been closed"
            )
        )
        is False
    )
    assert _is_transient_error(RuntimeError("target closed")) is False


def test_format_replay_error_async_load_prefix():
    """Timeout/async errors should be prefixed with async_load."""
    err = RuntimeError("Locator timed out: waiting for selector '#btn'")
    edge = GraphEdge(source="a", target="b", selector="#btn", action=ActionType.CLICK)
    msg = _format_replay_error(err, edge, "tab-0")
    assert msg.startswith("async_load: ")
    assert "selector:" in msg
    assert "tab=tab-0" in msg


@pytest.mark.asyncio
async def test_playback_retries_on_transient_click_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Click should succeed on retry after transient TimeoutError."""
    page = _FakePage()
    attempt = [0]

    async def _flaky_click(fake_page: _FakePage) -> None:
        attempt[0] += 1
        if attempt[0] < 2:
            raise TimeoutError("Locator timed out")
        fake_page.events.append(("click", "#btn"))

    page.click_hooks["#btn"] = _flaky_click
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert attempt[0] == 2


@pytest.mark.asyncio
async def test_playback_retries_on_transient_fill_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fill should succeed on retry after transient failure."""
    page = _FakePage()
    attempt = [0]

    class _FakeLocatorWithFlakyFill(_FakeLocator):
        async def fill(self, value: str) -> None:
            attempt[0] += 1
            if attempt[0] < 2:
                raise RuntimeError("Locator timed out: waiting for selector")
            self.page.events.append(("fill", self.selector, value))

    def _locator_with_flaky_fill(selector: str):
        return _FakeLocatorWithFlakyFill(page, selector)

    page.locator = _locator_with_flaky_fill
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#user",
            action=ActionType.FILL,
            param_name="username",
            action_value="alice",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert attempt[0] == 2
    assert ("fill", "#user", "alice") in page.events


@pytest.mark.asyncio
async def test_playback_element_selector_fallback_prefers_id_over_xpath(
    monkeypatch: pytest.MonkeyPatch,
):
    """When element has id, prefer #id over xpath (PRD selector stability)."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    from graph_agent.models import ElementSnapshot

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="xpath=html/body/div/form/input",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
            element=ElementSnapshot(
                selector="xpath=html/body/div/form/input",
                id="user",
                name="username",
                attributes={"id": "user", "name": "username"},
            ),
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "testuser"},
        start_url=_DATA_HTML,
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("fill", "#user", "testuser") in page.events


@pytest.mark.asyncio
async def test_playback_frame_selector_fallback_uses_css_when_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """When primary frame selector fails, fall back to css_selector."""
    page = _FakePage(missing_frames={"iframe[name='outer']"})
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[
                FrameLocatorSnapshot(
                    selector="iframe[name='outer']",
                    css_selector="iframe#frame-outer",
                ),
            ],
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("frame", "iframe#frame-outer") in page.events
    assert ("frame", "iframe[name='outer']") not in page.events


def test_is_closed_context_error_matches():
    """Closed-context errors should be detected."""
    assert (
        _is_closed_context_error(
            RuntimeError("Locator.click: Target page, context or browser has been closed")
        )
        is True
    )
    assert _is_closed_context_error(RuntimeError("target closed")) is True
    assert _is_closed_context_error(RuntimeError("something else")) is False
    assert _is_closed_context_error(ValueError("normal error")) is False


class _FakeMultiPageBrowserContext:
    """Browser context that tracks multiple pages for closed-context recovery tests."""

    def __init__(self, page_list: list["_FakePage"]) -> None:
        self._page_list = page_list

    @property
    def pages(self) -> list["_FakePage"]:
        return [p for p in self._page_list if not p.is_closed()]


@pytest.mark.asyncio
async def test_playback_recovers_from_closed_context_on_click(
    monkeypatch: pytest.MonkeyPatch,
):
    """When click raises 'has been closed', playback should recover to surviving page and retry."""
    page = _FakePage()
    recovery_page = _FakePage()
    recovery_page.url = "https://a.com/start"

    multi_ctx = _FakeMultiPageBrowserContext([page, recovery_page])
    page.context = multi_ctx

    async def _stale_click(fake_page: _FakePage) -> None:
        fake_page.closed = True
        raise RuntimeError(
            "Locator.click: Target page, context or browser has been closed"
        )

    page.click_hooks["#confirm"] = _stale_click
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#confirm") in recovery_page.events


@pytest.mark.asyncio
async def test_playback_recovers_from_closed_context_on_fill(
    monkeypatch: pytest.MonkeyPatch,
):
    """When fill raises 'has been closed', playback should recover to surviving page and retry."""
    page = _FakePage()
    recovery_page = _FakePage()
    recovery_page.url = "https://a.com/start"

    multi_ctx = _FakeMultiPageBrowserContext([page, recovery_page])
    page.context = multi_ctx

    class _ClosableLocator(_FakeLocator):
        async def fill(self, value: str) -> None:
            self.page.closed = True
            raise RuntimeError(
                "Locator.fill: Target page, context or browser has been closed"
            )

    page.locator = lambda selector: _ClosableLocator(page, selector)
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#user",
            action=ActionType.FILL,
            param_name="username",
            action_value="alice",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("fill", "#user", "alice") in recovery_page.events


@pytest.mark.asyncio
async def test_playback_ensure_frame_attached_absorbs_stale_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    """When iframe frame_locator is stale on first call, _ensure_frame_attached
    should absorb it so the real action succeeds without hitting recovery."""
    page = _FakePage()
    multi_ctx = _FakeMultiPageBrowserContext([page])
    page.context = multi_ctx

    attempt = [0]
    original_frame_locator = page.frame_locator

    def _frame_locator_with_stale_first(selector: str) -> _FakeFrameContext:
        attempt[0] += 1
        if attempt[0] == 1:
            raise RuntimeError(
                "Locator.click: Target page, context or browser has been closed"
            )
        return original_frame_locator(selector)

    page.frame_locator = _frame_locator_with_stale_first
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#menu-item",
            action=ActionType.CLICK,
            tab_id="tab-0",
            frame_path=[
                FrameLocatorSnapshot(
                    selector="xpath=/html/body/div/iframe"
                ),
            ],
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#menu-item") in page.events
    assert any(
        ev[0] == "wait-selector" and "iframe" in ev[1] for ev in page.events
    )


@pytest.mark.asyncio
async def test_playback_recovers_closed_iframe_context_with_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    """When click raises 'has been closed' (e.g. iframe navigation during click),
    recovery should wait for page load + iframe then retry."""
    page = _FakePage()
    multi_ctx = _FakeMultiPageBrowserContext([page])
    page.context = multi_ctx

    click_attempt = [0]

    async def _stale_first_click(fake_page: _FakePage) -> None:
        click_attempt[0] += 1
        if click_attempt[0] == 1:
            raise RuntimeError(
                "Locator.click: Target page, context or browser has been closed"
            )

    page.click_hooks["#menu-item"] = _stale_first_click
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#menu-item",
            action=ActionType.CLICK,
            tab_id="tab-0",
            frame_path=[
                FrameLocatorSnapshot(
                    selector="xpath=/html/body/div/iframe"
                ),
            ],
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert click_attempt[0] == 2
    assert ("wait-load", "domcontentloaded") in page.events


@pytest.mark.asyncio
async def test_playback_closed_context_no_recovery_page_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """When 'has been closed' and no surviving page, playback should fail gracefully."""
    page = _FakePage()
    multi_ctx = _FakeMultiPageBrowserContext([page])
    page.context = multi_ctx

    async def _stale_click(fake_page: _FakePage) -> None:
        fake_page.closed = True
        raise RuntimeError(
            "Locator.click: Target page, context or browser has been closed"
        )

    page.click_hooks["#btn"] = _stale_click
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "has been closed" in (result["error"] or "").lower()


@pytest.mark.asyncio
async def test_playback_popup_waits_for_load_state(monkeypatch: pytest.MonkeyPatch):
    """After OPEN, playback should wait for popup domcontentloaded when available."""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    load_state_called = [False]

    async def _fake_wait_for_load_state(state: str) -> None:
        load_state_called[0] = True
        assert state == "domcontentloaded"

    popup.wait_for_load_state = _fake_wait_for_load_state
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert load_state_called[0] is True

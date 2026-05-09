"""Tests for playback engine (T7: null intent compatibility)."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_agent.models import (
    ActionType,
    Intent,
    FrameLocatorSnapshot,
    GraphEdge,
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


class _FakeKeyboard:
    def __init__(self, page: "_FakePage") -> None:
        self._page = page

    async def press(self, key: str) -> None:
        self._page.events.append(("keyboard-press", key))

    async def type(self, text: str) -> None:
        self._page.events.append(("keyboard-type", text))


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

    async def select_option(self, value: str) -> None:
        self.page.events.append(("select_option", self.selector, value))

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
        self.keyboard = _FakeKeyboard(self)

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


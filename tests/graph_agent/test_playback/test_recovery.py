from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph_agent.lib.types import ActionResult
from graph_agent.models import Intent, GraphEdge, ElementConstraints, FrameLocatorSnapshot
from graph_agent.playback.engine.helpers import (
    _is_closed_context_error,
    _is_transient_error,
)

from tests.graph_agent.test_playback.fakes import (
    _FakeAsyncPlaywright,
    _FakeBrowser,
    _FakeBrowserContext,
    _FakeExpectPopup,
    _FakeExpectation,
    _FakeFrameContext,
    _FakeKeyboard,
    _FakeLocator,
    _FakeMultiPageBrowserContext,
    _FakePage,
    _FakePopupInfo,
    _FakeRequest,
    _install_fake_expect,
    _install_fake_playwright,
    _make_intent,
)



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


@pytest.mark.asyncio
async def test_playback_rich_text_uses_click_selectall_type(monkeypatch):
    """RICH_TEXT action should click, Ctrl+A, then keyboard.type."""
    page = _FakePage()
    page.url = "https://editor.com/"

    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="a",
            selector="#tinymce",
            action=ActionType.RICH_TEXT,
            frame_path=[FrameLocatorSnapshot(selector="#mce_0_ifr")],
            param_name="editor_content",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"editor_content": "Hello <b>World</b>"},
        start_url="https://editor.com/",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#tinymce") in page.events
    assert ("keyboard-press", "Control+a") in page.events
    assert ("keyboard-type", "Hello <b>World</b>") in page.events


@pytest.mark.asyncio
async def test_playback_rich_text_fallback_to_action_value(monkeypatch):
    """RICH_TEXT should use action_value when test_data has no matching key."""
    page = _FakePage()
    page.url = "https://editor.com/"

    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="a",
            selector="#editor",
            action=ActionType.RICH_TEXT,
            action_value="Recorded text",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://editor.com/",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("keyboard-type", "Recorded text") in page.events

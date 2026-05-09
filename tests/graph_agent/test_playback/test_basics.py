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



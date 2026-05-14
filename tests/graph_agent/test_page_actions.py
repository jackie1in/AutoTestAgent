"""Unit tests for PageActions — custom action handlers that browser-use doesn't provide.

Tests cover: W3C click, scroll_horizontally, close_overlay, query_knowledge,
discover_zones, extract_menu, solve_captcha.
"""

from __future__ import annotations

import pytest
from typing import cast

from browser_use.llm.base import BaseChatModel
from browser_use.tools.registry.service import Registry

from graph_agent.cartography.react_explorer.page_actions import PageActions
from graph_agent.lib.page_controller import PageController


class _FakeResult:
    def __init__(self, message: str):
        self.message = message


class _FakeController:
    """Minimal fake that records calls."""

    def __init__(self) -> None:
        self.click_calls: list[int] = []

    async def click_element(self, index: int) -> _FakeResult:
        self.click_calls.append(index)
        return _FakeResult(f"clicked [{index}]")

    async def scroll_horizontally(
        self, direction: str, amount: int, index: int | None
    ) -> _FakeResult:
        return _FakeResult(f"hscroll {direction} {amount}px")


# ---------------------------------------------------------------------------
# W3C click
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_delegates_to_controller():
    ctrl = _FakeController()
    pa = PageActions(cast(PageController, ctrl), None, cast(BaseChatModel, None), [0.0])
    result = await pa.click(5)
    assert result == "clicked [5]"
    assert ctrl.click_calls == [5]


@pytest.mark.asyncio
async def test_click_without_controller_raises():
    pa = PageActions(None, None, cast(BaseChatModel, None), [0.0])
    with pytest.raises(RuntimeError, match="PageController not set"):
        await pa.click(1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_populates_registry():
    registry = Registry()
    supported: set[str] = set()
    pa = PageActions(None, None, cast(BaseChatModel, object()), [0.0])
    pa.register(registry, supported)

    assert len(supported) == 9
    for name in (
        "click",
        "input",
        "dropdown_options",
        "scroll_horizontally",
        "close_overlay",
        "query_knowledge",
        "discover_zones",
        "extract_menu",
        "solve_captcha",
    ):
        assert name in supported, f"{name} missing"
        assert name in registry.registry.actions, f"{name} not in registry"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_overlay_without_browser():
    pa = PageActions(None, None, cast(BaseChatModel, None), [0.0])
    result = await pa.close_overlay()
    assert "no browser" in result


@pytest.mark.asyncio
async def test_discover_zones_graceful_without_controller():
    pa = PageActions(None, None, cast(BaseChatModel, None), [0.0])
    result = await pa.discover_zones()
    assert "no page controller" in result.lower()

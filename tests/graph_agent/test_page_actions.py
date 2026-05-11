"""Unit tests for PageActions — action handlers are independently testable.

These tests verify that each handler delegates correctly to PageController,
that register() populates a browser-use Registry, and that edge cases
(no controller, no browser) are handled gracefully.
"""

from __future__ import annotations

import pytest

from browser_use.tools.registry.service import Registry

from graph_agent.cartography.react_explorer.page_actions import PageActions


class _FakeResult:
    def __init__(self, message: str):
        self.message = message


class _FakeController:
    """Minimal fake that records calls."""

    def __init__(self) -> None:
        self.click_calls: list[int] = []
        self.input_calls: list[tuple[int, str]] = []
        self.scroll_calls: list[tuple[str, int, int | None]] = []

    async def click_element(self, index: int) -> _FakeResult:
        self.click_calls.append(index)
        return _FakeResult(f"clicked [{index}]")

    async def input_text(self, index: int, text: str) -> _FakeResult:
        self.input_calls.append((index, text))
        return _FakeResult(f"input [{index}]={text}")

    async def scroll(
        self, direction: str, amount: int, index: int | None
    ) -> _FakeResult:
        self.scroll_calls.append((direction, amount, index))
        return _FakeResult(f"scrolled {direction} {amount}px")

    async def get_last_update_time(self) -> float:
        return 0.0

    async def execute_javascript(self, script: str) -> _FakeResult:
        return _FakeResult("js executed")


# ---------------------------------------------------------------------------
# Basic handler delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_delegates_to_controller():
    ctrl = _FakeController()
    pa = PageActions(ctrl, None, None, [0.0])
    result = await pa.click(5)
    assert result == "clicked [5]"
    assert ctrl.click_calls == [5]


@pytest.mark.asyncio
async def test_input_text_delegates_to_controller():
    ctrl = _FakeController()
    pa = PageActions(ctrl, None, None, [0.0])
    result = await pa.input_text(3, "hello")
    assert result == "input [3]=hello"
    assert ctrl.input_calls == [(3, "hello")]


@pytest.mark.asyncio
async def test_scroll_delegates_to_controller():
    ctrl = _FakeController()
    pa = PageActions(ctrl, None, None, [0.0])
    result = await pa.scroll(down=True, pages=2, index=None)
    assert result == "scrolled down 1000px"
    assert ctrl.scroll_calls == [("down", 1000, None)]


@pytest.mark.asyncio
async def test_execute_javascript_delegates_to_controller():
    ctrl = _FakeController()
    pa = PageActions(ctrl, None, None, [0.0])
    result = await pa.execute_javascript("1+1")
    assert result == "js executed"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_populates_registry():
    registry = Registry()
    supported: set[str] = set()
    pa = PageActions(None, None, object(), [0.0])
    pa.register(registry, supported)

    # All 14 actions should be registered
    assert len(supported) == 14
    for name in (
        "click", "input", "select_dropdown", "scroll", "scroll_horizontally",
        "wait", "go_back", "close_overlay", "execute_javascript",
        "query_knowledge", "discover_zones", "extract_menu",
        "send_keys", "solve_captcha",
    ):
        assert name in supported
        assert name in registry.registry.actions


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_without_controller_raises():
    pa = PageActions(None, None, None, [0.0])
    with pytest.raises(RuntimeError, match="PageController not set"):
        await pa.click(1)


@pytest.mark.asyncio
async def test_go_back_without_browser():
    pa = PageActions(None, None, None, [0.0])
    result = await pa.go_back()
    assert "no browser" in result


@pytest.mark.asyncio
async def test_close_overlay_without_browser():
    pa = PageActions(None, None, None, [0.0])
    result = await pa.close_overlay()
    assert "no browser" in result


@pytest.mark.asyncio
async def test_wait_updates_total_wait_time_ref():
    ctrl = _FakeController()
    ref = [0.0]
    pa = PageActions(ctrl, None, None, ref)
    await pa.wait(seconds=2)
    assert ref[0] >= 1.9  # close to 2.0, allowing for minor timing
    assert ref[0] <= 3.0


@pytest.mark.asyncio
async def test_wait_caps_at_10_seconds():
    ctrl = _FakeController()
    ref = [0.0]
    pa = PageActions(ctrl, None, None, ref)
    await pa.wait(seconds=100)
    assert ref[0] <= 11.0  # should cap at 10 + small overhead


@pytest.mark.asyncio
async def test_discover_zones_returns_delegation_msg():
    pa = PageActions(None, None, None, [0.0])
    result = await pa.discover_zones()
    assert "delegated" in result

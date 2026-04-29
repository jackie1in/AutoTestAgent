from __future__ import annotations

import pytest

from graph_agent.cartography.base_agent import BaseAgent


class _FakeDomState:
    def __init__(self) -> None:
        self.selector_map = {}

    def llm_representation(self) -> str:
        return "<html></html>"


class _FakeBrowserSummary:
    def __init__(self) -> None:
        self.dom_state = _FakeDomState()
        self.title = "fake-title"


class _FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def get_browser_state_summary(
        self,
        *,
        include_screenshot: bool = True,
        cached: bool = False,
        include_recent_events: bool = False,
    ):
        _ = cached
        _ = include_recent_events
        self.calls.append(include_screenshot)
        return _FakeBrowserSummary()


class _FakeLLM:
    pass


def test_base_agent_auto_mode_enables_screenshot_action():
    agent = BaseAgent(
        task="t",
        llm=_FakeLLM(),
        browser=_FakeBrowser(),
        use_vision="auto",
    )
    assert "screenshot" in agent._supported_actions


@pytest.mark.asyncio
async def test_base_agent_true_mode_requests_screenshot_in_snapshot():
    browser = _FakeBrowser()
    agent = BaseAgent(
        task="t",
        llm=_FakeLLM(),
        browser=browser,
        use_vision=True,
    )
    await agent._get_browser_snapshot()
    assert browser.calls[-1] is True


@pytest.mark.asyncio
async def test_base_agent_false_mode_disables_screenshot_everywhere():
    browser = _FakeBrowser()
    agent = BaseAgent(
        task="t",
        llm=_FakeLLM(),
        browser=browser,
        use_vision=False,
    )
    assert "screenshot" not in agent._supported_actions
    await agent._get_browser_snapshot()
    assert browser.calls[-1] is False

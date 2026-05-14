from __future__ import annotations

import pytest

from graph_agent.cartography.react_explorer import ReActExplorer


class _FakeController:
    def __init__(self) -> None:
        self.selector_map = {1: {"dummy": True}}
        self.update_called = 0
        self.render_called = 0

    async def update_tree(self) -> str:
        self.update_called += 1
        return "[1]<button>Submit />"

    async def render_llm_dom_with_top(self, only_top: bool = True) -> str:
        self.render_called += 1
        return "[1]<button>Submit /> is_top=true"


@pytest.mark.asyncio
async def test_get_browser_snapshot_uses_update_tree_by_default() -> None:
    explorer = object.__new__(ReActExplorer)
    controller = _FakeController()
    explorer._controller = controller  # type: ignore[assignment]
    explorer.browser = None

    dom_text, title, selector_map = await ReActExplorer._get_browser_snapshot(explorer)

    assert dom_text == "[1]<button>Submit />"
    assert title == ""
    assert selector_map == {1: {"dummy": True}}
    assert controller.update_called == 1
    assert controller.render_called == 0

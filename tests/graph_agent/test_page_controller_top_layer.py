from __future__ import annotations

import pytest
from typing import cast

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import DOMSelectorMap

from graph_agent.lib.page_controller import PageController
from graph_agent.lib.page_controller.js_snippets import (
    _PATCH_ANTD_JS,
    _PATCH_REACT_JS,
)


def _fake_browser_session() -> BrowserSession:
    """Minimal BrowserSession — bypasses Pydantic init for tests that don't use browser."""
    bs = BrowserSession.__new__(BrowserSession)
    return bs


@pytest.mark.asyncio
async def test_render_llm_dom_with_top_marks_and_filters(monkeypatch: pytest.MonkeyPatch):
    controller = PageController(browser_session=_fake_browser_session())
    controller._is_indexed = True
    controller._simplified_html = "\n".join(
        [
            "[0]<button>Submit />",
            "[1]<button>Hidden />",
            "plain context",
        ]
    )
    controller._selector_map = cast(DOMSelectorMap, {0: object(), 1: object()})

    async def _fake_extract():
        return [
            {"index": 0, "is_top": True, "is_menu_container": False},
            {"index": 1, "is_top": False, "is_menu_container": False},
        ]

    monkeypatch.setattr(controller, "extract_interactive_elements_with_top", _fake_extract)

    full = await controller.render_llm_dom_with_top(only_top=False)
    assert "is_top=true" in full
    assert "is_top=false" in full

    top_only = await controller.render_llm_dom_with_top(only_top=True)
    assert "[0]<button>Submit /> is_top=true" in top_only
    assert "[1]<button>Hidden />" not in top_only


@pytest.mark.asyncio
async def test_click_element_reports_top_layer_fallback(monkeypatch: pytest.MonkeyPatch):
    class _FakeNode:
        xpath = ""
        attributes = {}

    class _FakeElement:
        def __init__(self) -> None:
            self._top_checks = 0

        async def evaluate(self, script: str):
            if "is_top" in script and "elementFromPoint" in script:
                self._top_checks += 1
                # First check false, retry check true.
                return {"is_top": self._top_checks > 1}
            return {"isBlank": False}

    controller = PageController(browser_session=_fake_browser_session())
    controller._is_indexed = True
    controller._selector_map = cast(DOMSelectorMap, {3: _FakeNode()})
    controller._element_text_map = {3: "Submit"}

    fake_element = _FakeElement()

    async def _fake_get_element(index: int):
        assert index == 3
        return fake_element

    monkeypatch.setattr(controller, "_get_element", _fake_get_element)

    result = await controller.click_element(3)
    assert result.success is True
    assert "Clicked [3] (Submit)." in result.message
    assert fake_element._top_checks >= 2


@pytest.mark.asyncio
async def test_update_tree_applies_react_then_antd_patches(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakePage:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        async def evaluate(self, script: str):
            self.scripts.append(script)
            return None

    class _FakeDomState:
        selector_map = {}

        def llm_representation(self) -> str:
            return ""

    class _FakeDomService:
        def __init__(self, _session) -> None:
            pass

        async def get_serialized_dom_tree(self):
            return _FakeDomState(), None, None

    fake_page = _FakePage()
    controller = PageController(browser_session=cast(BrowserSession, None))  # None ok: test monkeypatches away browser calls

    async def _fake_get_page():
        return fake_page

    monkeypatch.setattr(controller, "_get_page", _fake_get_page)
    monkeypatch.setattr("browser_use.dom.service.DomService", _FakeDomService)

    await controller.update_tree()

    assert fake_page.scripts[:2] == [_PATCH_REACT_JS, _PATCH_ANTD_JS]


@pytest.mark.asyncio
async def test_extract_interactive_elements_with_top_keeps_hit_source(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeNode:
        attributes = {}
        tag_name = "div"
        node_value = "x"

    class _FakeElement:
        async def evaluate(self, script: str):
            if "hit_source" in script:
                return {"is_visible": True, "is_top": False, "hit_source": "shadow"}
            return {}

    controller = PageController(browser_session=_fake_browser_session())
    controller._is_indexed = True
    controller._selector_map = cast(DOMSelectorMap, {1: _FakeNode()})

    async def _fake_get_element(index: int):
        assert index == 1
        return _FakeElement()

    monkeypatch.setattr(controller, "_get_element", _fake_get_element)

    rows = await controller.extract_interactive_elements_with_top()
    assert rows[0]["hit_source"] == "shadow"

from __future__ import annotations

import pytest

from graph_agent.lib.page_controller import PageController
from graph_agent.lib.page_controller.image_utils import extract_image_base64_from_src


@pytest.mark.asyncio
async def test_extract_image_base64_from_data_uri_returns_original_payload():
    class _FakePage:
        async def evaluate(self, script: str):  # pragma: no cover - should not run
            raise AssertionError("evaluate should not be called for data URI")

    payload = "aGVsbG8="
    src = f"data:image/png;base64,{payload}"

    result = await extract_image_base64_from_src(src, _FakePage())

    assert result == payload


@pytest.mark.asyncio
async def test_page_controller_extract_captcha_image_preserves_data_uri(monkeypatch: pytest.MonkeyPatch):
    class _FakeElement:
        async def evaluate(self, script: str):
            assert script == "el => el.src"
            return "data:image/jpeg;base64,ZmFrZV9jYXB0Y2hh"

    controller = PageController(browser_session=None)  # type: ignore[arg-type]
    controller._is_indexed = True

    async def _fake_get_element(index: int):
        assert index == 7
        return _FakeElement()

    monkeypatch.setattr(controller, "_get_element", _fake_get_element)

    result = await controller.extract_captcha_image(7)

    assert result.success is True
    assert result.message == "data:image/jpeg;base64,ZmFrZV9jYXB0Y2hh"

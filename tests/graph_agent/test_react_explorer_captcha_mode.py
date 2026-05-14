from __future__ import annotations

import pytest

from graph_agent.cartography.captcha import solve_captcha_from_page  # noqa: F401
from graph_agent.cartography.react_explorer.page_actions import PageActions


class _FakeInputResult:
    def __init__(self, message: str) -> None:
        self.message = message


class _FakeController:
    def __init__(self) -> None:
        self.filled: list[tuple[int, str]] = []

    async def input_text(self, index: int, text: str) -> _FakeInputResult:
        self.filled.append((index, text))
        return _FakeInputResult("input-ok")


class _FakePage:
    async def get_url(self) -> str:
        return "https://demo.local/login"

    async def evaluate(self, _script: str, *_args):
        return False


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def get_current_page(self) -> _FakePage:
        return self._page


def _make_page_actions(controller, browser, llm=None):
    return PageActions(
        controller=controller,
        browser=browser,
        llm=llm or object(),  # type: ignore[arg-type]
        total_wait_time_ref=[0.0],
    )


@pytest.mark.asyncio
async def test_solve_captcha_manual_mode_blocks_and_fills(
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _FakeController()
    browser = _FakeBrowser(_FakePage())
    pa = _make_page_actions(controller, browser)

    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "manual")

    async def _fake_prompt(_self, _page, _hint: str) -> str:
        return "Ab12"

    monkeypatch.setattr(PageActions, "_prompt_manual_captcha_code", _fake_prompt)

    result = await pa.solve_captcha(input_index=7, input_hint="验证码")

    assert "CAPTCHA_OK filled_index=7" in result
    assert "mode=manual" in result
    assert controller.filled == [(7, "Ab12")]


@pytest.mark.asyncio
async def test_solve_captcha_manual_mode_empty_input_returns_marker(
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _FakeController()
    browser = _FakeBrowser(_FakePage())
    pa = _make_page_actions(controller, browser)

    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "manual")

    async def _fake_prompt(_self, _page, _hint: str) -> str:
        return ""

    monkeypatch.setattr(PageActions, "_prompt_manual_captcha_code", _fake_prompt)

    result = await pa.solve_captcha(input_index=7, input_hint="验证码")

    assert result.startswith("CAPTCHA_MANUAL_EMPTY")
    assert "mode=manual" in result


@pytest.mark.asyncio
async def test_solve_captcha_auto_mode_uses_llm_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _FakeController()
    browser = _FakeBrowser(_FakePage())
    pa = _make_page_actions(controller, browser)

    monkeypatch.delenv("CAPTCHA_SOLVE_MODE", raising=False)

    called = {"value": False}

    async def _fake_solver(_page, _login_info, _llm, *, input_hint: str = "") -> str:
        called["value"] = True
        assert input_hint == "验证码"
        return "9A2B"

    monkeypatch.setattr(
        "graph_agent.cartography.react_explorer.page_actions.solve_captcha_from_page",
        _fake_solver,
    )

    result = await pa.solve_captcha(input_index=3, input_hint="验证码")

    assert called["value"] is True
    assert "CAPTCHA_OK filled_index=3" in result
    assert "mode=auto" in result

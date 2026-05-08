from __future__ import annotations

import pytest

import graph_agent.cartography.react_explorer as react_explorer_module
from graph_agent.cartography.react_explorer import ReActExplorer


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


@pytest.mark.asyncio
async def test_solve_captcha_manual_mode_blocks_and_fills(monkeypatch: pytest.MonkeyPatch):
    explorer = object.__new__(ReActExplorer)
    explorer._controller = _FakeController()
    explorer.browser = _FakeBrowser(_FakePage())
    explorer.llm = object()

    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "manual")

    async def _fake_prompt(_self, _page, _hint: str) -> str:
        return "Ab12"

    monkeypatch.setattr(ReActExplorer, "_prompt_manual_captcha_code", _fake_prompt)

    result = await ReActExplorer._execute_action(
        explorer,
        "solve_captcha",
        {"input_index": 7, "input_hint": "验证码"},
    )

    assert "CAPTCHA_OK filled_index=7" in result
    assert "mode=manual" in result
    assert "wait_ms=" in result
    assert explorer._controller.filled == [(7, "Ab12")]


@pytest.mark.asyncio
async def test_solve_captcha_manual_mode_empty_input_returns_marker(
    monkeypatch: pytest.MonkeyPatch,
):
    explorer = object.__new__(ReActExplorer)
    explorer._controller = _FakeController()
    explorer.browser = _FakeBrowser(_FakePage())
    explorer.llm = object()

    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "manual")

    async def _fake_prompt(_self, _page, _hint: str) -> str:
        return ""

    monkeypatch.setattr(ReActExplorer, "_prompt_manual_captcha_code", _fake_prompt)

    result = await ReActExplorer._execute_action(
        explorer,
        "solve_captcha",
        {"input_index": 7, "input_hint": "验证码"},
    )

    assert result.startswith("CAPTCHA_MANUAL_EMPTY")
    assert "mode=manual" in result


@pytest.mark.asyncio
async def test_solve_captcha_auto_mode_uses_llm_solver(monkeypatch: pytest.MonkeyPatch):
    explorer = object.__new__(ReActExplorer)
    explorer._controller = _FakeController()
    explorer.browser = _FakeBrowser(_FakePage())
    explorer.llm = object()

    monkeypatch.delenv("CAPTCHA_SOLVE_MODE", raising=False)

    called = {"value": False}

    async def _fake_solver(_page, _login_info, _llm, *, input_hint: str = "") -> str:
        called["value"] = True
        assert input_hint == "验证码"
        return "9A2B"

    monkeypatch.setattr(react_explorer_module, "solve_captcha_from_page", _fake_solver)

    result = await ReActExplorer._execute_action(
        explorer,
        "solve_captcha",
        {"input_index": 3, "input_hint": "验证码"},
    )

    assert called["value"] is True
    assert "CAPTCHA_OK filled_index=3" in result
    assert "mode=auto" in result

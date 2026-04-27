from __future__ import annotations

import base64

import pytest

from graph_agent.cartography.runner import (  # type: ignore[attr-defined]
    _recognize_captcha_with_fallback,
)
from graph_agent.lib.captcha_solver import solve_with_ddddocr  # type: ignore[import-untyped]


class _FakeDdddOcr:
    def __init__(self, out: str):
        self._out = out

    def classification(self, _img_bytes: bytes) -> str:
        return self._out


class _FakeLlmResult:
    def __init__(self, completion: str):
        self.completion = completion


class _FakeLlm:
    def __init__(self, completion: str):
        self._completion = completion
        self.called = 0

    async def ainvoke(self, _messages):
        self.called += 1
        return _FakeLlmResult(self._completion)


def _sample_data_url() -> str:
    payload = base64.b64encode(b"fake-image-bytes").decode()
    return f"data:image/png;base64,{payload}"


def test_solve_with_ddddocr_accepts_data_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "graph_agent.lib.captcha_solver._get_ocr_engine",
        lambda: _FakeDdddOcr(" A1b2 "),
    )
    assert solve_with_ddddocr(_sample_data_url()) == "A1b2"


def test_solve_with_ddddocr_invalid_input_returns_empty():
    assert solve_with_ddddocr("not-a-valid-image") == ""


def test_solve_with_ddddocr_cleans_non_alnum(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "graph_agent.lib.captcha_solver._get_ocr_engine",
        lambda: _FakeDdddOcr("`A-1_ B`"),
    )
    assert solve_with_ddddocr(_sample_data_url()) == "A1B"


@pytest.mark.asyncio
async def test_runner_prefers_ddddocr_and_skips_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CAPTCHA_DDDDOCR_ENABLED", "true")
    monkeypatch.setenv("CAPTCHA_DDDDOCR_ONLY", "false")
    monkeypatch.setattr(
        "graph_agent.cartography.runner.solve_with_ddddocr",
        lambda _img: "X7K2",
    )
    llm = _FakeLlm("SHOULD_NOT_BE_USED")
    code = await _recognize_captcha_with_fallback(_sample_data_url(), llm)
    assert code == "X7K2"
    assert llm.called == 0


@pytest.mark.asyncio
async def test_runner_falls_back_to_llm_when_ddddocr_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CAPTCHA_DDDDOCR_ENABLED", "true")
    monkeypatch.setenv("CAPTCHA_DDDDOCR_ONLY", "false")
    monkeypatch.setattr(
        "graph_agent.cartography.runner.solve_with_ddddocr",
        lambda _img: "",
    )
    llm = _FakeLlm("ab12")
    code = await _recognize_captcha_with_fallback(_sample_data_url(), llm)
    assert code == "ab12"
    assert llm.called == 1

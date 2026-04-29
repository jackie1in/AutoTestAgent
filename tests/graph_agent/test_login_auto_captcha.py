from __future__ import annotations

import pytest

import graph_agent.cartography.login as login_module


class _FakeBrowser:
    async def get_current_page(self):
        return object()

    async def get_current_page_url(self):
        return "https://example.com/dashboard"


class _FakePage:
    def __init__(self):
        self.evaluate_calls = 0

    async def evaluate(self, _script, *_args):
        self.evaluate_calls += 1
        return True


class _FakeBrowserWithPage:
    def __init__(self, page):
        self._page = page

    async def get_current_page(self):
        return self._page

    async def get_current_page_url(self):
        return "https://example.com/dashboard"


@pytest.mark.asyncio
async def test_auto_login_uses_captcha_code_when_login_page_has_captcha(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fast_sleep(_seconds: float):
        return None

    monkeypatch.setattr(login_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setenv("MAPPING_USERNAME", "demo_user")
    monkeypatch.setenv("MAPPING_PASSWORD", "demo_pass")

    async def _fake_detect_login_info(_page):
        return {
            "hasLogin": True,
            "hasCaptcha": True,
            "hasCaptchaInput": True,
            "captchaTag": "img",
            "captchaId": "captcha-img",
            "captchaSrc": "data:image/png;base64,AAAA",
        }

    observed: dict[str, str] = {"captcha_code": ""}

    async def _fake_fill_login_form_via_evaluate(
        _page,
        *,
        username: str,
        password: str,
        captcha_code: str = "",
    ):
        observed["captcha_code"] = captcha_code
        assert username == "demo_user"
        assert password == "demo_pass"
        return {"success": True, "submitClicked": True}

    monkeypatch.setattr(login_module, "detect_login_info", _fake_detect_login_info)
    async def _fake_solve_captcha_from_page(_page, _login_info, _llm):
        return "A1B2"

    monkeypatch.setattr(
        login_module,
        "solve_captcha_from_page",
        _fake_solve_captcha_from_page,
        raising=False,
    )
    monkeypatch.setattr(
        login_module,
        "fill_login_form_via_evaluate",
        _fake_fill_login_form_via_evaluate,
    )

    ok = await login_module.try_auto_login_orchestrated(_FakeBrowser(), llm=object())
    assert ok is True
    assert observed["captcha_code"] == "A1B2"


@pytest.mark.asyncio
async def test_auto_login_refreshes_captcha_and_retries_when_first_result_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fast_sleep(_seconds: float):
        return None

    monkeypatch.setattr(login_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setenv("MAPPING_USERNAME", "demo_user")
    monkeypatch.setenv("MAPPING_PASSWORD", "demo_pass")

    async def _fake_detect_login_info(_page):
        return {
            "hasLogin": True,
            "hasCaptcha": True,
            "hasCaptchaInput": True,
            "captchaTag": "img",
            "captchaId": "captcha-img",
            "captchaSrc": "https://example.com/captcha",
        }

    solve_calls = {"n": 0}

    async def _fake_solve_captcha_from_page(_page, _login_info, _llm):
        solve_calls["n"] += 1
        if solve_calls["n"] == 1:
            return ""
        return "Z9X8"

    observed: dict[str, str] = {"captcha_code": ""}

    async def _fake_fill_login_form_via_evaluate(
        _page,
        *,
        username: str,
        password: str,
        captcha_code: str = "",
    ):
        observed["captcha_code"] = captcha_code
        assert username == "demo_user"
        assert password == "demo_pass"
        return {"success": True, "submitClicked": True}

    fake_page = _FakePage()
    browser = _FakeBrowserWithPage(fake_page)

    monkeypatch.setattr(login_module, "detect_login_info", _fake_detect_login_info)
    monkeypatch.setattr(
        login_module,
        "solve_captcha_from_page",
        _fake_solve_captcha_from_page,
        raising=False,
    )
    monkeypatch.setattr(
        login_module,
        "fill_login_form_via_evaluate",
        _fake_fill_login_form_via_evaluate,
    )

    ok = await login_module.try_auto_login_orchestrated(browser, llm=object())
    assert ok is True
    assert solve_calls["n"] == 2
    assert fake_page.evaluate_calls == 1
    assert observed["captcha_code"] == "Z9X8"


@pytest.mark.asyncio
async def test_auto_login_does_not_refresh_captcha_without_captcha_input(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fast_sleep(_seconds: float):
        return None

    monkeypatch.setattr(login_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setenv("MAPPING_USERNAME", "demo_user")
    monkeypatch.setenv("MAPPING_PASSWORD", "demo_pass")

    async def _fake_detect_login_info(_page):
        return {
            "hasLogin": True,
            "hasCaptcha": True,
            "hasCaptchaInput": False,
            "captchaTag": "img",
            "captchaId": "captcha-img",
            "captchaSrc": "https://example.com/captcha",
        }

    solve_calls = {"n": 0}

    async def _fake_solve_captcha_from_page(_page, _login_info, _llm):
        solve_calls["n"] += 1
        return ""

    observed: dict[str, str] = {"captcha_code": ""}

    async def _fake_fill_login_form_via_evaluate(
        _page,
        *,
        username: str,
        password: str,
        captcha_code: str = "",
    ):
        observed["captcha_code"] = captcha_code
        assert username == "demo_user"
        assert password == "demo_pass"
        return {"success": True, "submitClicked": True}

    fake_page = _FakePage()
    browser = _FakeBrowserWithPage(fake_page)

    monkeypatch.setattr(login_module, "detect_login_info", _fake_detect_login_info)
    monkeypatch.setattr(
        login_module,
        "solve_captcha_from_page",
        _fake_solve_captcha_from_page,
        raising=False,
    )
    monkeypatch.setattr(
        login_module,
        "fill_login_form_via_evaluate",
        _fake_fill_login_form_via_evaluate,
    )

    ok = await login_module.try_auto_login_orchestrated(browser, llm=object())
    assert ok is True
    assert solve_calls["n"] == 1
    assert fake_page.evaluate_calls == 0
    assert observed["captcha_code"] == ""

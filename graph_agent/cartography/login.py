from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING, cast

from graph_agent.cartography.captcha import solve_captcha_from_page
from graph_agent.cartography.types import FillResult, LoginInfo

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.browser.session import BrowserSession as Browser
    from browser_use.llm.base import BaseChatModel


LOGIN_DETECT_SCRIPT = r"""
(...args) => {
    var pwd = document.querySelector('input[type="password"]');
    if (!pwd) return {hasLogin: false};
    var user = document.querySelector('input[type="text"], input[type="email"], input:not([type])');
    var form = pwd.closest('form');
    var submit = form ? form.querySelector('button[type="submit"], input[type="submit"]') : null;
    if (!submit) {
        submit = pwd.closest('form, div, section')?.querySelector('button[type="submit"], input[type="submit"]');
    }
    if (!submit) {
        var allBtns = document.querySelectorAll('button');
        for (var i = 0; i < allBtns.length; i++) {
            var txt = allBtns[i].innerText || allBtns[i].textContent || '';
            if (/登录|登入|login|sign.in|submit/i.test(txt)) {
                submit = allBtns[i];
                break;
            }
        }
    }
    var captchaImg = null;
    var allImgs = document.querySelectorAll('img');
    for (var j = 0; j < allImgs.length; j++) {
        var combined = (allImgs[j].src || '') + (allImgs[j].alt || '') + (allImgs[j].id || '') + (allImgs[j].className || '');
        if (/captcha|验证码|verify|auth|code/i.test(combined)) {
            captchaImg = allImgs[j];
            break;
        }
    }
    if (!captchaImg) {
        var canvases = document.querySelectorAll('canvas');
        for (var k = 0; k < canvases.length; k++) {
            if (/captcha|验证码|verify|auth|code/i.test((canvases[k].id || '') + (canvases[k].className || ''))) {
                captchaImg = canvases[k];
                break;
            }
        }
    }
    var captchaInput = null;
    var allInputs = document.querySelectorAll('input');
    for (var m = 0; m < allInputs.length; m++) {
        var inp = allInputs[m];
        var inpType = inp.type || 'text';
        var inpName = (inp.name || '').toLowerCase();
        var inpId = (inp.id || '').toLowerCase();
        var inpPlaceholder = (inp.placeholder || '').toLowerCase();
        var inpCls = (inp.className || '').toLowerCase();
        if (inpType === 'password') continue;
        if (/captcha|验证码|verify.*code|auth.*code|code/i.test(inpName + inpId + inpPlaceholder + inpCls)) {
            captchaInput = inp;
            break;
        }
    }
    return {
        hasLogin: true,
        hasUser: !!user,
        userTag: user ? user.tagName.toLowerCase() : '',
        userName: user ? (user.name || '') : '',
        userId: user ? (user.id || '') : '',
        pwdTag: pwd.tagName.toLowerCase(),
        pwdName: pwd.name || '',
        pwdId: pwd.id || '',
        submitTag: submit ? submit.tagName.toLowerCase() : '',
        submitType: submit ? (submit.type || '') : '',
        hasCaptcha: !!captchaImg,
        hasCaptchaInput: !!captchaInput,
        captchaTag: captchaImg ? captchaImg.tagName.toLowerCase() : '',
        captchaSrc: captchaImg ? (captchaImg.src || '') : '',
        captchaId: captchaImg ? (captchaImg.id || '') : '',
        captchaInputName: captchaInput ? (captchaInput.name || '') : '',
        captchaInputId: captchaInput ? (captchaInput.id || '') : '',
    };
}
"""


def parse_evaluate_result(raw: object) -> dict[str, object]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def is_login_url(url: str) -> bool:
    if not url:
        return False
    return bool(re.search(r"login|signin|sign-in|auth", url, re.IGNORECASE))


async def detect_login_info(page: "Page") -> LoginInfo:
    try:
        raw = await page.evaluate(LOGIN_DETECT_SCRIPT)
    except Exception:
        return {}
    return cast(LoginInfo, parse_evaluate_result(raw))


async def click_menu_by_text(browser: "Browser", text: str) -> bool:
    target = (text or "").strip()
    if not target:
        return False
    script = (
        "(target) => {\n"
        "  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();\n"
        "  const candidates = [\n"
        "    'a[role=\"menuitem\"]', '[role=\"menuitem\"]',\n"
        "    '.ant-menu-item', '.el-menu-item', '.el-submenu__title',\n"
        "    '.ant-menu-submenu-title', '.menu-item', '[class*=\"menu-item\"]',\n"
        "    'aside a', 'aside button', 'nav a', 'nav button',\n"
        "    'a', 'button', '[role=\"button\"]'\n"
        "  ];\n"
        "  const seen = new Set();\n"
        "  for (const sel of candidates) {\n"
        "    const els = Array.from(document.querySelectorAll(sel));\n"
        "    for (const el of els) {\n"
        "      if (seen.has(el)) continue;\n"
        "      seen.add(el);\n"
        "      const t = norm(el.innerText || el.textContent);\n"
        "      if (!t) continue;\n"
        "      if (t === target || (t.length <= 40 && t.includes(target))) {\n"
        "        const rect = el.getBoundingClientRect();\n"
        "        if (rect.width === 0 || rect.height === 0) continue;\n"
        "        el.scrollIntoView({block: 'center'});\n"
        "        el.click();\n"
        "        return JSON.stringify({ok: true, tag: el.tagName, selector: sel});\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  return JSON.stringify({ok: false});\n"
        "}"
    )
    try:
        page = await browser.get_current_page()
        raw = await page.evaluate(script, target)
        parsed = parse_evaluate_result(raw)
        if parsed.get("ok"):
            print(f"[PIPELINE] Menu '{target}' clicked (via {parsed.get('selector', '?')}, tag={parsed.get('tag', '?')})")
            return True
        print(f"[PIPELINE] Menu '{target}' not found on current page")
        return False
    except Exception as e:
        print(f"[PIPELINE] click_menu_by_text error: {e}")
        return False


async def fill_login_form_via_evaluate(
    page: "Page",
    *,
    username: str,
    password: str,
    captcha_code: str = "",
) -> FillResult:
    fill_script = r"""
    (...args) => {
        const [username, password, captchaCode] = args;
        const out = { success: false, userFilled: false, pwdFilled: false, captchaFilled: false, submitClicked: false, reason: "" };
        const setValue = (el, val) => {
            try { el.focus(); } catch (e) {}
            const proto = Object.getPrototypeOf(el);
            const desc = proto && Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) desc.set.call(el, val); else el.value = val;
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        };
        const pwd = document.querySelector('input[type="password"]');
        if (!pwd) { out.reason = "no password input"; return out; }
        const user = document.querySelector(
            'input[type="text"], input[type="email"], input[name*="user" i], ' +
            'input[id*="user" i], input[name*="account" i], input[id*="account" i], input:not([type])'
        );
        if (user) { setValue(user, username); out.userFilled = true; }
        setValue(pwd, password); out.pwdFilled = true;
        if (captchaCode) {
            const inputs = document.querySelectorAll('input');
            for (const inp of inputs) {
                const t = inp.type || 'text';
                if (t === 'password') continue;
                const sig = ((inp.name || '') + (inp.id || '') + (inp.placeholder || '') + (inp.className || '')).toLowerCase();
                if (/captcha|验证码|verify.*code|auth.*code|code/i.test(sig)) {
                    setValue(inp, captchaCode);
                    out.captchaFilled = true;
                    break;
                }
            }
        }
        let submitBtn = null;
        for (const b of document.querySelectorAll('button')) {
            const txt = b.innerText || b.textContent || '';
            if (/登录|登入|login|sign.in|submit/i.test(txt)) { submitBtn = b; break; }
        }
        if (!submitBtn) submitBtn = document.querySelector('button[type="submit"], input[type="submit"]');
        if (submitBtn) { submitBtn.click(); out.submitClicked = true; }
        else {
            try {
                const form = pwd.closest('form');
                if (form) { form.requestSubmit ? form.requestSubmit() : form.submit(); out.submitClicked = true; }
            } catch (e) { out.reason = "submit failed: " + (e && e.message ? e.message : e); }
        }
        out.success = out.pwdFilled;
        return out;
    }
    """
    try:
        raw = await page.evaluate(fill_script, username, password, captcha_code or "")
    except Exception as e:
        return {"success": False, "reason": f"evaluate failed: {e}"}
    parsed = parse_evaluate_result(raw)
    if not parsed:
        return {"success": False, "reason": f"empty evaluate result (raw={raw!r})"}
    return cast(FillResult, parsed)


async def refresh_captcha_on_page(page: "Page", login_info: LoginInfo) -> bool:
    captcha_id = str(login_info.get("captchaId") or "")
    refresh_script = r"""
    (...args) => {
        const [captchaId] = args;
        let target = null;
        if (captchaId) {
            target = document.getElementById(captchaId);
        }
        if (!target) {
            for (const img of document.querySelectorAll("img")) {
                const sig = ((img.src || "") + (img.alt || "") + (img.id || "") + (img.className || "")).toLowerCase();
                if (/captcha|验证码|verify|auth|code/.test(sig)) {
                    target = img;
                    break;
                }
            }
        }
        if (!target) {
            for (const canvas of document.querySelectorAll("canvas")) {
                const sig = ((canvas.id || "") + (canvas.className || "")).toLowerCase();
                if (/captcha|验证码|verify|auth|code/.test(sig)) {
                    target = canvas;
                    break;
                }
            }
        }
        if (!target || typeof target.click !== "function") {
            return false;
        }
        target.click();
        return true;
    }
    """
    try:
        refreshed = await page.evaluate(refresh_script, captcha_id)
        return bool(refreshed)
    except Exception as e:
        print(f"[ORCH-AUTO_LOGIN] Captcha refresh failed: {e}")
        return False


async def try_auto_login_orchestrated(browser: "Browser", llm: "BaseChatModel") -> bool:
    username = (os.getenv("MAPPING_USERNAME") or "").strip()
    password = (os.getenv("MAPPING_PASSWORD") or "").strip()
    if not username or not password:
        print("[ORCH-AUTO_LOGIN] No MAPPING_USERNAME/MAPPING_PASSWORD in env, skipping auto-login.")
        return False

    try:
        page = await browser.get_current_page()
    except Exception as e:
        print(f"[ORCH-AUTO_LOGIN] get_current_page() failed: {e}")
        return False
    if not page:
        return False

    await asyncio.sleep(0.5)
    login_info = await detect_login_info(page)
    if not login_info or not login_info.get("hasLogin"):
        return False

    captcha_code = ""
    if login_info.get("hasCaptcha"):
        has_captcha_input = bool(login_info.get("hasCaptchaInput"))
        try:
            captcha_code = await solve_captcha_from_page(page, login_info, llm)
            if captcha_code:
                print(f"[ORCH-AUTO_LOGIN] Captcha solved with code '{captcha_code}'.")
            else:
                print("[ORCH-AUTO_LOGIN] Captcha detected but solver returned empty code.")
                if has_captcha_input:
                    refreshed = await refresh_captcha_on_page(page, login_info)
                    if refreshed:
                        print("[ORCH-AUTO_LOGIN] Captcha refreshed. Retrying solve once.")
                        await asyncio.sleep(0.6)
                        captcha_code = await solve_captcha_from_page(page, login_info, llm)
                        if captcha_code:
                            print(f"[ORCH-AUTO_LOGIN] Captcha solved after refresh with code '{captcha_code}'.")
                        else:
                            print("[ORCH-AUTO_LOGIN] Captcha still unsolved after refresh retry.")
                else:
                    print("[ORCH-AUTO_LOGIN] Skip captcha refresh retry because captcha input is not detected.")
        except Exception as e:
            print(f"[ORCH-AUTO_LOGIN] Captcha solving failed: {e}")
            captcha_code = ""

    fill_result = await fill_login_form_via_evaluate(
        page,
        username=username,
        password=password,
        captcha_code=captcha_code,
    )
    if not fill_result.get("success"):
        return False
    await asyncio.sleep(5)
    post_url = await browser.get_current_page_url()
    return bool(post_url and not is_login_url(post_url))

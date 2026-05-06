from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)

from graph_agent.cartography.types import LoginInfo
from graph_agent.lib.page_controller import PageController

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.llm.base import BaseChatModel


_CAPTCHA_HINT_PATTERN = re.compile(
    r"captcha|验证码|verify|auth|check.*code|rand.*code", re.IGNORECASE
)
_WEAK_CAPTCHA_HINT_PATTERN = re.compile(r"code", re.IGNORECASE)
_NON_CAPTCHA_PATTERN = re.compile(r"logo|avatar|icon|banner|ad|wechat", re.IGNORECASE)


def _xpath_proximity_score(img_xpath: str, input_xpaths: list[str]) -> int:
    if not img_xpath or not input_xpaths:
        return 0
    best = 0
    img_parts = [p for p in img_xpath.split("/") if p]
    for input_xpath in input_xpaths:
        in_parts = [p for p in input_xpath.split("/") if p]
        common = 0
        for a, b in zip(img_parts, in_parts):
            if a != b:
                break
            common += 1
        if common > best:
            best = common
    return best


def _extract_form_ancestor_xpath(xpath: str) -> str:
    if not xpath:
        return ""
    parts = [p for p in xpath.split("/") if p]
    for i, part in enumerate(parts):
        if part.lower().startswith("form"):
            return "/" + "/".join(parts[: i + 1])
    return ""


def _in_same_form(img_xpath: str, form_xpaths: list[str]) -> bool:
    if not img_xpath or not form_xpaths:
        return False
    return any(
        img_xpath.startswith(form_xpath + "/") or img_xpath == form_xpath
        for form_xpath in form_xpaths
    )


def _score_captcha_img_node(
    *,
    attrs: dict[str, object],
    node_value: str,
    img_xpath: str,
    captcha_id_norm: str,
    input_xpaths: list[str],
    form_xpaths: list[str],
) -> int:
    attr_id = str(attrs.get("id") or "")
    attr_name = str(attrs.get("name") or "")
    attr_alt = str(attrs.get("alt") or "")
    attr_title = str(attrs.get("title") or "")
    attr_aria = str(attrs.get("aria-label") or "")
    attr_class = str(attrs.get("class") or "")
    attr_src = str(attrs.get("src") or "")
    semantic_sig = " ".join(
        [attr_id, attr_name, attr_alt, attr_title, attr_aria, attr_class, node_value]
    ).lower()
    score = 0

    if captcha_id_norm and attr_id.strip().lower() == captcha_id_norm:
        score += 12
    if _CAPTCHA_HINT_PATTERN.search(semantic_sig):
        score += 8
    elif _WEAK_CAPTCHA_HINT_PATTERN.search(semantic_sig):
        score += 2
    if attr_src.startswith("data:image"):
        score += 1
    if _NON_CAPTCHA_PATTERN.search(semantic_sig):
        score -= 8

    score += min(_xpath_proximity_score(img_xpath, input_xpaths), 8)
    if _in_same_form(img_xpath, form_xpaths):
        score += 10

    return score


def _should_keep_img_candidate(
    *,
    img_xpath: str,
    form_xpaths: list[str],
    login_anchor_xpaths: list[str],
) -> bool:
    # Same form → always keep.
    if form_xpaths and _in_same_form(img_xpath, form_xpaths):
        return True
    # Close enough to any anchor input (captcha input, password input, etc.).
    # Threshold lowered to 3 so that sibling-level img nodes are included even
    # when they are not wrapped in a <form> element.
    if login_anchor_xpaths:
        return _xpath_proximity_score(img_xpath, login_anchor_xpaths) >= 3
    return True


def _parse_evaluate_result(raw: object) -> dict[str, object]:
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


async def recognize_captcha_with_fallback(
    image_data_url: str,
    llm: "BaseChatModel",
) -> str:
    return await recognize_captcha_with_candidates([image_data_url], llm)


def _normalize_captcha_code(raw_code: str) -> str:
    cleaned = raw_code.replace("```", "").replace("`", "").strip()
    if cleaned.lower() in ("unknown", "", "n/a"):
        return ""
    # Handle arithmetic captchas like "9+8=?" or "9*3=?".
    expr = (
        cleaned.replace(" ", "")
        .replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
        .replace("÷", "/")
    )
    match = re.search(r"(\d+)([+\-*/])(\d+)", expr)
    if match:
        left = int(match.group(1))
        op = match.group(2)
        right = int(match.group(3))
        value: int | None = None
        if op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op == "*":
            value = left * right
        elif op == "/" and right != 0:
            value = left // right if left % right == 0 else None
        if value is not None:
            return str(value)
    fallback = "".join(ch for ch in cleaned if ch.isalnum())
    return fallback


def _needs_arithmetic_retry(raw_code: str) -> bool:
    raw = (raw_code or "").strip()
    if not raw:
        return False
    has_equal_or_qmark = "=" in raw or "?" in raw or "？" in raw
    has_operator = bool(re.search(r"[+\-*/×xX÷]", raw))
    return has_equal_or_qmark and not has_operator


async def _recognize_arithmetic_with_candidates(
    image_data_urls: list[str],
    llm: "BaseChatModel",
) -> str:
    content: list[ContentPartTextParam | ContentPartImageParam] = [
        ContentPartTextParam(
            text=(
                "You are reading arithmetic CAPTCHA images. "
                "Return the expression exactly as shown, including operator and symbols, "
                "e.g. '7+1=?' or '9-3=?'. "
                "If it is not an arithmetic captcha, return UNKNOWN. "
                "Output only the expression text."
            )
        )
    ]
    for idx, data_url in enumerate(image_data_urls, start=1):
        content.append(ContentPartTextParam(text=f"Arithmetic candidate #{idx}:"))
        content.append(
            ContentPartImageParam(image_url=ImageURL(url=data_url, detail="high"))
        )
    result = await llm.ainvoke([UserMessage(content=content)])
    return str(result.completion or "")


async def recognize_captcha_with_candidates(
    image_data_urls: list[str],
    llm: "BaseChatModel",
) -> str:
    normalized_urls = [
        u for u in image_data_urls if isinstance(u, str) and u.startswith("data:image")
    ]
    if not normalized_urls:
        return ""
    print("[CAPTCHA][fallback-llm] using LLM vision for captcha.")
    try:
        content: list[ContentPartTextParam | ContentPartImageParam] = [
            ContentPartTextParam(
                text=(
                    "You are a CAPTCHA solver. You will receive one or more candidate images from a login page. "
                    "Only one candidate may contain the captcha. First identify which image is the captcha, "
                    "then return ONLY the exact captcha characters. "
                    "The captcha is usually 4-6 characters. Pay attention to: "
                    "- Similar looking characters (0 vs O, 1 vs l vs I, 5 vs S, 8 vs B) "
                    "- Case sensitivity (uppercase vs lowercase letters) "
                    "- Ignore page text, logos, and QR codes. "
                    "- Do NOT guess; if truly unreadable, return 'UNKNOWN'. "
                    "Return ONLY the characters, no explanation, no quotes, no markdown."
                )
            )
        ]
        for idx, data_url in enumerate(normalized_urls, start=1):
            content.append(ContentPartTextParam(text=f"Candidate image #{idx}:"))
            content.append(
                ContentPartImageParam(image_url=ImageURL(url=data_url, detail="high"))
            )

        messages = [UserMessage(content=content)]
        result = await llm.ainvoke(messages)
        raw_code = str(result.completion or "")
        normalized_code = _normalize_captcha_code(raw_code)
        if _needs_arithmetic_retry(raw_code):
            retry_raw = await _recognize_arithmetic_with_candidates(
                normalized_urls, llm
            )
            retry_normalized = _normalize_captcha_code(retry_raw)
            if retry_normalized:
                normalized_code = retry_normalized
        return normalized_code
    except Exception as e:
        print(f"[CAPTCHA] LLM recognition failed with exception: {e}")
        return ""


# Broad pattern that matches any image-based verification code input.
# Covers: 图形验证码, 验证码, captcha, verify code, auth code, code, 图片码
_CAPTCHA_INPUT_PATTERN = re.compile(
    r"captcha|图形.*码|图片.*码|验证码|verify.*code|auth.*code|\bcode\b",
    re.IGNORECASE,
)


async def solve_captcha_from_page(
    page: "Page",
    login_info: "LoginInfo | None",
    llm: "BaseChatModel",
    *,
    input_hint: str = "",
) -> str:
    """Locate the verification-code image nearest to the target input and recognise it.

    Args:
        page: Current browser page.
        login_info: Legacy login-detection result; pass None when calling outside
            a login-form context (e.g. payment, bind, share pages with captchas).
        llm: Vision-capable LLM used for image recognition.
        input_hint: Semantic description of the target input (placeholder / label
            text, e.g. '图形验证码', 'captcha').  When provided, the function uses
            this to locate the anchor input and finds the nearest image to it,
            rather than assuming a password-form layout.
    """
    if login_info is None:
        login_info = {}
    captcha_tag = login_info.get("captchaTag", "")
    captcha_id = login_info.get("captchaId", "")
    captcha_src = login_info.get("captchaSrc", "")
    hint_norm = input_hint.strip().lower()
    print(
        f"[CAPTCHA] solve_captcha_from_page called. tag={captcha_tag}, "
        f"id={captcha_id}, hint='{hint_norm}', src={captcha_src if captcha_src else 'empty'}"
    )

    image_data_urls: list[str] = []
    has_scope_hints = False  # True once we have form/input anchor xpaths

    def _add_candidate(data_url: str | None) -> None:
        if not isinstance(data_url, str):
            return
        candidate = data_url.strip()
        if not candidate.startswith("data:image"):
            return
        if candidate not in image_data_urls:
            image_data_urls.append(candidate)

    # Strategy 0: use browser-use indexed DOM (selector_map) to locate captcha image first.
    try:
        bs = getattr(page, "_browser_session", None)
        if bs is not None:
            controller = PageController(bs)
            await controller.update_tree()
            captcha_id_norm = str(captcha_id or "").strip().lower()
            input_xpaths: list[str] = []
            form_xpaths: list[str] = []
            anchor_xpaths: list[str] = []  # any nearby input as proximity anchor

            # Build a hint-specific matcher when the caller tells us which input
            # field the captcha belongs to (e.g. placeholder='图形验证码').
            hint_re = (
                re.compile(re.escape(hint_norm), re.IGNORECASE) if hint_norm else None
            )

            for node in (controller.selector_map or {}).values():
                tag = str(getattr(node, "tag_name", "") or "").lower()
                if tag != "input":
                    continue
                attrs = getattr(node, "attributes", {}) or {}
                xp = str(getattr(node, "xpath", "") or "")
                inp_type = str(attrs.get("type") or "").lower()

                # Build a rich signature from every attribute the LLM might use
                # as a hint: id, name, placeholder, aria-label, class, label text
                sig = " ".join(
                    [
                        str(attrs.get("id") or ""),
                        str(attrs.get("name") or ""),
                        str(attrs.get("placeholder") or ""),
                        str(attrs.get("aria-label") or ""),
                        str(attrs.get("class") or ""),
                        str(getattr(node, "node_value", "") or ""),
                    ]
                ).lower()

                # Password inputs → mark as login-form anchor
                if inp_type == "password":
                    if xp:
                        anchor_xpaths.append(xp)
                    form_xpath = _extract_form_ancestor_xpath(xp)
                    if form_xpath and form_xpath not in form_xpaths:
                        form_xpaths.append(form_xpath)
                    continue

                # Priority 1: explicit hint from caller (most precise anchor)
                if hint_re and hint_re.search(sig):
                    if xp:
                        input_xpaths.append(xp)
                        anchor_xpaths.append(xp)
                    form_xpath = _extract_form_ancestor_xpath(xp)
                    if form_xpath and form_xpath not in form_xpaths:
                        form_xpaths.append(form_xpath)
                    continue

                # Priority 2: any input whose attributes indicate an image-code field
                if _CAPTCHA_INPUT_PATTERN.search(sig):
                    if xp:
                        input_xpaths.append(xp)
                        anchor_xpaths.append(xp)
                    form_xpath = _extract_form_ancestor_xpath(xp)
                    if form_xpath and form_xpath not in form_xpaths:
                        form_xpaths.append(form_xpath)
                    continue

                # Priority 3: adjacent credential fields (username / account)
                # used only as proximity anchors, not as direct scope markers
                if re.search(r"user|account|login|用户名|账号", sig):
                    if xp:
                        anchor_xpaths.append(xp)
                    form_xpath = _extract_form_ancestor_xpath(xp)
                    if form_xpath and form_xpath not in form_xpaths:
                        form_xpaths.append(form_xpath)

            has_scope_hints = bool(form_xpaths or anchor_xpaths)

            ranked_img_indices: list[tuple[int, int]] = []
            for idx, node in (controller.selector_map or {}).items():
                tag = str(getattr(node, "tag_name", "") or "").lower()
                if tag != "img":
                    continue
                attrs = getattr(node, "attributes", {}) or {}
                img_xpath = str(getattr(node, "xpath", "") or "")
                if not _should_keep_img_candidate(
                    img_xpath=img_xpath,
                    form_xpaths=form_xpaths,
                    login_anchor_xpaths=anchor_xpaths,
                ):
                    continue
                score = _score_captcha_img_node(
                    attrs=attrs,
                    node_value=str(getattr(node, "node_value", "") or ""),
                    img_xpath=img_xpath,
                    captcha_id_norm=captcha_id_norm,
                    input_xpaths=input_xpaths or anchor_xpaths,
                    form_xpaths=form_xpaths,
                )
                # Keep any image that passed the proximity/form filter,
                # even if its own semantic attributes score 0.
                # Negative score (NON_CAPTCHA_PATTERN match) still excluded.
                if score >= 0:
                    ranked_img_indices.append((score, int(idx)))

            ranked_img_indices.sort(reverse=True)
            for _, img_idx in ranked_img_indices[:3]:
                extracted = await controller.extract_captcha_image(img_idx)
                if extracted.success and extracted.message:
                    message = extracted.message
                    if isinstance(message, str) and message.startswith("data:image"):
                        _add_candidate(message)
                    else:
                        _add_candidate(f"data:image/png;base64,{message}")
    except Exception as e:
        print(f"[CAPTCHA] Strategy 0 (browser-use selector_map) failed: {e}")

    if not image_data_urls and not has_scope_hints and captcha_tag == "img":
        print(
            f"[CAPTCHA] Strategy 1: captcha is <img>. src={captcha_src if captcha_src else 'empty'}"
        )
        if captcha_src.startswith("data:image"):
            _add_candidate(captcha_src)
        elif captcha_src:
            try:
                fetch_script = f"""
                (...args) => fetch('{captcha_src}', {{credentials: 'same-origin'}})
                    .then(resp => resp.blob())
                    .then(blob => new Promise((resolve) => {{
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result);
                        reader.readAsDataURL(blob);
                    }}))
                    .catch(() => null)
                """
                _add_candidate(await page.evaluate(fetch_script))
            except Exception as e:
                print(f"[CAPTCHA] Strategy 1 failed: {e}")

    if not image_data_urls and captcha_tag == "canvas":
        try:
            canvas_script = f"""
            (...args) => {{
                var canvas = document.getElementById('{captcha_id}') || document.querySelector('canvas');
                if (canvas) return canvas.toDataURL('image/png');
                return null;
            }}
            """
            _add_candidate(await page.evaluate(canvas_script))
        except Exception as e:
            print(f"[CAPTCHA] Strategy 2 failed: {e}")

    if not image_data_urls:
        try:
            # Let LLM decide captcha from multiple candidates, avoid static single-element bias.
            candidates_script = r"""
            (...args) => {
                const out = [];
                const pwd = document.querySelector('input[type="password"]');
                const loginForm = pwd && typeof pwd.closest === 'function' ? pwd.closest('form') : null;
                const pushUrl = (url) => {
                    if (typeof url === 'string' && url.startsWith('data:image') && !out.includes(url)) {
                        out.push(url);
                    }
                };
                const imgNodes = loginForm ? loginForm.querySelectorAll('img') : document.querySelectorAll('img');
                for (const img of imgNodes) {
                    const sig = (
                        (img.src || '') +
                        (img.alt || '') +
                        (img.title || '') +
                        (img.id || '') +
                        (img.getAttribute('aria-label') || '') +
                        (img.className || '')
                    ).toLowerCase();
                    if (/captcha|验证码|verify|auth|code/.test(sig) && (img.src || '').startsWith('data:image')) {
                        pushUrl(img.src);
                    }
                }
                const canvasNodes = loginForm ? loginForm.querySelectorAll('canvas') : document.querySelectorAll('canvas');
                for (const canvas of canvasNodes) {
                    const sig = ((canvas.id || '') + (canvas.className || '')).toLowerCase();
                    if (/captcha|验证码|verify|auth|code/.test(sig)) {
                        try { pushUrl(canvas.toDataURL('image/png')); } catch (e) {}
                    }
                }
                return out.slice(0, 6);
            }
            """
            urls_raw = await page.evaluate(candidates_script)
            if isinstance(urls_raw, list):
                for item in urls_raw:
                    _add_candidate(item if isinstance(item, str) else None)
        except Exception as e:
            print(f"[CAPTCHA] Candidate collection failed: {e}")

    if not image_data_urls:
        try:
            scope_restricted = "true" if has_scope_hints else "false"
            # JS uses hint_norm to find the anchor input; falls back to general captcha keywords.
            js_hint_pattern = hint_norm.replace("'", "\\'") if hint_norm else ""
            screenshot_script = f"""
            (...args) => {{
                const scopeRestricted = {scope_restricted};
                const hintPattern = '{js_hint_pattern}';
                const isVisible = (node) => {{
                    if (!node) return false;
                    const rect = node.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                }};
                // Locate anchor input: prefer hint-matched input, then captcha-keyword input,
                // then password input (login context), then nothing.
                let anchorInput = null;
                const allInputs = Array.from(document.querySelectorAll('input'));
                if (hintPattern) {{
                    const hintRe = new RegExp(hintPattern, 'i');
                    anchorInput = allInputs.find(inp => {{
                        const sig = ((inp.name||'')+(inp.id||'')+(inp.placeholder||'')+(inp.getAttribute('aria-label')||'')+(inp.className||'')).toLowerCase();
                        return hintRe.test(sig) && isVisible(inp);
                    }}) || null;
                }}
                if (!anchorInput) {{
                    anchorInput = allInputs.find(inp => {{
                        const sig = ((inp.name||'')+(inp.id||'')+(inp.placeholder||'')+(inp.getAttribute('aria-label')||'')+(inp.className||'')).toLowerCase();
                        return /captcha|图形.*码|验证码|verify.*code|auth.*code/.test(sig) && isVisible(inp);
                    }}) || null;
                }}
                // Determine scope root: form that contains the anchor, or document
                const anchorForm = anchorInput && typeof anchorInput.closest === 'function'
                    ? anchorInput.closest('form') : null;
                const scopeRoot = (scopeRestricted && anchorForm) ? anchorForm : document;
                const queryAll = (sel) => Array.from(scopeRoot.querySelectorAll(sel));
                let el = {('document.getElementById("' + captcha_id + '")') if captcha_id else "null"};
                if (el && scopeRestricted && anchorForm && !anchorForm.contains(el)) el = null;
                if (!el) {{
                    for (const img of queryAll('img')) {{
                        const sig = ((img.src||'')+(img.alt||'')+(img.title||'')+(img.id||'')+(img.className||'')).toLowerCase();
                        if (/captcha|验证码|verify|auth|code/i.test(sig) && isVisible(img)) {{ el = img; break; }}
                    }}
                }}
                if (!el) {{
                    for (const cv of queryAll('canvas')) {{
                        const sig = ((cv.id||'')+(cv.className||'')).toLowerCase();
                        if (/captcha|验证码|verify|auth|code/i.test(sig) && isVisible(cv)) {{ el = cv; break; }}
                    }}
                }}
                // Proximity fallback: nearest visible image to the anchor input
                if (!el && anchorInput && isVisible(anchorInput)) {{
                    const a = anchorInput.getBoundingClientRect();
                    const ax = a.left + a.width / 2, ay = a.top + a.height / 2;
                    let best = null, bestD = Infinity;
                    for (const img of queryAll('img')) {{
                        if (!isVisible(img)) continue;
                        const r = img.getBoundingClientRect();
                        const d = Math.hypot(r.left + r.width/2 - ax, r.top + r.height/2 - ay);
                        if (d < bestD) {{ bestD = d; best = img; }}
                    }}
                    if (best) el = best;
                }}
                if (el) {{
                    const rect = el.getBoundingClientRect();
                    return {{
                        x: Math.round(rect.left), y: Math.round(rect.top),
                        width: Math.round(rect.width), height: Math.round(rect.height),
                        scopeRestricted,
                        selectedTag: (el.tagName||'').toLowerCase(),
                        selectedSig: ((el.id||'')+'|'+(el.className||'')+'|'+(el.getAttribute&&el.getAttribute('title')||'')).slice(0,120),
                    }};
                }}
                return null;
            }}
            """
            bbox_raw = await page.evaluate(screenshot_script)
            bbox = (
                cast(dict[str, int], _parse_evaluate_result(bbox_raw))
                if bbox_raw
                else None
            )
            if bbox and bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0:
                bs = page._browser_session
                if bs and hasattr(bs, "cdp_client"):
                    session_id = await page.session_id
                    result = await bs.cdp_client.send.Page.captureScreenshot(
                        {
                            "format": "png",
                            "clip": {
                                "x": bbox["x"],
                                "y": bbox["y"],
                                "width": bbox["width"],
                                "height": bbox["height"],
                                "scale": 1,
                            },
                        },
                        session_id=session_id,
                    )
                    data = result.get("data", "")
                    if data:
                        _add_candidate(f"data:image/png;base64,{data}")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 3 failed: {e}")

    if not image_data_urls and not has_scope_hints:
        try:
            bs = page._browser_session
            if bs and hasattr(bs, "cdp_client"):
                session_id = await page.session_id
                result = await bs.cdp_client.send.Page.captureScreenshot(
                    {"format": "png"},
                    session_id=session_id,
                )
                data = result.get("data", "")
                if data:
                    _add_candidate(f"data:image/png;base64,{data}")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 4 failed: {e}")

    if not image_data_urls:
        print(
            "[CAPTCHA] All strategies failed. Could not extract CAPTCHA image from page."
        )
        return ""

    print(f"[CAPTCHA] Sending {len(image_data_urls)} candidate image(s) to recognizer.")
    for idx, data_url in enumerate(image_data_urls, start=1):
        print(f"[CAPTCHA] Candidate #{idx} data_url={data_url}")
    code = await recognize_captcha_with_candidates(image_data_urls, llm)
    if code:
        print(f"[CAPTCHA] Recognized code: '{code}'")
    else:
        print("[CAPTCHA] Recognizer returned empty code.")
    return code

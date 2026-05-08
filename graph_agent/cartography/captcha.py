from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, cast

from browser_use.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)
from pydantic import BaseModel, Field

from graph_agent.cartography.types import LoginInfo
from graph_agent.lib.page_controller import PageController

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)

_CAPTCHA_SOLVE_MODE_ENV = "CAPTCHA_SOLVE_MODE"
_CAPTCHA_SOLVE_MODES = {"auto", "manual"}


def resolve_captcha_solve_mode(raw_mode: str | None = None) -> str:
    """Resolve captcha solve mode from env or explicit override."""
    candidate = (
        raw_mode
        if raw_mode is not None
        else os.getenv(_CAPTCHA_SOLVE_MODE_ENV, "auto")
    )
    normalized = str(candidate or "").strip().lower()
    if normalized in _CAPTCHA_SOLVE_MODES:
        return normalized
    return "auto"


def normalize_manual_captcha_code(raw_code: str) -> str:
    """Normalize manual captcha input to a safe token."""
    cleaned = str(raw_code or "").replace("`", "").strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned.lower() in {"", "unknown", "n/a"}:
        return ""
    return "".join(ch for ch in cleaned if ch.isalnum())


class CaptchaRecognitionResult(BaseModel):
    """Structured output for captcha image recognition."""

    code: str = Field(
        description=(
            "The exact characters shown in the captcha image. "
            "For plain character/digit captchas return only the characters you see. "
            "For arithmetic captchas return the expression as shown (e.g. '7+1=?') only when "
            "is_arithmetic is true. "
            "Return empty string if the image is not a captcha or is unreadable."
        )
    )
    is_arithmetic: bool = Field(
        default=False,
        description=(
            "True ONLY when the captcha image literally contains a math expression "
            "with an arithmetic operator (+, -, ×, ÷) and a question mark or equals sign. "
            "A plain sequence of digits or letters is NOT arithmetic."
        ),
    )
    confidence: str = Field(
        default="high",
        description="Confidence level: 'high', 'medium', or 'low'.",
    )


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


def _normalize_captcha_code(raw_code: str, is_arithmetic: bool = False) -> str:
    cleaned = raw_code.replace("```", "").replace("`", "").strip()
    if cleaned.lower() in ("unknown", "", "n/a"):
        logger.warning(f"[CAPTCHA] Normalize captcha code failed: {raw_code}")
        return ""
    has_explicit_arithmetic_shape = bool(
        re.search(r"[+\-*/×xX÷]", cleaned)
        and ("=" in cleaned or "?" in cleaned or "？" in cleaned)
    )
    # Prefer LLM flag, but still evaluate when expression shape is explicit.
    # This keeps plain 4-digit codes safe while handling direct helper calls
    # like `_normalize_captcha_code("9+8=?")` in tests.
    if is_arithmetic or has_explicit_arithmetic_shape:
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


def _build_captcha_content(
    image_data_urls: list[str],
    system_text: str,
) -> list[ContentPartTextParam | ContentPartImageParam]:
    """Build a multimodal content list (text + images) for captcha recognition.

    Note: ImageURL is constructed WITHOUT the ``detail`` field because Kimi's
    API does not support it and rejects requests that include it.
    """
    content: list[ContentPartTextParam | ContentPartImageParam] = [
        ContentPartTextParam(text=system_text)
    ]
    for idx, data_url in enumerate(image_data_urls, start=1):
        content.append(ContentPartTextParam(text=f"Candidate image #{idx}:"))
        # Do NOT pass detail= here — Kimi rejects unknown image_url fields.
        content.append(ContentPartImageParam(image_url=ImageURL(url=data_url)))
    return content


async def recognize_captcha_with_candidates(
    image_data_urls: list[str],
    llm: "BaseChatModel",
) -> str:
    """Recognise a captcha from one or more candidate image data-URLs.

    Uses json_schema structured output (CaptchaRecognitionResult) so the model
    returns a well-formed JSON object instead of free text.  Falls back to
    plain-text parsing if the provider does not support structured output.
    """
    normalized_urls = [
        u for u in image_data_urls if isinstance(u, str) and u.startswith("data:image")
    ]
    if not normalized_urls:
        return ""
    logger.info(
        "[CAPTCHA] Sending %d image(s) to LLM for recognition.", len(normalized_urls)
    )
    for i, u in enumerate(normalized_urls, 1):
        logger.info("[CAPTCHA]   image #%d: %s... (total %d chars)", i, u[:60], len(u))
    try:
        system_text = (
            "You are a CAPTCHA solver. "
            "You will receive one or more candidate images. "
            "Identify which image contains the captcha, then read it carefully. "
            "Rules:\n"
            "- MOST captchas are plain character/digit codes (4-6 chars). "
            "Return the exact characters. "
            "Watch for similar-looking chars: 0/O, 1/l/I, 5/S, 8/B.\n"
            "- Set is_arithmetic=true ONLY when the image literally shows a "
            "math expression with an arithmetic operator (+, -, ×, ÷) AND a question mark "
            "or equals sign (e.g. '9+3=?', '7-2=?'). "
            "A plain sequence of digits or letters is NOT arithmetic — return it exactly as you see it.\n"
            "- If no captcha is found or image is unreadable: return code as empty string.\n"
            "- Ignore logos, banners, QR codes, and decorative images."
        )
        content = _build_captcha_content(normalized_urls, system_text)
        messages: list[BaseMessage] = [UserMessage(content=content)]

        # Use structured output when supported; fall back to plain text.
        is_arithmetic = False
        recognition_path = "structured"
        try:
            result = await llm.ainvoke(messages, output_format=CaptchaRecognitionResult)
            recognition: CaptchaRecognitionResult = result.completion  # type: ignore[assignment]
            raw_code = recognition.code
            is_arithmetic = recognition.is_arithmetic
            logger.info(
                f"[CAPTCHA] Structured result: code={raw_code!r} "
                f"arithmetic={is_arithmetic} confidence={recognition.confidence}"
            )
        except Exception as structured_err:
            logger.warning(
                f"[CAPTCHA] Structured output failed ({structured_err}), falling back to plain text."
            )
            recognition_path = "plain_fallback"
            result_plain = await llm.ainvoke(messages)
            raw_code = str(result_plain.completion or "")
            logger.info("[CAPTCHA] Plain-text result: %r", raw_code)

        normalized_code = _normalize_captcha_code(raw_code, is_arithmetic=is_arithmetic)
        # If the model returned an expression with '=?' but no operator yet
        # (e.g. plain-text fallback returned "9?"), retry with an arithmetic prompt.
        if _needs_arithmetic_retry(raw_code):
            recognition_path = (
                "structured+arith_retry"
                if recognition_path == "structured"
                else "plain_fallback+arith_retry"
            )
            arith_text = (
                "You are reading an arithmetic CAPTCHA. "
                "Return the mathematical expression exactly as shown, "
                "e.g. '7+1=?' or '9-3=?'. "
                "If it is not arithmetic, return UNKNOWN."
            )
            arith_msgs: list[BaseMessage] = [
                UserMessage(content=_build_captcha_content(normalized_urls, arith_text))
            ]
            try:
                arith_result = await llm.ainvoke(
                    arith_msgs,
                    output_format=CaptchaRecognitionResult,
                )
                arith_recognition: CaptchaRecognitionResult = arith_result.completion  # type: ignore[assignment]
                retry_normalized = _normalize_captcha_code(
                    arith_recognition.code,
                    is_arithmetic=arith_recognition.is_arithmetic,
                )
            except Exception:
                arith_plain = await llm.ainvoke(arith_msgs)
                retry_normalized = _normalize_captcha_code(
                    str(arith_plain.completion or ""), is_arithmetic=True
                )
            if retry_normalized:
                normalized_code = retry_normalized

        logger.info(
            "[CAPTCHA] recognition_path=%s candidate_count=%d normalized_code_len=%d",
            recognition_path,
            len(normalized_urls),
            len(normalized_code),
        )
        return normalized_code
    except Exception as e:
        logger.warning("[CAPTCHA] LLM recognition failed: %s", e)
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
    logger.info(
        f"[CAPTCHA] solve_captcha_from_page called. tag={captcha_tag}, "
        f"id={captcha_id}, hint='{hint_norm}', src={captcha_src if captcha_src else 'empty'}"
    )

    image_data_urls: list[str] = []
    has_scope_hints = False  # True once we have form/input anchor xpaths
    strategy_hit = "none"

    def _add_candidate(data_url: str | None, *, source: str = "") -> None:
        nonlocal strategy_hit
        if not isinstance(data_url, str):
            return
        candidate = data_url.strip()
        if not candidate.startswith("data:image"):
            return
        if candidate not in image_data_urls:
            image_data_urls.append(candidate)
            if strategy_hit == "none" and source:
                strategy_hit = source

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
                        _add_candidate(message, source="strategy0_selector_map")
                    else:
                        _add_candidate(
                            f"data:image/png;base64,{message}",
                            source="strategy0_selector_map",
                        )
    except Exception as e:
        logger.warning("[CAPTCHA] Strategy 0 (browser-use selector_map) failed: %s", e)

    if not image_data_urls and not has_scope_hints and captcha_tag == "img":
        logger.info(
            f"[CAPTCHA] Strategy 1: captcha is <img>. src={captcha_src if captcha_src else 'empty'}"
        )
        if captcha_src.startswith("data:image"):
            _add_candidate(captcha_src, source="strategy1_img_src")
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
                _add_candidate(
                    await page.evaluate(fetch_script), source="strategy1_img_fetch"
                )
            except Exception as e:
                logger.warning("[CAPTCHA] Strategy 1 failed: %s", e)

    if not image_data_urls and captcha_tag == "canvas":
        try:
            canvas_script = f"""
            (...args) => {{
                var canvas = document.getElementById('{captcha_id}') || document.querySelector('canvas');
                if (canvas) return canvas.toDataURL('image/png');
                return null;
            }}
            """
            _add_candidate(
                await page.evaluate(canvas_script), source="strategy2_canvas"
            )
        except Exception as e:
            logger.warning("[CAPTCHA] Strategy 2 failed: %s", e)

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
                    _add_candidate(
                        item if isinstance(item, str) else None,
                        source="candidate_collection",
                    )
        except Exception as e:
            logger.warning("[CAPTCHA] Candidate collection failed: %s", e)

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
                        _add_candidate(
                            f"data:image/png;base64,{data}",
                            source="strategy3_clip_screenshot",
                        )
        except Exception as e:
            logger.warning("[CAPTCHA] Strategy 3 failed: %s", e)

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
                    _add_candidate(
                        f"data:image/png;base64,{data}",
                        source="strategy4_full_screenshot",
                    )
        except Exception as e:
            logger.warning("[CAPTCHA] Strategy 4 failed: %s", e)

    if not image_data_urls:
        logger.warning(
            "[CAPTCHA] All strategies failed. Could not extract CAPTCHA image from page."
        )
        return ""

    logger.info(
        "[CAPTCHA] Sending %d candidate image(s) to recognizer. strategy_hit=%s",
        len(image_data_urls),
        strategy_hit,
    )
    for idx, data_url in enumerate(image_data_urls, start=1):
        logger.debug("[CAPTCHA] Candidate #%d data_url=%s", idx, data_url)
    code = await recognize_captcha_with_candidates(image_data_urls, llm)
    if code:
        logger.info("[CAPTCHA] Recognized code: '%s'", code)
    else:
        logger.info("[CAPTCHA] Recognizer returned empty code.")
    return code

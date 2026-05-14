from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from browser_use.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)

from graph_agent.cartography.captcha.config_dom import CaptchaRecognitionResult

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


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
            result = await llm.ainvoke(messages, output_format=CaptchaRecognitionResult)  # type: ignore[arg-type]
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
            result_plain = await llm.ainvoke(messages)  # type: ignore[arg-type]
            raw_code = str(getattr(result_plain, "completion", "") or "")
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
                    arith_msgs,  # type: ignore[arg-type]
                    output_format=CaptchaRecognitionResult,
                )
                arith_recognition: CaptchaRecognitionResult = arith_result.completion  # type: ignore[assignment]
                retry_normalized = _normalize_captcha_code(
                    arith_recognition.code,
                    is_arithmetic=arith_recognition.is_arithmetic,
                )
            except Exception:
                arith_plain = await llm.ainvoke(arith_msgs)  # type: ignore[arg-type]
                retry_normalized = _normalize_captcha_code(
                    str(getattr(arith_plain, "completion", "") or ""), is_arithmetic=True
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


def extract_captcha_result_code(result_text: str) -> str:
    """Extract CAPTCHA_* status code from result text."""
    match = re.search(r"(CAPTCHA_[A-Z_]+)", str(result_text or "").upper())
    return match.group(1) if match else "CAPTCHA_UNKNOWN"


def extract_captcha_fill_path(result_text: str) -> str:
    """Determine how captcha was filled: input_index, semantic_match, fill_failed, or none."""
    text = str(result_text or "").lower()
    if "filled_index=" in text:
        return "input_index"
    if "filled_by_semantic_match" in text:
        return "semantic_match"
    if "fill_failed" in text:
        return "fill_failed"
    return "none"



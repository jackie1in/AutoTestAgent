from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass

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



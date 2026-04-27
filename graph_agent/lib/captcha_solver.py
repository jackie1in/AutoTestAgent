from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

_OCR_ENGINE: Any | None = None


def _get_ocr_engine() -> Any:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        import ddddocr  # type: ignore[import-not-found]

        _OCR_ENGINE = ddddocr.DdddOcr(show_ad=False)
    return _OCR_ENGINE


def _normalize_code(raw_text: str) -> str:
    text = (raw_text or "").replace("```", "").replace("`", "").strip()
    return "".join(ch for ch in text if ch.isalnum())


def _to_image_bytes(image_data: str) -> bytes:
    if not image_data:
        return b""
    payload = image_data
    if image_data.startswith("data:image"):
        if "," not in image_data:
            return b""
        payload = image_data.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except Exception:
        return b""


def solve_with_ddddocr(image_data_url: str) -> str:
    """Recognize captcha text with ddddocr from data URL/base64."""
    img_bytes = _to_image_bytes(image_data_url)
    if not img_bytes:
        return ""
    try:
        engine = _get_ocr_engine()
        raw = engine.classification(img_bytes)
        return _normalize_code(str(raw))
    except Exception as exc:
        logger.warning("ddddocr failed: %s", exc)
        return ""

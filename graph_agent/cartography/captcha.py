from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)

from graph_agent.cartography.config import env_bool
from graph_agent.cartography.types import LoginInfo
from graph_agent.lib.captcha_solver import solve_with_ddddocr

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.llm.base import BaseChatModel


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
    dddd_enabled = env_bool("CAPTCHA_DDDDOCR_ENABLED", True)
    dddd_only = env_bool("CAPTCHA_DDDDOCR_ONLY", False)

    if dddd_enabled:
        dddd_code = solve_with_ddddocr(image_data_url)
        if dddd_code and 3 <= len(dddd_code) <= 8:
            print(f"[CAPTCHA][ddddocr] recognized code: {dddd_code}")
            return dddd_code
        if dddd_code and dddd_only:
            print(f"[CAPTCHA][ddddocr] only-mode uses code: {dddd_code}")
            return dddd_code
        if dddd_only:
            print("[CAPTCHA][ddddocr] only-mode failed to recognize code.")
            return ""

    print("[CAPTCHA][fallback-llm] using LLM vision for captcha.")
    try:
        messages = [
            UserMessage(
                content=[
                    ContentPartTextParam(
                        text=(
                            "You are a CAPTCHA solver. Look at the image carefully and return ONLY the "
                            "exact characters/numbers/letters shown in the CAPTCHA image. "
                            "The CAPTCHA is usually 4-6 characters. Pay attention to: "
                            "- Similar looking characters (0 vs O, 1 vs l vs I, 5 vs S, 8 vs B) "
                            "- Case sensitivity (uppercase vs lowercase letters) "
                            "- Do NOT guess; if truly unreadable, return 'UNKNOWN'. "
                            "Return ONLY the characters, no explanation, no quotes, no markdown."
                        )
                    ),
                    ContentPartImageParam(
                        image_url=ImageURL(url=image_data_url, detail="high")
                    ),
                ]
            )
        ]
        result = await llm.ainvoke(messages)
        raw_code = str(result.completion or "").strip()
        code = raw_code.replace("```", "").replace("`", "").strip()
        if code.lower() in ("unknown", "", "n/a"):
            return ""
        return "".join(ch for ch in code if ch.isalnum())
    except Exception as e:
        print(f"[CAPTCHA] LLM recognition failed with exception: {e}")
        return ""


async def solve_captcha_from_page(
    page: "Page",
    login_info: LoginInfo,
    llm: "BaseChatModel",
) -> str:
    captcha_tag = login_info.get("captchaTag", "")
    captcha_id = login_info.get("captchaId", "")
    captcha_src = login_info.get("captchaSrc", "")
    print(
        f"[CAPTCHA] solve_captcha_from_page called. tag={captcha_tag}, "
        f"id={captcha_id}, src={captcha_src[:60] if captcha_src else 'empty'}..."
    )

    image_data_url: str | None = None

    if captcha_tag == "img":
        print(f"[CAPTCHA] Strategy 1: captcha is <img>. src={captcha_src[:60] if captcha_src else 'empty'}...")
        if captcha_src.startswith("data:image"):
            image_data_url = captcha_src
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
                image_data_url = await page.evaluate(fetch_script)
            except Exception as e:
                print(f"[CAPTCHA] Strategy 1 failed: {e}")

    if not image_data_url and captcha_tag == "canvas":
        try:
            canvas_script = f"""
            (...args) => {{
                var canvas = document.getElementById('{captcha_id}') || document.querySelector('canvas');
                if (canvas) return canvas.toDataURL('image/png');
                return null;
            }}
            """
            image_data_url = await page.evaluate(canvas_script)
        except Exception as e:
            print(f"[CAPTCHA] Strategy 2 failed: {e}")

    if not image_data_url:
        try:
            screenshot_script = f"""
            (...args) => {{
                var el = document.getElementById('{captcha_id}');
                if (!el) {{
                    var imgs = document.querySelectorAll('img');
                    for (var i = 0; i < imgs.length; i++) {{
                        if (/captcha|验证码|verify|auth|code/i.test(imgs[i].src + imgs[i].alt + imgs[i].id + imgs[i].className)) {{
                            el = imgs[i];
                            break;
                        }}
                    }}
                }}
                if (!el) {{
                    var canvases = document.querySelectorAll('canvas');
                    for (var j = 0; j < canvases.length; j++) {{
                        if (/captcha|验证码|verify|auth|code/i.test(canvases[j].id + canvases[j].className)) {{
                            el = canvases[j];
                            break;
                        }}
                    }}
                }}
                if (el) {{
                    var rect = el.getBoundingClientRect();
                    return {{
                        x: Math.round(rect.left),
                        y: Math.round(rect.top),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    }};
                }}
                return null;
            }}
            """
            bbox_raw = await page.evaluate(screenshot_script)
            bbox = cast(dict[str, int], _parse_evaluate_result(bbox_raw)) if bbox_raw else None
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
                        image_data_url = f"data:image/png;base64,{data}"
        except Exception as e:
            print(f"[CAPTCHA] Strategy 3 failed: {e}")

    if not image_data_url:
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
                    image_data_url = f"data:image/png;base64,{data}"
        except Exception as e:
            print(f"[CAPTCHA] Strategy 4 failed: {e}")

    if not image_data_url:
        print("[CAPTCHA] All strategies failed. Could not extract CAPTCHA image from page.")
        return ""

    print(f"[CAPTCHA] Sending image to recognizer. data_url length={len(image_data_url)}")
    code = await recognize_captcha_with_fallback(image_data_url, llm)
    if code:
        print(f"[CAPTCHA] Recognized code: '{code}'")
    else:
        print("[CAPTCHA] Recognizer returned empty code.")
    return code

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)

from graph_agent.cartography.types import LoginInfo

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
    return await recognize_captcha_with_candidates([image_data_url], llm)


def _normalize_captcha_code(raw_code: str) -> str:
    cleaned = raw_code.replace("```", "").replace("`", "").strip()
    if cleaned.lower() in ("unknown", "", "n/a"):
        return ""
    return "".join(ch for ch in cleaned if ch.isalnum())


async def recognize_captcha_with_candidates(
    image_data_urls: list[str],
    llm: "BaseChatModel",
) -> str:
    normalized_urls = [u for u in image_data_urls if isinstance(u, str) and u.startswith("data:image")]
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
            content.append(ContentPartImageParam(image_url=ImageURL(url=data_url, detail="high")))

        messages = [
            UserMessage(
                content=content
            )
        ]
        result = await llm.ainvoke(messages)
        raw_code = str(result.completion or "")
        return _normalize_captcha_code(raw_code)
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
        f"id={captcha_id}, src={captcha_src if captcha_src else 'empty'}"
    )

    image_data_urls: list[str] = []

    def _add_candidate(data_url: str | None) -> None:
        if not isinstance(data_url, str):
            return
        candidate = data_url.strip()
        if not candidate.startswith("data:image"):
            return
        if candidate not in image_data_urls:
            image_data_urls.append(candidate)

    if captcha_tag == "img":
        print(f"[CAPTCHA] Strategy 1: captcha is <img>. src={captcha_src if captcha_src else 'empty'}")
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
                const pushUrl = (url) => {
                    if (typeof url === 'string' && url.startsWith('data:image') && !out.includes(url)) {
                        out.push(url);
                    }
                };
                for (const img of document.querySelectorAll('img')) {
                    const sig = ((img.src || '') + (img.alt || '') + (img.id || '') + (img.className || '')).toLowerCase();
                    if (/captcha|验证码|verify|auth|code/.test(sig) && (img.src || '').startsWith('data:image')) {
                        pushUrl(img.src);
                    }
                }
                for (const canvas of document.querySelectorAll('canvas')) {
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
                        _add_candidate(f"data:image/png;base64,{data}")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 3 failed: {e}")

    if not image_data_urls:
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
        print("[CAPTCHA] All strategies failed. Could not extract CAPTCHA image from page.")
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

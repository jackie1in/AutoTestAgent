"""Run browser-use Agent for mapping: explore flow and save to Neo4j."""

from __future__ import annotations

import json
import os
import asyncio
import hashlib
import re
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from graph_agent.llm import get_llm
from graph_agent.llm.utils import ainvoke_structured
from browser_use.llm.messages import UserMessage, ContentPartTextParam, ContentPartImageParam, ImageURL
from graph_agent.lib.captcha_solver import solve_with_ddddocr

# Global registry for active browser sessions (for cleanup on Ctrl+C)
_active_browsers: list[Any] = []
_shutdown_requested = False


def _register_browser(browser: Any) -> None:
    """Register a browser instance for cleanup on shutdown."""
    if browser not in _active_browsers:
        _active_browsers.append(browser)


def _unregister_browser(browser: Any) -> None:
    """Unregister a browser instance after cleanup."""
    if browser in _active_browsers:
        _active_browsers.remove(browser)


async def _cleanup_all_browsers() -> None:
    """Clean up all registered browser sessions using kill() API."""
    global _active_browsers
    if not _active_browsers:
        return
    
    print(f"\n[INFO] Cleaning up {len(_active_browsers)} browser session(s)...")
    for browser in list(_active_browsers):
        try:
            # Use kill() API for forceful cleanup (browser-use recommended)
            if hasattr(browser, "kill"):
                await browser.kill()
                print("  [OK] Browser killed")
            elif hasattr(browser, "stop"):
                await browser.stop()
                print("  [OK] Browser stopped")
            elif hasattr(browser, "close"):
                await browser.close()
                print("  [OK] Browser closed")
        except Exception as e:
            print(f"  [WARN] Error during browser cleanup: {e}")
        finally:
            _unregister_browser(browser)
    print("[INFO] Browser cleanup complete")


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C (SIGINT) and SIGTERM signals."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[FORCE] Force exit requested")
        sys.exit(1)
    
    _shutdown_requested = True
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[INFO] Received {signal_name}, shutting down gracefully...")
    print("[INFO] Press Ctrl+C again to force exit")
    
    # Note: We can't do async cleanup here, so we set a flag
    # The main loop should check _shutdown_requested


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@asynccontextmanager
async def managed_browser(browser: Any):
    """Context manager for browser lifecycle with cleanup on exit."""
    _register_browser(browser)
    try:
        yield browser
    finally:
        try:
            # Use kill() API for forceful cleanup (browser-use recommended)
            if hasattr(browser, "kill"):
                await browser.kill()
            elif hasattr(browser, "stop"):
                await browser.stop()
            elif hasattr(browser, "close"):
                await browser.close()
        except Exception as e:
            print(f"[WARN] Browser cleanup error: {e}")
        finally:
            _unregister_browser(browser)

def _clean_url(url: str) -> str:
    """Strip query parameters and hash fragments from URL to ensure stable Node IDs."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        # Keep scheme, netloc, path. Drop params, query, fragment.
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return url


def _resolve_mapping_url(url: str | None) -> str:
    """Resolve mapping URL: function arg first, then MAPPING_URL env, else raise."""
    value = (url or "").strip()
    if value:
        return value
    env_value = (os.getenv("MAPPING_URL") or "").strip()
    if env_value:
        return env_value
    raise ValueError("url is required. Provide --url or set MAPPING_URL.")


def _resolve_mapping_headless() -> bool:
    """Resolve mapping headless mode from MAPPING_HEADLESS env."""
    raw = (os.getenv("MAPPING_HEADLESS") or "").strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_mapping_channel() -> str | None:
    """Resolve optional browser channel from MAPPING_CHANNEL env."""
    raw = (os.getenv("MAPPING_CHANNEL") or "").strip()
    return raw or None


def _setup_browser_use_timeouts():
    """Setup browser-use timeout environment variables from MAPPING_TIMEOUT.
    
    browser-use's _navigate_and_wait has hardcoded 8s timeout, but we can
    increase the overall event timeout to give more time for slow pages.
    """
    mapping_timeout = os.getenv("MAPPING_TIMEOUT", "").strip()
    if mapping_timeout:
        try:
            timeout_val = float(mapping_timeout)
            # Set browser-use timeout environment variables
            os.environ.setdefault("TIMEOUT_NavigateToUrlEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStartEvent", str(timeout_val))
        except ValueError:
            pass


# =============================================================================
# LLM-first orchestration models  (replaces MenuExtractor + ZoneDiscoverer +
# CoverageAnalyzer + ExplorationScheduler with LLM-driven decisions)
# =============================================================================

class _LLMMenuItem(BaseModel):
    text: str = Field(description="Menu item display text")
    href: str = Field(default="", description="Link URL or route path")
    level: int = Field(default=0, description="Hierarchy level (0 = top level)")


class _LLMFunctionalZone(BaseModel):
    zone_type: str = Field(
        description="Zone type: form, table, nav, action_bar, filter, modal, card, tabs, chart, list, content"
    )
    selector: str = Field(description="Best CSS selector to locate the zone")
    description: str = Field(default="", description="Brief description of what this zone does")


class LLMPageAnalysis(BaseModel):
    """Structured output for LLM-driven page analysis."""

    page_type: str = Field(
        description="Page type: login, dashboard, list, detail, form, settings, welcome, unknown"
    )
    menu_items: list[_LLMMenuItem] = Field(
        default_factory=list, description="Navigation menu items found on the page"
    )
    functional_zones: list[_LLMFunctionalZone] = Field(
        default_factory=list, description="Functional zones / regions on the page"
    )
    is_login_page: bool = Field(default=False, description="Whether this is a login/authentication page")
    reasoning: str = Field(default="", description="Brief reasoning for the analysis")


class _LLMExplorationTask(BaseModel):
    task_type: str = Field(
        description="Type: explore_page, explore_zone, click_menu, validate_transition, stop"
    )
    target_url: str = Field(default="", description="Target URL or empty for current page")
    target_selector: str = Field(default="", description="CSS selector for zone or menu item")
    description: str = Field(default="", description="What to do in this task")
    expected_outcome: str = Field(default="", description="Expected page state after completing the task")


class LLMExplorationPlan(BaseModel):
    """Structured output for LLM-driven exploration planning."""

    tasks: list[_LLMExplorationTask] = Field(
        default_factory=list, description="Ordered list of exploration tasks"
    )
    strategy: str = Field(
        default="continue",
        description="Overall strategy recommendation: continue | pivot | consolidate | stop",
    )
    coverage_estimate: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated coverage 0.0-1.0")
    reasoning: str = Field(default="", description="Reasoning for the plan")


async def _analyze_page_with_llm(
    llm: Any,
    dom_text: str,
    current_url: str,
    page_title: str = "",
) -> LLMPageAnalysis:
    """Use a single LLM call to analyze page structure (LLM-first replacement for
    MenuExtractor + ZoneDiscoverer + login-detection heuristics).
    """
    system_prompt = (
        "You are an expert web UI analyzer. Given a DOM text representation of a web page, "
        "analyze its structure and produce a structured exploration plan:\n"
        "  (1) navigation menu items, listed STRICTLY in the order they appear in the "
        "      DOM / visual menu bar (left-to-right for a horizontal nav, top-to-bottom "
        "      for a sidebar). Do NOT reorder by importance; this list drives FIFO "
        "      exploration order.\n"
        "  (2) functional zones, also listed in DOM order (top-to-bottom as they appear "
        "      on the page).\n"
        "  (3) whether it is a login page.\n"
        "  (4) page_type: login | dashboard | list | detail | form | settings | welcome | unknown.\n"
        "Be precise with CSS selectors. Ignore decorative wrappers and plain text blocks."
    )
    user_prompt = (
        f"Current URL: {current_url}\n"
        f"Page Title: {page_title}\n\n"
        f"DOM representation (first 12000 chars):\n{dom_text[:12000]}\n\n"
        "Analyze this page and return structured results. Preserve the menu-bar / "
        "DOM order when listing menu_items and functional_zones — this ordering is "
        "how a human QA tester walks through the page."
    )

    try:
        result = await ainvoke_structured(
            llm,
            system_prompt,
            user_prompt,
            LLMPageAnalysis,
            max_retries=2,
            timeout_ms=45_000,
        )
        return result
    except Exception as e:
        print(f"[LLM-ORCH] Page analysis failed: {e}. Falling back to empty analysis.")
        return LLMPageAnalysis(
            page_type="unknown",
            menu_items=[],
            functional_zones=[],
            is_login_page=False,
            reasoning=f"Analysis failed: {e}",
        )


# =============================================================================
# Page-type action policy
#
# Gives the ReActExplorer a short, page-type-specific hint about WHAT kinds
# of interactions are valuable here (e.g. fill fields on a form, click rows
# on a list). Strategy selection has been removed — the orchestrator always
# walks menus FIFO, and ReActExplorer always explores the CURRENT page then
# calls `done`, letting the orchestrator proceed to the next queued menu.
# =============================================================================

_PAGE_TYPE_ACTION_POLICY: dict[str, str] = {
    "login": (
        "ACTION POLICY: Login page. If credentials are available via the "
        "auto-login hint, use them. Otherwise inventory the form fields "
        "(input/password/captcha) and call done."
    ),
    "form": (
        "ACTION POLICY: Form page. Prefer `input` over `click`. Fill every "
        "text/number/email/password field with a plausible sample value, "
        "select defaults for dropdowns, check required checkboxes, then "
        "click the primary submit button. Call done after submission."
    ),
    "list": (
        "ACTION POLICY: List/table page. Prefer `click` on table rows and "
        "row-level action buttons (view / edit / delete). Try at least the "
        "first 3 visible rows. Also click pagination / sort / filter controls."
    ),
    "detail": (
        "ACTION POLICY: Detail page. Click every sub-tab, expand every "
        "collapsible section, and try primary actions (edit, delete, export). "
        "Do NOT navigate back until the page is fully inspected."
    ),
    "dashboard": (
        "ACTION POLICY: Dashboard / portal. Menu navigation is handled by the "
        "orchestrator, NOT by you. Focus only on in-page content: KPI cards, "
        "chart legends, quick-action buttons inside the dashboard body. Do "
        "NOT click top / side navigation menu items. Do NOT type into search "
        "boxes. Call done once the main dashboard widgets have been sampled."
    ),
    "settings": (
        "ACTION POLICY: Settings page. Toggle every switch, change every "
        "dropdown to a non-default value, and observe state changes. Save at "
        "the end if there is a save button."
    ),
    "welcome": (
        "ACTION POLICY: Welcome / landing page. Click the primary CTA; ignore "
        "marketing copy and footer links."
    ),
}


_UNIVERSAL_EXPLORER_RULES = (
    "GENERAL RULES (apply to every page):\n"
    "  - Your job is to thoroughly exercise the CURRENT page's feature, then "
    "call `done`. The orchestrator will automatically pick up the next menu / "
    "page after you finish.\n"
    "  - Do NOT repeatedly click navigation menus to jump between features. "
    "Click a menu item ONCE to record its target, then move on — cross-page "
    "traversal is the orchestrator's job, not yours.\n"
    "  - Cross-domain rule: if a click navigates to a different domain (SSO "
    "portal, third-party docs, external site), immediately call `done` so the "
    "orchestrator can close that tab. Never use `go_back` to escape — just "
    "`done`.\n"
    "  - If the page has not changed after an action, continue with other "
    "elements on the SAME page; do not use `go_back`."
)


def _build_exploration_guidance(page_analysis: LLMPageAnalysis) -> str:
    """Build extra system-prompt text for ReActExplorer.

    Combines:
      (a) universal rules (done-after-feature, no-cross-page-jump, cross-domain),
      (b) page-type action policy,
      (c) zone hint in DOM / LLM-returned order (top 5),
      (d) menu hint in DOM / LLM-returned order (only on dashboard pages).
    """
    blocks: list[str] = [_UNIVERSAL_EXPLORER_RULES]

    policy_text = _PAGE_TYPE_ACTION_POLICY.get(page_analysis.page_type)
    if policy_text:
        blocks.append(policy_text)

    if page_analysis.functional_zones:
        zone_lines = [
            f"  {i+1}. [{z.zone_type}] {z.selector}"
            + (f" — {z.description}" if z.description else "")
            for i, z in enumerate(page_analysis.functional_zones[:5])
        ]
        blocks.append(
            "ZONE ORDER (explore in this sequence, as they appear on the page):\n"
            + "\n".join(zone_lines)
        )

    return "\n\n".join(blocks)


async def _plan_next_exploration_with_llm(
    llm: Any,
    current_url: str,
    page_title: str,
    page_analysis: LLMPageAnalysis,
    explored_urls: list[str],
    discovered_transitions: list[dict[str, Any]],
    time_budget_remaining_ms: float,
    max_steps_remaining: int,
) -> LLMExplorationPlan:
    """Use LLM to plan the next batch of exploration tasks (LLM-first replacement
    for CoverageAnalyzer + ExplorationScheduler).
    """
    # Build a compact context summary
    explored_summary = "\n".join(
        f"  - {url}"
        for url in explored_urls[-20:]
    ) or "  (none yet)"

    transition_summary = "\n".join(
        f"  - {t.get('action', '?')} on {t.get('selector', '?')} -> {t.get('to_url', '?')[:60]}"
        for t in discovered_transitions[-15:]
    ) or "  (none yet)"

    menu_summary = "\n".join(
        f"  - [{m.level}] {m.text} ({m.href or 'no href'})"
        for m in page_analysis.menu_items[:15]
    ) or "  (none detected)"

    zone_summary = "\n".join(
        f"  - [{z.zone_type}] {z.selector}"
        for z in page_analysis.functional_zones[:15]
    ) or "  (none detected)"

    system_prompt = (
        "You are an expert exploration planner for web application cartography. "
        "Given the current page analysis and exploration history, decide the next "
        "most valuable tasks to perform. Balance coverage and depth. "
        "Recommend stopping only when coverage is high AND remaining budget is low."
    )
    user_prompt = (
        f"Current URL: {current_url}\n"
        f"Page Title: {page_title}\n"
        f"Page Type: {page_analysis.page_type}\n\n"
        f"Time budget remaining: {time_budget_remaining_ms / 1000:.0f}s\n"
        f"Max steps remaining: {max_steps_remaining}\n\n"
        f"Menu items detected:\n{menu_summary}\n\n"
        f"Functional zones detected:\n{zone_summary}\n\n"
        f"Already explored URLs ({len(explored_urls)} total):\n{explored_summary}\n\n"
        f"Recent transitions:\n{transition_summary}\n\n"
        "Plan the next exploration tasks. Return an ordered list of tasks."
    )

    try:
        result = await ainvoke_structured(
            llm,
            system_prompt,
            user_prompt,
            LLMExplorationPlan,
            max_retries=2,
            timeout_ms=45_000,
        )
        return result
    except Exception as e:
        print(f"[LLM-ORCH] Exploration planning failed: {e}. Falling back to single-page stop.")
        return LLMExplorationPlan(
            tasks=[_LLMExplorationTask(task_type="stop", description=f"Planning failed: {e}")],
            strategy="stop",
            coverage_estimate=0.0,
            reasoning=f"Planning failed: {e}",
        )


def _build_login_hint_from_env() -> str:
    """Build optional login hint from env; empty when no credentials configured."""
    username = (os.getenv("MAPPING_USERNAME") or "").strip()
    password = (os.getenv("MAPPING_PASSWORD") or "").strip()
    if not username and not password:
        return ""

    parts: list[str] = []
    if username:
        parts.append(f"username={username}")
    if password:
        parts.append(f"password={password}")
    credentials = ", ".join(parts)
    return (
        "若页面包含登录表单，优先使用以下测试账号完成登录："
        f"{credentials}。"
        "如字段名不同，请根据语义匹配对应输入框。"
    )


def _is_http_url(value: str) -> bool:
    """Return True if value looks like a stable http(s) URL."""
    v = (value or "").strip()
    return v.startswith("http://") or v.startswith("https://")


def _same_origin(a: str, b: str) -> bool:
    """Return True iff two URLs share scheme and host (case-insensitive on host).

    Empty / non-http URLs never match. Ports are compared as-is (treated as
    part of netloc by urlparse).
    """
    if not a or not b:
        return False
    try:
        from urllib.parse import urlparse

        pa, pb = urlparse(a), urlparse(b)
        if not pa.scheme or not pb.scheme or not pa.netloc or not pb.netloc:
            return False
        return (
            pa.scheme.lower() == pb.scheme.lower()
            and pa.netloc.lower() == pb.netloc.lower()
        )
    except Exception:
        return False


def _load_inventory(inventory_path: str | Path) -> list[dict]:
    """Load elements list from inventory JSON; raise if file missing or invalid."""
    path = Path(inventory_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {path}. Run scout first: uv run python -m graph_agent.cartography.runner --url <url> (with --inventory and --output)."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    elements = raw.get("elements") if isinstance(raw, dict) else []
    return elements if isinstance(elements, list) else []


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


async def _recognize_captcha_with_fallback(image_data_url: str, llm: Any) -> str:
    """Try ddddocr first, then fallback to LLM vision."""
    dddd_enabled = _env_bool("CAPTCHA_DDDDOCR_ENABLED", True)
    dddd_only = _env_bool("CAPTCHA_DDDDOCR_ONLY", False)

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


def rank_warm_start_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank warm-start candidates by exploration value."""
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.get("zone_unexplored") else 1,
            -(float(item.get("confidence") or 0.0)),
            str(item.get("transition_id") or ""),
        ),
    )


async def _solve_captcha_with_llm(
    page: Any,
    login_info: dict[str, Any],
    llm: Any,
) -> str:
    """Extract CAPTCHA image from page and use LLM vision to solve it.

    Returns the recognized CAPTCHA code string, or empty string on failure.
    """
    captcha_tag = login_info.get("captchaTag", "")
    captcha_id = login_info.get("captchaId", "")
    captcha_src = login_info.get("captchaSrc", "")
    print(f"[CAPTCHA] _solve_captcha_with_llm called. tag={captcha_tag}, id={captcha_id}, src={captcha_src[:60] if captcha_src else 'empty'}...")

    # Strategy 1: If captcha is an <img> with a data URL or direct URL, fetch it
    image_data_url: str | None = None

    if captcha_tag == "img":
        print(f"[CAPTCHA] Strategy 1: captcha is <img>. src={captcha_src[:60] if captcha_src else 'empty'}...")
        if captcha_src.startswith("data:image"):
            image_data_url = captcha_src
            print(f"[CAPTCHA] Strategy 1 success: got data URL, length={len(image_data_url)}")
        elif captcha_src:
            # Try to fetch the image and convert to base64
            try:
                # NOTE: browser-use's Page.evaluate() checks
                #   page_function.startswith('(') and '=>' in page_function
                # so ``async (...args) => ...`` is rejected (starts with 'a').
                # Use a sync arrow that returns a Promise; evaluate has
                # ``awaitPromise: True`` so the result is still awaited.
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
                if image_data_url:
                    print(f"[CAPTCHA] Strategy 1 success: fetched image via JS, length={len(image_data_url)}")
                else:
                    print("[CAPTCHA] Strategy 1: JS fetch returned null.")
            except Exception as e:
                print(f"[CAPTCHA] Strategy 1 failed: {e}")

    # Strategy 2: If captcha is a <canvas>, extract as data URL
    if not image_data_url and captcha_tag == "canvas":
        print("[CAPTCHA] Strategy 2: captcha is <canvas>. Attempting toDataURL...")
        try:
            canvas_script = f"""
            (...args) => {{
                var canvas = document.getElementById('{captcha_id}') || document.querySelector('canvas');
                if (canvas) return canvas.toDataURL('image/png');
                return null;
            }}
            """
            image_data_url = await page.evaluate(canvas_script)
            if image_data_url:
                print(f"[CAPTCHA] Strategy 2 success: canvas.toDataURL returned data, length={len(image_data_url)}")
            else:
                print("[CAPTCHA] Strategy 2: canvas.toDataURL returned null.")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 2 failed: {e}")

    # Strategy 3: Screenshot the CAPTCHA element
    if not image_data_url:
        print("[CAPTCHA] Strategy 3: Trying CDP element screenshot...")
        try:
            # Try to locate the element and take a bounding-box screenshot via CDP
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
            bbox = _parse_evaluate_result(bbox_raw) if bbox_raw else None
            print(f"[CAPTCHA] Strategy 3: element bbox={bbox}")
            # Bail early on zero-size bbox (image may not have loaded yet).
            if bbox and bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0:
                # Use browser session's CDP client to capture screenshot.
                # IMPORTANT: must pass session_id=... or CDP replies with
                # "'Page.captureScreenshot' wasn't found" because the call
                # isn't routed to any target.
                bs = page._browser_session
                print(f"[CAPTCHA] Strategy 3: browser_session={bs}, has cdp_client={hasattr(bs, 'cdp_client') if bs else False}")
                if bs and hasattr(bs, 'cdp_client'):
                    try:
                        session_id = await page.session_id
                    except Exception as e:
                        print(f"[CAPTCHA] Strategy 3: could not obtain session_id: {e}")
                        session_id = None
                    if session_id:
                        clip = {
                            "x": bbox["x"],
                            "y": bbox["y"],
                            "width": bbox["width"],
                            "height": bbox["height"],
                            "scale": 1,
                        }
                        result = await bs.cdp_client.send.Page.captureScreenshot(
                            {"format": "png", "clip": clip},
                            session_id=session_id,
                        )
                        data = result.get("data", "")
                        if data:
                            image_data_url = f"data:image/png;base64,{data}"
                            print(f"[CAPTCHA] Strategy 3 success: CDP screenshot captured, length={len(image_data_url)}")
                        else:
                            print("[CAPTCHA] Strategy 3: CDP screenshot returned empty data.")
                    else:
                        print("[CAPTCHA] Strategy 3: no session_id; skipping CDP screenshot.")
                else:
                    print("[CAPTCHA] Strategy 3: No CDP client available on browser_session.")
            elif bbox:
                print(f"[CAPTCHA] Strategy 3: bbox has zero size ({bbox}); skipping screenshot.")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 3 failed: {e}")

    # Strategy 4: Full-page screenshot fallback (crop to login-form area)
    if not image_data_url:
        print("[CAPTCHA] Strategy 4: Trying full-page screenshot and crop to login form area...")
        try:
            bs = page._browser_session
            if bs and hasattr(bs, 'cdp_client'):
                try:
                    session_id = await page.session_id
                except Exception as e:
                    print(f"[CAPTCHA] Strategy 4: could not obtain session_id: {e}")
                    session_id = None
                if not session_id:
                    raise RuntimeError("no session_id for Page.captureScreenshot")
                result = await bs.cdp_client.send.Page.captureScreenshot(
                    {"format": "png"},
                    session_id=session_id,
                )
                data = result.get("data", "")
                if data:
                    # Try to find the login form bbox to crop
                    form_bbox_script = """
                    (...args) => {
                        var pwd = document.querySelector('input[type="password"]');
                        if (!pwd) return null;
                        var form = pwd.closest('form, div, section');
                        if (!form) return null;
                        var rect = form.getBoundingClientRect();
                        return {
                            x: Math.max(0, Math.round(rect.left)),
                            y: Math.max(0, Math.round(rect.top)),
                            width: Math.round(rect.width),
                            height: Math.round(rect.height)
                        };
                    }
                    """
                    form_bbox_raw = await page.evaluate(form_bbox_script)
                    form_bbox = _parse_evaluate_result(form_bbox_raw) if form_bbox_raw else None
                    if form_bbox:
                        print(f"[CAPTCHA] Strategy 4: Full page captured, cropping to form bbox={form_bbox}")
                        # We have full page base64; crop it using PIL
                        try:
                            from PIL import Image
                            import io, base64
                            full_img = Image.open(io.BytesIO(base64.b64decode(data)))
                            cropped = full_img.crop((
                                form_bbox["x"],
                                form_bbox["y"],
                                form_bbox["x"] + form_bbox["width"],
                                form_bbox["y"] + form_bbox["height"]
                            ))
                            buf = io.BytesIO()
                            cropped.save(buf, format="PNG")
                            image_data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                            print(f"[CAPTCHA] Strategy 4 success: Cropped login form screenshot, length={len(image_data_url)}")
                        except Exception as crop_err:
                            print(f"[CAPTCHA] Strategy 4: Crop failed ({crop_err}), using full page.")
                            image_data_url = f"data:image/png;base64,{data}"
                    else:
                        print("[CAPTCHA] Strategy 4: No form bbox found, using full page screenshot.")
                        image_data_url = f"data:image/png;base64,{data}"
                else:
                    print("[CAPTCHA] Strategy 4: Full page screenshot returned empty data.")
            else:
                print("[CAPTCHA] Strategy 4: No CDP client available.")
        except Exception as e:
            print(f"[CAPTCHA] Strategy 4 failed: {e}")

    if not image_data_url:
        print("[CAPTCHA] All strategies failed. Could not extract CAPTCHA image from page.")
        return ""

    print(f"[CAPTCHA] Sending image to recognizer. data_url length={len(image_data_url)}")
    code = await _recognize_captcha_with_fallback(image_data_url, llm)
    if code:
        print(f"[CAPTCHA] Recognized code: '{code}'")
    else:
        print("[CAPTCHA] Recognizer returned empty code.")
    return code


def _map_llm_zone_type(zone_type: str) -> "ZoneType | None":
    """Map LLM-returned zone type string to ZoneType enum.

    LLM may return: form, table, nav, action-bar, filter-panel, pagination,
    chart, modal, card, tabs, steps, list, content.
    ZoneType enum: search_form, data_table, detail_form, action_bar,
    tab_panel, tree_panel, modal.
    """
    from graph_agent.models import ZoneType

    mapping: dict[str, "ZoneType"] = {
        "form": ZoneType.DETAIL_FORM,
        "table": ZoneType.DATA_TABLE,
        "nav": ZoneType.TREE_PANEL,
        "action-bar": ZoneType.ACTION_BAR,
        "action_bar": ZoneType.ACTION_BAR,
        "filter-panel": ZoneType.SEARCH_FORM,
        "filter_panel": ZoneType.SEARCH_FORM,
        "modal": ZoneType.MODAL,
        "tabs": ZoneType.TAB_PANEL,
        "tab_panel": ZoneType.TAB_PANEL,
        "pagination": ZoneType.DATA_TABLE,
        "card": ZoneType.DETAIL_FORM,
        "list": ZoneType.DATA_TABLE,
        "chart": ZoneType.DETAIL_FORM,
        "steps": ZoneType.TAB_PANEL,
        "content": ZoneType.DETAIL_FORM,
    }
    return mapping.get(zone_type.lower().strip())


async def _ensure_browser_ready(browser: Any, target_url: str) -> bool:
    """Check if browser session is alive; restart if needed."""
    try:
        current = await browser.get_current_page_url()
        if current:
            return True
    except Exception:
        pass
    try:
        print("[ORCH] Browser session appears reset, attempting restart...")
        await browser.start()
        await browser.navigate_to(target_url)
        await asyncio.sleep(2)
        return True
    except Exception as e:
        print(f"[ORCH] Browser restart failed: {e}")
        return False


def _is_login_url(url: str) -> bool:
    """Heuristic: is the URL on a login / sign-in page?"""
    if not url:
        return False
    return bool(re.search(r"login|signin|sign-in|auth", url, re.IGNORECASE))


async def _try_auto_login_orchestrated(browser: Any, llm: Any) -> bool:
    """Re-use runner.py's auto-login flow inside the orchestrated loop.

    Differs from the pre-login version in run_mapping() only in that it is
    callable at any point when the browser lands on a login page. Uses
    Playwright's native fill/click so typing is visible in non-headless
    mode and Vue/React controlled inputs are respected.
    """
    username = (os.getenv("MAPPING_USERNAME") or "").strip()
    password = (os.getenv("MAPPING_PASSWORD") or "").strip()
    if not username or not password:
        print("[ORCH-AUTO_LOGIN] No MAPPING_USERNAME/MAPPING_PASSWORD in env, skipping auto-login.")
        return False

    page = None
    try:
        page = await browser.get_current_page()
    except Exception as e:
        print(f"[ORCH-AUTO_LOGIN] get_current_page() failed: {e}")
        return False

    if not page:
        print("[ORCH-AUTO_LOGIN] No active page on the browser; cannot auto-login.")
        return False

    # Give SPA/redirect a brief moment to settle so execution context isn't
    # stale (browser-use's Page has no wait_for_load_state; use a short sleep).
    await asyncio.sleep(0.5)

    # Detect login form using the same script as run_mapping pre-login.
    # NOTE: browser-use's page.evaluate() requires arrow-function format
    # ``(...args) => { ... }`` — an IIFE ``(function(){...})()`` is rejected
    # with "JavaScript code must start with (...args) => format".
    login_detect_script = r"""
    (...args) => {
        var pwd = document.querySelector('input[type="password"]');
        if (!pwd) return {hasLogin: false};
        var user = document.querySelector('input[type="text"], input:not([type])');
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
        return {
            hasLogin: true,
            hasCaptcha: !!captchaImg,
            captchaTag: captchaImg ? captchaImg.tagName.toLowerCase() : '',
            captchaSrc: captchaImg ? (captchaImg.src || '') : '',
            captchaId: captchaImg ? (captchaImg.id || '') : '',
        };
    }
    """
    try:
        raw = await page.evaluate(login_detect_script)
    except Exception as e:
        print(f"[ORCH-AUTO_LOGIN] login_detect_script failed: {e}")
        return False

    login_info = _parse_evaluate_result(raw)
    if not login_info:
        print(f"[ORCH-AUTO_LOGIN] login_detect_script returned empty result (raw={raw!r})")
        return False
    if not login_info.get("hasLogin"):
        cur_url = await browser.get_current_page_url() or ""
        print(f"[ORCH-AUTO_LOGIN] No login form detected on {cur_url}")
        return False

    print(f"[ORCH-AUTO_LOGIN] Detected login form. Filling credentials for {username}...")

    # CAPTCHA solving
    captcha_code = ""
    if login_info.get("hasCaptcha"):
        print("[ORCH-AUTO_LOGIN] CAPTCHA detected, attempting LLM solve...")
        captcha_code = await _solve_captcha_with_llm(page, login_info, llm)
        if captcha_code:
            print(f"[ORCH-AUTO_LOGIN] CAPTCHA solved: {captcha_code}")
        else:
            print("[ORCH-AUTO_LOGIN] CAPTCHA solve failed, continuing anyway...")

    fill_result = await _fill_login_form_via_evaluate(
        page,
        username=username,
        password=password,
        captcha_code=captcha_code,
    )
    if not fill_result.get("success"):
        print(f"[ORCH-AUTO_LOGIN] Fill script reported failure: {fill_result}")
        return False
    print(
        f"[ORCH-AUTO_LOGIN] Fill result: user={fill_result.get('userFilled')}, "
        f"pwd={fill_result.get('pwdFilled')}, captcha={fill_result.get('captchaFilled')}, "
        f"submit={fill_result.get('submitClicked')}"
    )

    print("[ORCH-AUTO_LOGIN] Credentials submitted. Waiting for redirect...")
    await asyncio.sleep(5)

    post_url = await browser.get_current_page_url()
    if post_url and not _is_login_url(post_url):
        print(f"[ORCH-AUTO_LOGIN] Login successful. Now at: {post_url}")
        return True
    print(f"[ORCH-AUTO_LOGIN] URL after submit: {post_url}. May still be on login page.")
    return False


def _parse_evaluate_result(raw: Any) -> dict[str, Any]:
    """Parse browser-use ``page.evaluate()`` return value into a dict.

    browser-use's ``Page.evaluate()`` returns a **string** (dicts/lists are
    ``json.dumps``'d server-side). Accept either an already-parsed dict or
    a JSON string and return ``{}`` on failure so callers can branch safely.
    """
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


async def _click_menu_by_text(browser: Any, text: str) -> bool:
    """Locate and click a visible navigation menu element whose text matches.

    Used by the orchestrator when a menu item lacks a navigable href
    (SPA-style click-handler menus). Tries common menu selectors first,
    then falls back to any clickable element whose trimmed text matches.
    Returns True if the click was dispatched, False otherwise.
    """
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
        parsed = _parse_evaluate_result(raw)
        if parsed.get("ok"):
            print(
                f"[ORCH] Menu '{target}' clicked "
                f"(via {parsed.get('selector', '?')}, tag={parsed.get('tag', '?')})"
            )
            return True
        print(f"[ORCH] Menu '{target}' not found on current page")
        return False
    except Exception as e:
        print(f"[ORCH] _click_menu_by_text error: {e}")
        return False


async def _fill_login_form_via_evaluate(
    page: Any,
    *,
    username: str,
    password: str,
    captcha_code: str = "",
) -> dict[str, Any]:
    """Fill username/password (and captcha if provided), then click submit.

    Implemented via a single ``page.evaluate()`` arrow function that takes
    the credentials as args (avoiding f-string quoting issues). Uses the
    React/Vue-compatible native value setter so controlled inputs update.
    Returns a dict with booleans for each field filled and submitClicked.
    """
    fill_script = r"""
    (...args) => {
        const [username, password, captchaCode] = args;
        const out = {
            success: false,
            userFilled: false,
            pwdFilled: false,
            captchaFilled: false,
            submitClicked: false,
            reason: "",
        };

        const setValue = (el, val) => {
            try { el.focus(); } catch (e) {}
            const proto = Object.getPrototypeOf(el);
            const desc = proto && Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) {
                desc.set.call(el, val);
            } else {
                el.value = val;
            }
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
        if (!submitBtn) {
            submitBtn = document.querySelector('button[type="submit"], input[type="submit"]');
        }
        if (submitBtn) {
            submitBtn.click();
            out.submitClicked = true;
        } else {
            try {
                const form = pwd.closest('form');
                if (form) {
                    form.requestSubmit ? form.requestSubmit() : form.submit();
                    out.submitClicked = true;
                }
            } catch (e) {
                out.reason = "submit failed: " + (e && e.message ? e.message : e);
            }
        }
        out.success = out.pwdFilled;
        return out;
    }
    """
    try:
        raw = await page.evaluate(fill_script, username, password, captcha_code or "")
    except Exception as e:
        return {"success": False, "reason": f"evaluate failed: {e}"}
    parsed = _parse_evaluate_result(raw)
    if not parsed:
        return {"success": False, "reason": f"empty evaluate result (raw={raw!r})"}
    return parsed


async def _run_orchestrated_mapping(
    browser: Any,
    llm: Any,
    start_url: str,
    current_url: str,
    app_id: str,
    session_id: str,
    max_steps: int,
    time_budget_ms: int = 600_000,
    warm_start_candidates: list[dict[str, Any]] | None = None,
) -> "CartographyResult":
    """Run LLM-first orchestrated exploration across multiple pages.

    Replaces the hard-coded MenuExtractor + ZoneDiscoverer +
    CoverageAnalyzer + ExplorationScheduler pipeline with
    LLM-driven page analysis, planning, and ReActExplorer execution.
    """
    from graph_agent.cartography.react_explorer import ReActExplorer
    from graph_agent.cartography.snapshot import capture_dom_fingerprint
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import State, Transition, Zone, ActionType
    from urllib.parse import urljoin

    print("[ORCH] === LLM-first orchestrated exploration starting ===")

    all_states: list[State] = []
    all_transitions: list[Transition] = []
    all_zones: list[Zone] = []
    all_history: list[dict[str, Any]] = []
    explored_urls: list[str] = []
    menu_items_discovered: list[dict[str, Any]] = []
    zones_discovered: list[dict[str, Any]] = []

    # FIFO queue of (url, reason) tuples to explore, ordered by the
    # sequence in which menus / tasks are discovered (DOM order of the
    # navigation bar for menu items; LLM task order otherwise). This
    # preserves the visual left-to-right / top-to-bottom menu order a
    # human QA tester would naturally follow.
    pages_to_explore: list[tuple[str, str]] = [
        (current_url or start_url, "start page")
    ]
    pages_explored: set[str] = set()

    # Menus that cannot be enqueued as URLs (SPA-style menus with no href).
    # Each entry: {"text": ..., "source_url": <page whose menu bar contains it>}.
    # When `pages_to_explore` is empty we pop from here, navigate back to
    # source_url and dispatch a click-by-text to enter the feature.
    pending_menus: list[dict[str, str]] = []
    # URLs whose menu bar has already been harvested, so re-visiting the
    # dashboard does not re-enqueue the same menus ad infinitum.
    menu_scanned_urls: set[str] = set()
    # Pending menu entries that have already been clicked (source + text),
    # to avoid retrying the same menu in a loop.
    menus_clicked: set[tuple[str, str]] = set()

    # Primary origin: every URL we enqueue or navigate to must share this
    # scheme+host, otherwise we treat it as a foreign page (cross-domain
    # redirect, SSO tab, external link) and refuse to explore it.
    primary_origin_url: str = (start_url or current_url or "").strip()

    def _enqueue_page(url: str, reason: str) -> None:
        clean = _clean_url(url)
        if clean in pages_explored:
            return
        if clean in {_clean_url(u) for u, _ in pages_to_explore}:
            return
        if primary_origin_url and not _same_origin(url, primary_origin_url):
            print(f"[ORCH] Skip foreign-origin URL: {url[:80]}")
            return
        pages_to_explore.append((url, reason))

    async def _cleanup_foreign_tabs() -> str:
        """Close every open tab that is NOT on primary origin.

        Returns the URL of a surviving same-origin tab (best guess of where
        to resume), or '' if none remains.
        """
        if not primary_origin_url:
            return ""
        try:
            tabs = await browser.get_tabs()
        except Exception as e:
            print(f"[ORCH] get_tabs failed during cleanup: {e}")
            return ""
        surviving = ""
        for t in tabs:
            tab_url = getattr(t, "url", "") or ""
            if _same_origin(tab_url, primary_origin_url):
                surviving = surviving or tab_url
                continue
            # Foreign (cross-origin / about:blank / chrome://): close it.
            target_id = getattr(t, "target_id", None)
            if not target_id:
                continue
            try:
                await browser.close_page(target_id)
                print(
                    f"[ORCH] Closed foreign tab ({tab_url[:60] or 'blank'})"
                )
            except Exception as e:
                print(f"[ORCH] Failed to close foreign tab {target_id}: {e}")
        return surviving

    warm_candidates = rank_warm_start_candidates(warm_start_candidates or [])
    warm_targets = [
        (item.get("target_url") or "").strip()
        for item in warm_candidates
        if (item.get("target_url") or "").strip()
    ]
    for target_url in warm_targets:
        clean = _clean_url(target_url)
        if clean == _clean_url(current_url or start_url):
            continue
        if primary_origin_url and not _same_origin(target_url, primary_origin_url):
            continue
        pages_to_explore.append((target_url, "warm-start"))

    orchestration_step = 0
    max_orchestration_steps = 50
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    first_page = True

    while (
        (pages_to_explore or pending_menus)
        and orchestration_step < max_orchestration_steps
    ):
        if _shutdown_requested:
            print("[ORCH] Shutdown requested, stopping exploration.")
            break

        pending_menu_task: dict[str, str] | None = None
        if pages_to_explore:
            url, reason = pages_to_explore.pop(0)
            url_clean = _clean_url(url)
            if url_clean in pages_explored:
                continue
        else:
            # URL queue drained but we still have SPA menus to click on a
            # previously visited portal page. Pop the next menu, navigate
            # back to its source page, then let the click happen below.
            pending_menu_task = pending_menus.pop(0)
            key = (pending_menu_task["source_url"], pending_menu_task["text"])
            if key in menus_clicked:
                continue
            menus_clicked.add(key)
            url = pending_menu_task["source_url"]
            reason = f"menu-click: {pending_menu_task['text']}"
            url_clean = _clean_url(url)

        orchestration_step += 1
        elapsed_ms = (loop.time() - start_time) * 1000

        print(
            f"\n[ORCH] Step {orchestration_step}/{max_orchestration_steps}: "
            f"{url[:80]} (reason: {reason}) "
            f"[queue={len(pages_to_explore)}, pending_menus={len(pending_menus)}]"
        )

        # Ensure browser is alive before navigating
        if not await _ensure_browser_ready(browser, url):
            print(f"[ORCH] Browser unrecoverable, skipping {url[:80]}")
            continue

        if first_page and pending_menu_task is None:
            # First iteration: reuse the current browser state from pre-login.
            # Do NOT navigate — the pre-login phase already loaded the page.
            first_page = False
        else:
            # Subsequent pages (including menu-click tasks): navigate to the
            # target URL (for menu-click tasks this is the source page).
            first_page = False
            try:
                await browser.navigate_to(url)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"[ORCH] Navigation failed: {e}")
                continue

        # If this iteration is a pending menu-click task, dispatch the click
        # now that we're back on the source page. The click usually triggers
        # SPA navigation to a feature page; what follows (DOM snapshot, LLM
        # analysis, ReActExplorer) will then operate on the feature page.
        if pending_menu_task is not None:
            clicked = await _click_menu_by_text(
                browser, pending_menu_task["text"]
            )
            if not clicked:
                print(
                    f"[ORCH] Menu click failed for "
                    f"'{pending_menu_task['text']}', skipping"
                )
                continue
            await asyncio.sleep(2)

        # Get page snapshot for LLM analysis
        dom_text = ""
        page_title = ""
        try:
            bs_summary = await browser.get_browser_state_summary(
                include_screenshot=False, include_recent_events=False
            )
            dom_text = bs_summary.dom_state.llm_representation()
            page_title = getattr(bs_summary, "title", "") or ""
        except Exception as e:
            print(f"[ORCH] Failed to get DOM text: {e}")

        current_page_url = await browser.get_current_page_url() or url

        # If still on a login page, try auto-login again using the same
        # script that runner.py uses in the pre-login phase.
        if _is_login_url(current_page_url):
            print("[ORCH] Still on login page, attempting auto-login...")
            await _try_auto_login_orchestrated(browser, llm)
            # Re-check URL after login attempt
            current_page_url = await browser.get_current_page_url() or current_page_url
            # Refresh DOM after login
            try:
                bs_summary = await browser.get_browser_state_summary(
                    include_screenshot=False, include_recent_events=False
                )
                dom_text = bs_summary.dom_state.llm_representation()
                page_title = getattr(bs_summary, "title", "") or ""
            except Exception:
                pass

        # ------------------------------------------------------------------
        # LLM-driven page analysis (replaces MenuExtractor + ZoneDiscoverer)
        # ------------------------------------------------------------------
        page_analysis = await _analyze_page_with_llm(
            llm, dom_text, current_page_url, page_title
        )
        print(
            f"[ORCH] LLM analysis: page_type={page_analysis.page_type}, "
            f"menus={len(page_analysis.menu_items)}, "
            f"zones={len(page_analysis.functional_zones)}"
        )

        # Record discovered menus and zones
        for m in page_analysis.menu_items:
            menu_items_discovered.append(
                {
                    "text": m.text,
                    "href": m.href,
                    "level": m.level,
                    "source_url": current_page_url,
                }
            )
        for z in page_analysis.functional_zones:
            zones_discovered.append(
                {
                    "zone_type": z.zone_type,
                    "selector": z.selector,
                    "description": z.description,
                    "source_url": current_page_url,
                }
            )
            mapped_zone_type = _map_llm_zone_type(z.zone_type)
            if mapped_zone_type is None:
                continue
            zone_id = (
                f"zone:{z.zone_type}:"
                f"{hashlib.md5(z.selector.encode()).hexdigest()[:8]}"
            )
            all_zones.append(
                Zone(
                    id=zone_id,
                    zone_type=mapped_zone_type,
                    root_selector=z.selector,
                    summary=z.description,
                )
            )

        # Queue internal menu items for later exploration in the order the
        # LLM returned them (which mirrors DOM / visual menu-bar order).
        # Menus with an `href` are enqueued as URLs; SPA-style menus without
        # href are stored as `pending_menus` so we can come back and click
        # them by text after draining the URL queue.
        enqueued_menu_count = 0
        pending_menu_added = 0
        already_scanned = _clean_url(current_page_url) in menu_scanned_urls
        if not already_scanned:
            menu_scanned_urls.add(_clean_url(current_page_url))
            for m in page_analysis.menu_items:
                text = (m.text or "").strip()
                href = (m.href or "").strip()
                full_href = (
                    urljoin(current_page_url, href)
                    if href and not href.startswith("http")
                    else href
                )
                if full_href and _is_http_url(full_href):
                    before = len(pages_to_explore)
                    _enqueue_page(full_href, f"menu: {text}")
                    if len(pages_to_explore) > before:
                        enqueued_menu_count += 1
                elif text:
                    key = (current_page_url, text)
                    if key in menus_clicked:
                        continue
                    if any(
                        p["source_url"] == current_page_url and p["text"] == text
                        for p in pending_menus
                    ):
                        continue
                    pending_menus.append(
                        {"text": text, "source_url": current_page_url}
                    )
                    pending_menu_added += 1

        # ------------------------------------------------------------------
        # Short-circuit: on portal / dashboard / welcome pages, the LLM
        # analysis has already given us every menu (either as URL or as a
        # pending click target). Skip the in-page ReActExplorer and jump
        # straight to the first menu so we don't waste budget clicking nav
        # items on the portal before exploring actual features.
        # ------------------------------------------------------------------
        total_menu_targets = enqueued_menu_count + pending_menu_added
        if (
            total_menu_targets > 0
            and page_analysis.page_type in ("dashboard", "welcome", "login")
            and pending_menu_task is None  # this iteration is not itself a menu-click
        ):
            print(
                f"[ORCH] Skip in-page exploration for {page_analysis.page_type}; "
                f"{enqueued_menu_count} url-menu(s), "
                f"{pending_menu_added} click-menu(s) queued — jumping to first menu."
            )
            pages_explored.add(_clean_url(current_page_url))
            continue

        # ------------------------------------------------------------------
        # ReActExplorer deep page exploration (LLM-driven within the page)
        # ------------------------------------------------------------------
        steps_remaining = max_steps - len(all_history)
        time_remaining = time_budget_ms - elapsed_ms
        if steps_remaining <= 0 or time_remaining <= 0:
            print(
                f"[ORCH] Budget exhausted (steps={steps_remaining}, "
                f"time={time_remaining / 1000:.0f}s)"
            )
            break

        # Build explorer task with guardrails for the current page.
        explorer_hint = f"You are exploring the page at {current_page_url}."

        # Inject universal rules + page-type action policy + zone/menu order.
        exploration_guidance = _build_exploration_guidance(page_analysis)
        if exploration_guidance:
            explorer_hint += "\n\n" + exploration_guidance

        # Per-page step budget: dashboards need fewer steps (just sample menus
        # once), detail/form/list pages need more to exercise the feature.
        _page_type_cap = {
            "dashboard": 20,
            "welcome": 15,
            "login": 10,
            "list": 40,
            "detail": 40,
            "form": 40,
            "settings": 40,
        }.get(page_analysis.page_type, 30)
        per_page_steps = min(steps_remaining, _page_type_cap)
        print(
            f"[ORCH] page_type={page_analysis.page_type} "
            f"step_budget={per_page_steps}"
        )

        explorer = ReActExplorer(
            max_steps=per_page_steps,
            browser_session=browser,
            extra_system_prompt=_build_login_hint_from_env() + "\n\n" + explorer_hint,
        )

        try:
            fp = ""
            try:
                page = await browser.get_current_page()
                fp = await capture_dom_fingerprint(page)
            except Exception:
                pass

            explore_result = await explorer.explore_page(
                session=browser,
                state_id=f"state:{current_page_url}",
                page_title=page_title,
            )

            # Merge results (deduplicate states by id)
            seen_state_ids = {s.id for s in all_states}
            for state in explore_result.states:
                if state.id not in seen_state_ids:
                    all_states.append(state)
                    seen_state_ids.add(state.id)
            for transition in explore_result.transitions:
                all_transitions.append(transition)
            for zone in explore_result.zones:
                all_zones.append(zone)
            if explore_result.history:
                all_history.extend(explore_result.history)

            explored_urls.append(current_page_url)
            pages_explored.add(url_clean)

            print(
                f"[ORCH] Page explored: {len(explore_result.states)} states, "
                f"{len(explore_result.transitions)} transitions, "
                f"{len(explore_result.zones)} zones"
            )

            # --------------------------------------------------------------
            # Harvest same-origin URLs discovered during in-page exploration.
            # ReActExplorer records each clicked state with its URL; menus
            # that triggered SPA navigation or page-load navigation will
            # appear here even if the LLM page-analysis could not extract a
            # static href. This is what keeps the orchestrator's queue
            # alive after a menu_first scan.
            # --------------------------------------------------------------
            harvested = 0
            for state in explore_result.states:
                if getattr(state, "is_external", False):
                    continue
                s_url = (getattr(state, "url", "") or "").strip()
                if not _is_http_url(s_url):
                    continue
                if primary_origin_url and not _same_origin(s_url, primary_origin_url):
                    continue
                if _clean_url(s_url) == url_clean:
                    continue  # skip the page we just explored
                before = len(pages_to_explore)
                _enqueue_page(s_url, "in-page discovery")
                if len(pages_to_explore) > before:
                    harvested += 1
            if harvested:
                print(f"[ORCH] Harvested {harvested} same-origin URL(s) from in-page states")

            # --------------------------------------------------------------
            # Tab hygiene: menu clicks often open external sites (SSO,
            # third-party portals, docs) in new tabs. Close every tab that
            # is not on our primary origin so the next iteration starts
            # from a clean same-origin browsing context.
            # --------------------------------------------------------------
            surviving_url = await _cleanup_foreign_tabs()

            # ReActExplorer may leave us on about:blank or a closed/foreign
            # tab's replacement. Recover by navigating back to a safe URL.
            try:
                post_url = await browser.get_current_page_url() or ""
            except Exception:
                post_url = ""
            needs_recovery = (
                not post_url
                or post_url in ("about:blank", "chrome://newtab/")
                or (
                    primary_origin_url
                    and not _same_origin(post_url, primary_origin_url)
                )
            )
            if needs_recovery:
                recover_target = (
                    surviving_url
                    or current_page_url
                    or url
                    or primary_origin_url
                )
                print(
                    f"[ORCH] Recovering from {post_url or 'empty URL'} "
                    f"-> {recover_target[:80]}"
                )
                try:
                    await browser.navigate_to(recover_target)
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"[ORCH] Recovery navigation failed: {e}")

            # Ensure the session is still alive before the next iteration.
            if not await _ensure_browser_ready(browser, url):
                print("[ORCH] Browser unrecoverable after ReActExplorer, stopping.")
                break

        except Exception as e:
            print(f"[ORCH] ReActExplorer failed: {e}")
            pages_explored.add(url_clean)
            # Try to recover browser for next iteration
            await _ensure_browser_ready(browser, url)
            continue

        # ------------------------------------------------------------------
        # LLM-driven planning (replaces CoverageAnalyzer + ExplorationScheduler)
        # Run every 3 steps or when queue is empty to save LLM calls.
        # ------------------------------------------------------------------
        if orchestration_step % 3 == 0 or not pages_to_explore:
            recent_transitions = [
                {
                    "action": str(t.action),
                    "selector": t.selector,
                    "from_url": t.from_state_id or "",
                    "to_url": t.to_state_id or "",
                }
                for t in all_transitions[-20:]
            ]
            plan = await _plan_next_exploration_with_llm(
                llm,
                current_page_url,
                page_title,
                page_analysis,
                explored_urls,
                recent_transitions,
                time_remaining,
                steps_remaining,
            )
            print(
                f"[ORCH] LLM plan: strategy={plan.strategy}, "
                f"coverage_estimate={plan.coverage_estimate:.2f}, "
                f"tasks={len(plan.tasks)}"
            )

            if plan.strategy == "stop":
                if pages_to_explore or pending_menus:
                    print(
                        "[ORCH] LLM recommended stopping, but "
                        f"{len(pages_to_explore)} URL(s) and "
                        f"{len(pending_menus)} menu(s) still pending — "
                        "continuing anyway."
                    )
                else:
                    print("[ORCH] LLM recommended stopping and no menus remain.")
                    break

            # Enqueue any new page targets from the LLM plan, preserving
            # the order the LLM produced them in.
            for task in plan.tasks:
                if task.task_type == "explore_page" and task.target_url:
                    _enqueue_page(
                        task.target_url,
                        f"llm-plan: {task.description}",
                    )

    # Loop exit diagnostics so it's obvious WHY exploration stopped.
    if orchestration_step >= max_orchestration_steps:
        print(
            f"[ORCH] Reached max_orchestration_steps={max_orchestration_steps}"
            f" (queue={len(pages_to_explore)}, pending_menus={len(pending_menus)})"
        )
    elif not pages_to_explore and not pending_menus:
        print("[ORCH] URL queue and pending menus both empty — exploration done.")

    # Build final CartographyResult
    result = CartographyResult()
    result.states = all_states
    result.transitions = all_transitions
    result.zones = all_zones
    result.history = all_history
    result.menus = menu_items_discovered
    result.zone_hints = zones_discovered

    print("\n[ORCH] === Orchestrated exploration complete ===")
    print(f"  Pages explored: {len(pages_explored)}")
    print(
        f"  States: {len(all_states)}, "
        f"Transitions: {len(all_transitions)}, "
        f"Zones: {len(all_zones)}"
    )

    return result


async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,  # Kept for API compatibility, ignored
    task: str | None = None,
    max_steps: int = 100,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,  # Kept for API compatibility, ignored
) -> str:
    """Run LLM-first orchestrated exploration and store results in Neo4j.

    - url: Start URL (required via arg or MAPPING_URL env).
    - output_path: Kept for API compatibility, data now stored in Neo4j.
    - task: Kept for API compatibility, no longer used by the orchestrator.
    - max_steps: Maximum agent steps.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the app_id of the newly created app in Neo4j.
    """
    if inventory_path is None or not str(inventory_path).strip():
        raise ValueError(
            "inventory_path is required. Run scout first, then pass --inventory to mapping."
        )
    inventory = _load_inventory(inventory_path)

    from browser_use import Browser

    resolved_url = _resolve_mapping_url(url)
    pkg_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        # Resolve default relative to package: graph_agent/data/graph.json
        output_path = str(pkg_root / "data" / "graph.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    browser_kwargs: dict[str, Any] = {
        "headless": _resolve_mapping_headless(),
        "args": ["--incognito"],  # Force incognito mode: no session cache
    }
    channel = _resolve_mapping_channel()
    if channel:
        browser_kwargs["channel"] = channel
    try:
        browser = Browser(**browser_kwargs)
    except TypeError:
        # Older browser-use versions may not support channel keyword.
        browser_kwargs.pop("channel", None)
        browser = Browser(**browser_kwargs)
    
    async with managed_browser(browser):
        llm = get_llm()
        print("[RUNNER] === Pre-navigation & auto-login phase starting ===")

        # Use graph_agent/data as the base for file operations
        # Note: browser-use will append 'browseruse_agent_data' to this path
        data_dir = pkg_root / "data"

        # ------------------------------------------------------------------
        # Pre-navigation & auto-login: start browser, navigate to target,
        # detect login form, auto-fill credentials, THEN start agent.
        # All steps are recorded in ``initial_actions_log`` so they can be
        # merged into the agent history and stored in Neo4j.
        # ------------------------------------------------------------------
        initial_actions_log: list[dict[str, Any]] = []

        def _log_initial(action: dict[str, Any], thought: str, url: str) -> None:
            """Record a pre-login step for later history injection."""
            initial_actions_log.append({
                "action": action,
                "thought": {"next_goal": thought},
                "url": url,
            })

        print(f"[PreLogin] Starting browser and navigating to {resolved_url}...")
        try:
            await browser.start()
            await browser.navigate_to(resolved_url)
            current_url = await browser.get_current_page_url() or resolved_url
            _log_initial(
                {"navigate": {"url": resolved_url}},
                f"Navigate to {resolved_url}",
                current_url,
            )
            # Give the page time to load / redirect. browser-use's Page is
            # a CDP wrapper (not Playwright) and has no wait_for_load_state;
            # just sleep a few seconds so redirects like /index.html →
            # /login.html have time to settle before we re-acquire the page.
            await asyncio.sleep(4)
            current_url = await browser.get_current_page_url() or current_url
            _log_initial(
                {"wait": {"seconds": 4}},
                "Wait for page to fully load",
                current_url,
            )
        except Exception as e:
            print(f"[WARN] Pre-navigation failed: {e}. Agent will attempt recovery.")

        # Auto-login detection & fill. Re-acquire page AFTER the load wait so
        # we don't evaluate against a stale execution context that was blown
        # away by a redirect.
        page = None
        try:
            page = await browser.get_current_page()
        except Exception as e:
            print(f"[AUTO_LOGIN] get_current_page() failed: {e}")

        if page is None:
            print("[AUTO_LOGIN] No active page after pre-navigation; skipping auto-login.")

        if page:
            # NOTE: browser-use's page.evaluate() requires arrow-function
            # format ``(...args) => { ... }``; IIFE is rejected.
            login_detect_script = """
            (...args) => {
                var pwd = document.querySelector('input[type="password"]');
                if (!pwd) return {hasLogin: false};
                var user = document.querySelector('input[type="text"], input:not([type])');
                var form = pwd.closest('form');
                var submit = form ? form.querySelector('button[type="submit"], input[type="submit"]') : null;
                if (!submit) {
                    submit = pwd.closest('form, div, section')?.querySelector('button[type="submit"], input[type="submit"]');
                }
                // Fallback: any button near the password field
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
                // Detect CAPTCHA image and input
                var captchaImg = null;
                var captchaInput = null;
                var allImgs = document.querySelectorAll('img');
                for (var j = 0; j < allImgs.length; j++) {
                    var imgSrc = allImgs[j].src || '';
                    var imgAlt = allImgs[j].alt || '';
                    var imgId = allImgs[j].id || '';
                    var imgCls = allImgs[j].className || '';
                    var combined = imgSrc + imgAlt + imgId + imgCls;
                    if (/captcha|验证码|verify|auth|code/i.test(combined)) {
                        captchaImg = allImgs[j];
                        break;
                    }
                }
                // Also check for canvas-based CAPTCHA
                if (!captchaImg) {
                    var canvases = document.querySelectorAll('canvas');
                    for (var k = 0; k < canvases.length; k++) {
                        var cId = canvases[k].id || '';
                        var cCls = canvases[k].className || '';
                        if (/captcha|验证码|verify|auth|code/i.test(cId + cCls)) {
                            captchaImg = canvases[k];
                            break;
                        }
                    }
                }
                // Find CAPTCHA input (near the image or with code/verify in name/placeholder)
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
            login_info: dict[str, Any] = {}
            raw_detect: Any = None
            try:
                raw_detect = await page.evaluate(login_detect_script)
            except Exception as e:
                print(f"[AUTO_LOGIN] login_detect_script failed: {e}")
                raw_detect = None

            if raw_detect is None:
                print("[AUTO_LOGIN] login_detect_script returned no result; skipping auto-login.")
            else:
                login_info = _parse_evaluate_result(raw_detect)
            if not login_info:
                # Already logged above; continue to skip branch.
                if raw_detect is not None:
                    print(f"[AUTO_LOGIN] Could not parse detect result (raw={raw_detect!r}); skipping auto-login.")
            elif not login_info.get("hasLogin"):
                cur_url = await browser.get_current_page_url() or current_url
                print(f"[AUTO_LOGIN] No login form detected on {cur_url}; skipping auto-login.")
            else:
                username = (os.getenv("MAPPING_USERNAME") or "").strip()
                password = (os.getenv("MAPPING_PASSWORD") or "").strip()
                if not (username and password):
                    print("[AUTO_LOGIN] Login form detected but no MAPPING_USERNAME/MAPPING_PASSWORD in env.")
                else:
                    print(f"[AUTO_LOGIN] Detected login form. Filling credentials for {username}...")
                    current_url = await browser.get_current_page_url() or current_url
                    _log_initial(
                        {"detect_login": {"has_captcha": login_info.get("hasCaptcha", False)}},
                        "Detected login form on the page",
                        current_url,
                    )

                    # --- CAPTCHA solving ---
                    captcha_code = ""
                    if login_info.get("hasCaptcha"):
                        print("[AUTO_LOGIN] CAPTCHA image detected. Attempting to solve...")
                        captcha_code = await _solve_captcha_with_llm(page, login_info, llm)
                        if captcha_code:
                            print(f"[AUTO_LOGIN] CAPTCHA solved: {captcha_code}")
                            _log_initial(
                                {"solve_captcha": {"code": captcha_code}},
                                f"Solved CAPTCHA: {captcha_code}",
                                current_url,
                            )
                        else:
                            print("[AUTO_LOGIN] Failed to solve CAPTCHA. Login may fail.")
                            _log_initial(
                                {"solve_captcha": {"code": "", "status": "failed"}},
                                "Failed to solve CAPTCHA",
                                current_url,
                            )

                    # Fill + submit in one shot via page.evaluate (arrow
                    # function). If the execution context was destroyed by
                    # a redirect, re-acquire the page once and retry.
                    fill_result = await _fill_login_form_via_evaluate(
                        page,
                        username=username,
                        password=password,
                        captcha_code=captcha_code,
                    )
                    if not fill_result.get("success"):
                        print(f"[AUTO_LOGIN] Fill failed once ({fill_result}); re-acquiring page and retrying...")
                        try:
                            page = await browser.get_current_page()
                        except Exception as e2:
                            print(f"[AUTO_LOGIN] Could not re-acquire page: {e2}")
                            page = None
                        if page is not None:
                            fill_result = await _fill_login_form_via_evaluate(
                                page,
                                username=username,
                                password=password,
                                captcha_code=captcha_code,
                            )

                    if fill_result.get("success") and fill_result.get("submitClicked"):
                        print(
                            f"[AUTO_LOGIN] Fill result: user={fill_result.get('userFilled')}, "
                            f"pwd={fill_result.get('pwdFilled')}, captcha={fill_result.get('captchaFilled')}, "
                            f"submit={fill_result.get('submitClicked')}"
                        )
                        _log_initial(
                            {
                                "input_text": {
                                    "username": username,
                                    "password": "***",
                                    "captcha": captcha_code or "",
                                },
                                "click": {"target": "submit"},
                            },
                            "Filled login credentials and clicked submit",
                            current_url,
                        )
                        print("[AUTO_LOGIN] Credentials submitted. Waiting for redirect...")
                        await asyncio.sleep(5)
                        current_url = await browser.get_current_page_url() or current_url
                        _log_initial(
                            {"wait": {"seconds": 5}},
                            "Wait for login redirect to complete",
                            current_url,
                        )
                        current_url = await browser.get_current_page_url() or current_url
                        if current_url and not re.search(r'login|signin|sign-in', current_url, re.IGNORECASE):
                            print(f"[AUTO_LOGIN] Login successful. Now at: {current_url}")
                        else:
                            print(f"[AUTO_LOGIN] URL after submit: {current_url}. May still be on login page.")
                    else:
                        print(
                            f"[AUTO_LOGIN] Auto-login skipped (fill_result={fill_result}). "
                            "Mapping will continue; agent may attempt login manually."
                        )

        # Get current URL after pre-navigation / auto-login
        current_url = ""
        try:
            current_url = await browser.get_current_page_url()
        except Exception:
            pass

        print(f"resolved_url {resolved_url}")
        if current_url and current_url.startswith(resolved_url):
            print(f"[PreLogin] Browser is now at {current_url}.")
        else:
            print(f"[WARN] Could not confirm current URL ({current_url}).")

        # Generate app_id and session_id early (needed by the orchestrator and Neo4j)
        from urllib.parse import urlparse
        parsed = urlparse(resolved_url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        app_name = domain or "unknown"
        app_id = f"app:{app_name}:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        session_id = f"session:{app_id}:{datetime.now(timezone.utc).isoformat()}"
        warm_start_candidates: list[dict[str, Any]] = []
        try:
            from graph_agent.neo4j_client.manager import GraphManager

            async with GraphManager() as warm_manager:
                warm_start_candidates = await warm_manager._run_read(
                    """
                    MATCH (a:App {name: $app_name})
                    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                    OPTIONAL MATCH (target)-[:HAS_ZONE]->(z:Zone)
                    RETURN t.id AS transition_id,
                           coalesce(t.confidence, 0.0) AS confidence,
                           target.url AS target_url,
                           count(z) = 0 AS zone_unexplored
                    ORDER BY confidence DESC
                    LIMIT 200
                    """,
                    app_name=app_name,
                )
        except Exception as e:
            print(f"[ORCH] Warm-start candidate query skipped: {e}")

        # Check for shutdown request before running
        if _shutdown_requested:
            print("[INFO] Shutdown requested before agent run, cleaning up...")
            return ""

        print("[RUNNER] === LLM-first orchestrated mode ===")
        try:
            result = await _run_orchestrated_mapping(
                browser=browser,
                llm=llm,
                start_url=resolved_url,
                current_url=current_url or resolved_url,
                app_id=app_id,
                session_id=session_id,
                max_steps=max_steps,
                warm_start_candidates=warm_start_candidates,
            )
        except asyncio.CancelledError:
            print("[INFO] Orchestrated exploration cancelled, cleaning up...")
            raise

    # ------------------------------------------------------------------
    # Write CartographyResult directly to Neo4j
    # ------------------------------------------------------------------
    from graph_agent.neo4j_client.manager import GraphManager
    from graph_agent.models import (
        App,
        Session,
        State,
        Transition,
        ActionType,
        Evidence,
        EvidenceType,
    )

    async with GraphManager() as manager:
        # Create app and session
        app = App(id=app_id, name=app_name, base_url=resolved_url)
        await manager.add_app(app)

        session = Session(
            id=session_id,
            app_id=app_id,
            start_url=resolved_url,
        )
        await manager.add_session(session)

        # Link inventory
        if inventory:
            await manager.set_session_inventory(session_id, inventory)

        # Stats tracking
        stats = {
            "states_added": 0,
            "transitions_added": 0,
            "filtered_non_ui_edges": 0,
            "semantic_mismatch_warnings": 0,
            "url_discontinuity_warnings": 0,
            "frame_context_transition_warnings": 0,
        }

        # Optionally record pre-login transition if we have before/after URLs
        login_url = resolved_url
        post_login_url = current_url or resolved_url
        if login_url != post_login_url and initial_actions_log:
            from hashlib import md5
            login_fp = md5(login_url.encode()).hexdigest()[:12]
            post_fp = md5(post_login_url.encode()).hexdigest()[:12]
            from_state = State(
                id=f"state:prelogin:{login_fp}",
                url=login_url,
                title="Login page",
            )
            to_state = State(
                id=f"state:prelogin:{post_fp}",
                url=post_login_url,
                title="Post-login page",
            )
            prelogin_transition = Transition(
                id=f"{session_id}:prelogin",
                selector="[pre-login]",
                action=ActionType.CLICK,
                from_state_id=from_state.id,
                to_state_id=to_state.id,
                thought="Auto-login: filled credentials and submitted",
                confidence=0.9,
            )
            await manager.add_state(from_state)
            await manager.link_app_state(app_id, from_state.id)
            await manager.link_session_discovered(session_id, from_state.id)
            stats["states_added"] += 1

            await manager.add_state(to_state)
            await manager.link_app_state(app_id, to_state.id)
            await manager.link_session_discovered(session_id, to_state.id)
            stats["states_added"] += 1

            await manager.add_transition(prelogin_transition)
            await manager.link_session_transition(session_id, prelogin_transition.id)
            stats["transitions_added"] += 1

        # Write CartographyResult states (deduplicate by ID)
        seen_state_ids: set[str] = set()
        for state in result.states:
            if state.id in seen_state_ids:
                continue
            await manager.add_state(state)
            await manager.link_app_state(app_id, state.id)
            await manager.link_session_discovered(session_id, state.id)
            seen_state_ids.add(state.id)
            stats["states_added"] += 1

        # Write transitions
        for transition in result.transitions:
            await manager.add_transition(transition)
            await manager.link_session_transition(session_id, transition.id)
            stats["transitions_added"] += 1
            # Log transition for visibility
            print(
                f"  [{transition.step_index or 0}] url={transition.from_state_id!r} "
                f"selector={transition.selector!r} action={transition.action!r} "
                f"intent={transition.intent.key if transition.intent else '<none>'!r}"
            )
            for evidence_id in transition.evidence_ids:
                evidence_type = (
                    EvidenceType.URL_CHANGE
                    if evidence_id.endswith("url_change")
                    else EvidenceType.DOM_DIFF
                )
                evidence = Evidence(
                    id=evidence_id,
                    transition_id=transition.id,
                    session_id=session_id,
                    evidence_type=evidence_type,
                    summary=f"{evidence_type.value} evidence for {transition.selector}",
                    payload=json.dumps(
                        {
                            "from_state_id": transition.from_state_id,
                            "to_state_id": transition.to_state_id,
                            "selector": transition.selector,
                            "step_index": transition.step_index,
                        },
                        ensure_ascii=False,
                    ),
                    confidence=min(1.0, max(0.1, transition.confidence)),
                )
                await manager.add_evidence(evidence)
                await manager.link_transition_evidence(transition.id, evidence.id)
                await manager.link_session_evidence(session_id, evidence.id)

        # Persist discovered menu/zones to improve replay metadata reuse
        menu_rows: list[dict[str, Any]] = []
        if getattr(result, "menus", None):
            for i, menu in enumerate(result.menus):
                text = (menu.get("text") or "").strip()
                href = (menu.get("href") or "").strip()
                level = int(menu.get("level") or 0)
                source_url = (menu.get("source_url") or "").strip()
                if not text and not href:
                    continue
                menu_id_src = f"{app_id}|{source_url}|{level}|{text}|{href}"
                menu_rows.append(
                    {
                        "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
                        "text": text or href,
                        "href": href,
                        "level": level,
                        "order": i,
                        "is_active": False,
                    }
                )
            if menu_rows:
                await manager.add_menus(
                    app_id=app_id,
                    menus=menu_rows,
                    page_url=current_url or resolved_url,
                    session_id=session_id,
                )

        if getattr(result, "zone_hints", None):
            zone_rows: list[dict[str, Any]] = []
            for zone in result.zone_hints:
                selector = (zone.get("selector") or "").strip()
                z_type = (zone.get("zone_type") or "content").strip()
                summary = (zone.get("description") or "").strip()
                if not selector:
                    continue
                zid_src = f"{app_id}|{z_type}|{selector}"
                zone_rows.append(
                    {
                        "id": f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}",
                        "type": z_type,
                        "selector": selector,
                        "element_count": 0,
                        "bounds": "",
                        "text_sample": summary,
                    }
                )
            if zone_rows:
                await manager.add_zones(
                    app_id=app_id,
                    zones=zone_rows,
                )

        if menu_rows:
            for transition in result.transitions:
                signal = f"{transition.selector} {transition.thought or ''}".lower()
                for menu in menu_rows:
                    text = str(menu.get("text") or "").strip().lower()
                    if text and text in signal:
                        await manager.link_transition_navigated_via(
                            transition.id, str(menu["id"])
                        )
                        break

        # Collect visited URLs from result history
        visited_urls: list[str] = []
        if result.history:
            for h in result.history:
                url = h.get("url", "")
                if url and _is_http_url(url):
                    visited_urls.append(url)
        visited_urls = list(dict.fromkeys(visited_urls))

        # Parse stop reason from final history entry
        final_text = ""
        if result.history:
            last_entry = result.history[-1]
            final_text = last_entry.get("result", "")

        mapping_stopped = False
        stop_reason = None
        for marker in ("Stopped:", "停止："):
            if marker in final_text:
                idx = final_text.find(marker)
                stop_reason = final_text[idx + len(marker) :].strip()
                if len(stop_reason) > 200:
                    stop_reason = stop_reason[:200] + "..."
                mapping_stopped = True
                break

        # Update session stats
        await manager.update_session_stats(
            session_id=session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": resolved_url,
                **stats,
            }
        )

        # Print summary
        print(
            f"Graph stored in Neo4j: app_id={app_id} "
            f"(states={stats.get('states_added', 0)}, transitions={stats.get('transitions_added', 0)})"
        )

        return app_id


def main() -> None:
    """CLI entry: run scout then mapping. Scout writes inventory, mapping uses it to build graph."""
    import argparse
    from dotenv import load_dotenv

    # Try loading from current directory first, then fallback to project root
    if not load_dotenv():
        # Fallback: try to find .env in project root
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded .env from {env_path}")
        else:
            print(f"Warning: .env not found at {env_path} or current directory")
    
    # Setup browser-use timeouts from MAPPING_TIMEOUT env
    _setup_browser_use_timeouts()

    pkg_root = Path(__file__).resolve().parent.parent
    default_output = str(pkg_root / "data" / "graph.json")
    default_inventory = str(pkg_root / "data" / "element_inventory.json")

    parser = argparse.ArgumentParser(
        description="Run scout (list page elements) then mapping (explore flow, build graph)."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MAPPING_URL", ""),
        help="Start URL for scout and mapping. If omitted, uses MAPPING_URL.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MAPPING_OUTPUT", default_output),
        help="Output graph JSON path (default: graph_agent/data/graph.json)",
    )
    parser.add_argument(
        "--inventory",
        default=os.getenv("MAPPING_INVENTORY", default_inventory),
        help="Scout inventory JSON path (default: graph_agent/data/element_inventory.json)",
    )
    parser.add_argument(
        "--scout-pages",
        default=os.getenv("SCOUT_PAGES", ""),
        help="Comma-separated extra pages for multi-page scout aggregation. Supports relative paths or absolute URLs.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly mapped graph with existing output file.",
    )
    args = parser.parse_args()
    output = (args.output or "").strip() or default_output
    inventory = (args.inventory or "").strip() or default_inventory
    scout_pages_arg = (args.scout_pages or "").strip()
    scout_pages = [item.strip() for item in scout_pages_arg.split(",") if item.strip()]
    merge_existing = bool(args.merge_existing)
    url = _resolve_mapping_url(args.url)

    async def _run() -> None:
        global _shutdown_requested

        if _shutdown_requested:
            print("[INFO] Shutdown requested, exiting before mapping...")
            return

        # Scout module removed; create empty inventory for backward compat
        import json
        if not Path(inventory).exists():
            Path(inventory).parent.mkdir(parents=True, exist_ok=True)
            Path(inventory).write_text(json.dumps({"elements": []}), encoding="utf-8")

        print("Mapping (explore flow, build graph)...")
        try:
            app_id = await run_mapping(
                url=url,
                output_path=output,
                inventory_path=inventory,
                merge_existing=merge_existing,
            )
        except asyncio.CancelledError:
            print("[INFO] Mapping cancelled")
            raise

    # Use a custom event loop to handle signals and cleanup properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Add signal handlers for the main thread
    def _sigint_handler():
        """Handle Ctrl+C in the main thread."""
        global _shutdown_requested
        _shutdown_requested = True
        print("\n[INFO] SIGINT received, shutting down gracefully...")
        # Cancel all tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
    
    # Use asyncio's signal handling (works on Unix and Windows with ProactorEventLoop)
    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
        loop.add_signal_handler(signal.SIGTERM, _sigint_handler)
    except NotImplementedError:
        # Fallback for Windows with SelectorEventLoop
        signal.signal(signal.SIGINT, lambda s, f: _sigint_handler())
        signal.signal(signal.SIGTERM, lambda s, f: _sigint_handler())
    
    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, initiating shutdown...")
    except asyncio.CancelledError:
        print("\n[INFO] Tasks cancelled, cleaning up...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise
    finally:
        # Clean up browsers
        try:
            loop.run_until_complete(_cleanup_all_browsers())
            print("[INFO] Browser cleanup complete")
        except Exception as cleanup_error:
            print(f"[WARN] Cleanup error: {cleanup_error}")
        
        # Remove signal handlers
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass
        
        loop.close()
        print("[INFO] Shutdown complete")
        sys.exit(130 if _shutdown_requested else 0)


if __name__ == "__main__":
    main()

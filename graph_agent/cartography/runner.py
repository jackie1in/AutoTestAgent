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
from types import FrameType
from typing import TYPE_CHECKING, cast
from pydantic import BaseModel, Field

from browser_use.browser.session import BrowserSession as Browser
from browser_use.llm.base import BaseChatModel
from graph_agent.cartography import browser_lifecycle, config as cartography_config
from graph_agent.cartography.captcha import (
    recognize_captcha_with_fallback,
    solve_captcha_from_page,
)
from graph_agent.cartography.login import (
    click_menu_by_text as login_click_menu_by_text,
    detect_login_info,
    fill_login_form_via_evaluate as login_fill_login_form_via_evaluate,
    is_login_url as login_is_login_url,
    parse_evaluate_result as login_parse_evaluate_result,
    try_auto_login_orchestrated as login_try_auto_login_orchestrated,
)
from graph_agent.cartography.mapping_pipeline import (
    map_llm_zone_type as orch_map_llm_zone_type,
    rank_warm_start_candidates as orch_rank_warm_start_candidates,
    run_orchestrated_mapping,
    ensure_browser_ready as orch_ensure_browser_ready,
)
from graph_agent.cartography.manual_capture import ManualCaptureSession
from graph_agent.cartography.persistence import persist_mapping_result
from graph_agent.cartography.types import (
    EvidenceBundleItem,
    FillResult,
    LLMTransitionHint,
    LoginInfo,
)
from graph_agent.llm import get_llm
from graph_agent.llm.utils import ainvoke_structured
from graph_agent.lib.observability import initialize_laminar, observe
from browser_use.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from graph_agent.lib.captcha_solver import solve_with_ddddocr

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import ZoneType

# Global registry for active browser sessions (for cleanup on Ctrl+C)
_active_browsers: list[Browser] = []
_shutdown_requested = False


def _register_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.register_browser(browser)


def _unregister_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.unregister_browser(browser)


async def _cleanup_all_browsers() -> None:
    """Compatibility wrapper."""
    await browser_lifecycle.cleanup_all_browsers()


def _signal_handler(signum: int, frame: FrameType | None) -> None:
    """Compatibility wrapper."""
    global _shutdown_requested
    _shutdown_requested = True
    browser_lifecycle.request_shutdown()


# Register signal handlers
browser_lifecycle.install_signal_handlers()


@asynccontextmanager
async def managed_browser(browser: Browser):
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
    return cartography_config.clean_url(url)


def _resolve_mapping_url(url: str | None) -> str:
    return cartography_config.resolve_mapping_url(url)


def _resolve_mapping_headless() -> bool:
    return cartography_config.resolve_mapping_headless()


def _resolve_mapping_channel() -> str | None:
    return cartography_config.resolve_mapping_channel()


def _setup_browser_use_timeouts():
    cartography_config.setup_browser_use_timeouts()


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
    llm: BaseChatModel,
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
# on a list). Strategy selection has been removed — the pipeline always
# walks menus FIFO, and ReActExplorer always explores the CURRENT page then
# calls `done`, letting the pipeline proceed to the next queued menu.
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
        "pipeline, NOT by you. Focus only on in-page content: KPI cards, "
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
    "call `done`. The pipeline will automatically pick up the next menu / "
    "page after you finish.\n"
    "  - Do NOT repeatedly click navigation menus to jump between features. "
    "Click a menu item ONCE to record its target, then move on — cross-page "
    "traversal is the pipeline's job, not yours.\n"
    "  - Cross-domain rule: if a click navigates to a different domain (SSO "
    "portal, third-party docs, external site), immediately call `done` so the "
    "pipeline can close that tab. Never use `go_back` to escape — just "
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
    llm: BaseChatModel,
    current_url: str,
    page_title: str,
    page_analysis: LLMPageAnalysis,
    explored_urls: list[str],
    discovered_transitions: list[LLMTransitionHint],
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
    return cartography_config.is_http_url(value)


def _same_origin(a: str, b: str) -> bool:
    return cartography_config.same_origin(a, b)


def _load_inventory(inventory_path: str | Path) -> list[dict]:
    return cartography_config.load_inventory(inventory_path)


def _env_bool(name: str, default: bool) -> bool:
    return cartography_config.env_bool(name, default)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def _recognize_captcha_with_fallback(image_data_url: str, llm: BaseChatModel) -> str:
    # Keep local resolver for backward-compatible monkeypatch path in tests.
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
        messages: list[UserMessage | SystemMessage | AssistantMessage] = [
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
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    return orch_rank_warm_start_candidates(candidates)


async def _solve_captcha_with_llm(
    page: "Page",
    login_info: LoginInfo,
    llm: BaseChatModel,
) -> str:
    return await solve_captcha_from_page(page, login_info, llm)


def _map_llm_zone_type(zone_type: str) -> "ZoneType | None":
    return orch_map_llm_zone_type(zone_type)


async def _ensure_browser_ready(browser: Browser, target_url: str) -> bool:
    return await orch_ensure_browser_ready(browser, target_url)


def _is_login_url(url: str) -> bool:
    return login_is_login_url(url)


async def _try_auto_login_orchestrated(browser: Browser, llm: BaseChatModel) -> bool:
    return await login_try_auto_login_orchestrated(browser, llm)


def _parse_evaluate_result(raw: object) -> dict[str, object]:
    return login_parse_evaluate_result(raw)


async def _click_menu_by_text(browser: Browser, text: str) -> bool:
    return await login_click_menu_by_text(browser, text)


async def _fill_login_form_via_evaluate(
    page: "Page",
    *,
    username: str,
    password: str,
    captcha_code: str = "",
) -> FillResult:
    return await login_fill_login_form_via_evaluate(
        page,
        username=username,
        password=password,
        captcha_code=captcha_code,
    )


async def _run_orchestrated_mapping(
    browser: Browser,
    llm: BaseChatModel,
    start_url: str,
    current_url: str,
    app_id: str,
    session_id: str,
    max_steps: int,
    time_budget_ms: int = 600_000,
    warm_start_candidates: list[dict[str, object]] | None = None,
) -> "CartographyResult":
    return await run_orchestrated_mapping(
        browser=browser,
        llm=llm,
        start_url=start_url,
        current_url=current_url,
        app_id=app_id,
        session_id=session_id,
        max_steps=max_steps,
        time_budget_ms=time_budget_ms,
        warm_start_candidates=warm_start_candidates,
    )


@observe(
    name="cartography.run_mapping",
    metadata={"component": "cartography", "stage": "entry"},
)
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
    - task: Kept for API compatibility, no longer used by the pipeline.
    - max_steps: Maximum agent steps.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the app_id of the newly created app in Neo4j.
    """
    if inventory_path is None or not str(inventory_path).strip():
        raise ValueError(
            "inventory_path is required. Run scout first, then pass --inventory to mapping."
        )
    initialize_laminar()
    inventory = _load_inventory(inventory_path)

    from browser_use import Browser

    resolved_url = _resolve_mapping_url(url)
    pkg_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        # Resolve default relative to package: graph_agent/data/graph.json
        output_path = str(pkg_root / "data" / "graph.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    channel = _resolve_mapping_channel()
    base_headless = _resolve_mapping_headless()
    base_args = ["--incognito"]  # Force incognito mode: no session cache
    if channel:
        try:
            browser = Browser(headless=base_headless, args=base_args, channel=channel)
        except TypeError:
            # Older browser-use versions may not support channel keyword.
            browser = Browser(headless=base_headless, args=base_args)
    else:
        browser = Browser(headless=base_headless, args=base_args)
    
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
        initial_actions_log: list[dict[str, object]] = []

        def _log_initial(action: dict[str, object], thought: str, url: str) -> None:
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
            login_info = await detect_login_info(page)
            if not login_info:
                print("[AUTO_LOGIN] detect_login_info returned empty result; skipping auto-login.")
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

                    captcha_code = ""
                    if login_info.get("hasCaptcha"):
                        print("[AUTO_LOGIN] CAPTCHA image detected. Attempting to solve...")
                        captcha_code = await _solve_captcha_with_llm(page, login_info, llm)
                        if captcha_code:
                            _log_initial(
                                {"solve_captcha": {"code": captcha_code}},
                                f"Solved CAPTCHA: {captcha_code}",
                                current_url,
                            )
                        else:
                            _log_initial(
                                {"solve_captcha": {"code": "", "status": "failed"}},
                                "Failed to solve CAPTCHA",
                                current_url,
                            )

                    fill_result = await _fill_login_form_via_evaluate(
                        page,
                        username=username,
                        password=password,
                        captcha_code=captcha_code,
                    )
                    if not fill_result.get("success"):
                        try:
                            page = await browser.get_current_page()
                        except Exception:
                            page = None
                        if page is not None:
                            fill_result = await _fill_login_form_via_evaluate(
                                page,
                                username=username,
                                password=password,
                                captcha_code=captcha_code,
                            )

                    if fill_result.get("success") and fill_result.get("submitClicked"):
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
                        await asyncio.sleep(5)
                        current_url = await browser.get_current_page_url() or current_url
                        _log_initial(
                            {"wait": {"seconds": 5}},
                            "Wait for login redirect to complete",
                            current_url,
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

        # Generate app_id and session_id early (needed by the pipeline and Neo4j)
        from urllib.parse import urlparse
        parsed = urlparse(resolved_url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        app_name = domain or "unknown"
        app_id = f"app:{app_name}:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        session_id = f"session:{app_id}:{datetime.now(timezone.utc).isoformat()}"
        warm_start_candidates: list[dict[str, object]] = []
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
            print(f"[PIPELINE] Warm-start candidate query skipped: {e}")

        # Check for shutdown request before running
        if browser_lifecycle.is_shutdown_requested():
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

    await persist_mapping_result(
        app_id=app_id,
        app_name=app_name,
        session_id=session_id,
        resolved_url=resolved_url,
        current_url=current_url or resolved_url,
        inventory=inventory,
        initial_actions_log=initial_actions_log,
        result=result,
    )
    return app_id


async def run_manual_mapping(
    *,
    app_name: str,
    session_id: str,
    events: list[dict[str, object]],
    mode: str = "manual_raw",
    app_id: str | None = None,
    operator_id: str = "human:operator",
    start_state_hint: str = "",
    start_url: str = "",
) -> str:
    """Persist manual capture events through the same cartography pipeline."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resolved_app_id = app_id or f"app:{app_name}:{ts}"
    if mode == "manual_graph_assisted":
        manual_session = ManualCaptureSession.from_graph_context(
            session_id=session_id,
            operator_id=operator_id,
            app_id=resolved_app_id,
            start_state_hint=start_state_hint or start_url,
            start_url=start_url,
        )
    else:
        manual_session = ManualCaptureSession.from_raw(
            session_id=session_id,
            operator_id=operator_id,
            start_url=start_url,
        )
    for event in events:
        raw_evidence = event.get("evidence_bundle")
        evidence_bundle = raw_evidence if isinstance(raw_evidence, list) else []
        manual_session.record_action(
            action=str(event.get("action") or "unknown"),
            selector=str(event.get("selector") or "[manual]"),
            url_before=str(event.get("url_before") or start_url),
            url_after=str(event.get("url_after") or start_url),
            thought=str(event.get("thought") or ""),
            action_value=str(event.get("action_value") or ""),
            param_name=str(event.get("param_name") or ""),
            confidence_hint=_as_float(event.get("confidence_hint") or 0.9, 0.9),
            evidence_bundle=cast(list[EvidenceBundleItem], evidence_bundle),
            from_state_hint=str(event.get("from_state_hint") or ""),
            to_state_hint=str(event.get("to_state_hint") or ""),
        )
    result = await manual_session.to_cartography_result()
    await persist_mapping_result(
        app_id=resolved_app_id,
        app_name=app_name,
        session_id=session_id,
        resolved_url=start_url,
        current_url=start_url,
        inventory=[],
        initial_actions_log=[],
        result=result,
    )
    return resolved_app_id


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
    initialize_laminar()

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
    parser.add_argument(
        "--mode",
        default=os.getenv("MAPPING_MODE", "auto"),
        choices=["auto", "manual_graph_assisted", "manual_raw"],
        help="Run mode: auto (default), manual_graph_assisted, manual_raw.",
    )
    parser.add_argument(
        "--manual-events",
        default=os.getenv("MAPPING_MANUAL_EVENTS", ""),
        help="Path to manual capture events JSON for manual modes.",
    )
    parser.add_argument(
        "--manual-operator",
        default=os.getenv("MAPPING_MANUAL_OPERATOR", "human:operator"),
        help="Operator id for manual mode.",
    )
    parser.add_argument(
        "--manual-start-state",
        default=os.getenv("MAPPING_MANUAL_START_STATE", ""),
        help="Start state hint for manual_graph_assisted mode.",
    )
    parser.add_argument(
        "--manual-session-id",
        default=os.getenv("MAPPING_MANUAL_SESSION_ID", ""),
        help="Optional manual session id. Auto generated when omitted.",
    )
    args = parser.parse_args()
    output = (args.output or "").strip() or default_output
    inventory = (args.inventory or "").strip() or default_inventory
    scout_pages_arg = (args.scout_pages or "").strip()
    scout_pages = [item.strip() for item in scout_pages_arg.split(",") if item.strip()]
    merge_existing = bool(args.merge_existing)
    run_mode = (args.mode or "auto").strip()
    url = _resolve_mapping_url(args.url)

    async def _run() -> None:
        if browser_lifecycle.is_shutdown_requested():
            print("[INFO] Shutdown requested, exiting before mapping...")
            return

        # Scout module removed; create empty inventory for backward compat
        import json
        if not Path(inventory).exists():
            Path(inventory).parent.mkdir(parents=True, exist_ok=True)
            Path(inventory).write_text(json.dumps({"elements": []}), encoding="utf-8")

        if run_mode == "auto":
            print("Mapping (explore flow, build graph)...")
            try:
                _ = await run_mapping(
                    url=url,
                    output_path=output,
                    inventory_path=inventory,
                    merge_existing=merge_existing,
                )
            except asyncio.CancelledError:
                print("[INFO] Mapping cancelled")
                raise
            return

        manual_events_path = (args.manual_events or "").strip()
        if not manual_events_path:
            raise ValueError("--manual-events is required in manual mode.")
        manual_events_file = Path(manual_events_path)
        if not manual_events_file.exists():
            raise FileNotFoundError(f"Manual events file not found: {manual_events_file}")
        raw_manual = json.loads(manual_events_file.read_text(encoding="utf-8"))
        events = raw_manual if isinstance(raw_manual, list) else raw_manual.get("events", [])
        if not isinstance(events, list):
            raise ValueError("manual events must be a JSON list or {\"events\": [...]}")
        manual_session_id = (args.manual_session_id or "").strip() or (
            f"session:manual:{datetime.now(timezone.utc).isoformat()}"
        )
        from urllib.parse import urlparse

        parsed = urlparse(url)
        app_name = (parsed.netloc or parsed.path.split("/")[0] or "manual").strip()
        _ = await run_manual_mapping(
            app_name=app_name,
            session_id=manual_session_id,
            events=events,
            mode=run_mode,
            operator_id=(args.manual_operator or "human:operator").strip(),
            start_state_hint=(args.manual_start_state or "").strip(),
            start_url=url,
        )

    # Use a custom event loop to handle signals and cleanup properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Add signal handlers for the main thread
    def _sigint_handler():
        """Handle Ctrl+C in the main thread."""
        global _shutdown_requested
        _shutdown_requested = True
        browser_lifecycle.request_shutdown()
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
        sys.exit(130 if browser_lifecycle.is_shutdown_requested() else 0)


if __name__ == "__main__":
    main()

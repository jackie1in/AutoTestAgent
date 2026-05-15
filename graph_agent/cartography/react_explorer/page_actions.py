"""Custom exploration actions registered via browser-use's @tools.action() decorator.

Uses the documented pattern from https://docs.browser-use.com/open-source/customize/tools/add
with special injectable parameters (browser_session, page_extraction_llm) for
browser interaction and LLM access.
"""

import asyncio
import json
import logging
import re
from typing import Any

from browser_use import Tools
from browser_use.browser.events import TypeTextEvent
from browser_use.browser.session import BrowserSession
from browser_use.llm.base import BaseChatModel
from browser_use.tools.views import NoParamsAction
from pydantic import BaseModel, Field

from graph_agent.cartography.captcha import (
    resolve_captcha_solve_mode,
    solve_captcha_from_page,
)
from graph_agent.lib.page_controller.js_snippets import _EXTRACT_MENU_JS

logger = logging.getLogger(__name__)


class ScrollHorizontallyParams(BaseModel):
    direction: str = Field(default="right", description="Scroll direction: 'left' or 'right'")
    amount: int = Field(default=300, description="Pixels to scroll horizontally")
    index: int | None = Field(default=None, description="Optional element index to scroll within a specific container")


class QueryKnowledgeParams(BaseModel):
    query_text: str = Field(description="Natural language query about previously explored knowledge")
    target_type: str = Field(default="all", description="Which index to search: 'state' (pages), 'intent' (actions), or 'all'")


class SolveCaptchaParams(BaseModel):
    input_index: int | None = Field(default=None, description="Optional index of the captcha input element; tool locates captcha image nearby and fills result automatically")
    input_hint: str = Field(default="", description="Optional semantic hint for locating the captcha input, e.g. placeholder or label text")


class HumanHelpParams(BaseModel):
    reason: str = Field(description="Why you need human help, e.g. 'captcha too complex', 'unexpected login wall', 'stuck overlay'")


def _parse_menu_json(raw_text: str) -> dict:
    """Extract menu/action JSON from LLM response, handling markdown fences."""
    text = (raw_text or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r'\{[\s\S]*"items"[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return {"items": [], "error": "Failed to parse LLM response"}


def create_explorer_tools() -> Tools:
    """Build a browser-use Tools instance with all 6 project-specific custom actions.

    Actions registered:
      - scroll_horizontally  (direction, amount, index)
      - close_overlay        (no params)
      - query_knowledge      (query_text, target_type)
      - discover_zones       (no params)
      - extract_menu         (no params)
      - solve_captcha        (input_index?, input_hint?)

    Standard actions (click, input, select_dropdown, scroll, wait, send_keys,
    go_back, evaluate, done) are browser-use built-ins and are not re-registered.
    """
    tools = Tools(exclude_actions=["search"])

    # ── Shared state (closures) ──────────────────────────────────────
    _overlay_handler: Any = None  # lazy init
    _last_zone_analysis: Any = None

    def _get_overlay_handler(
        browser_session: BrowserSession,
        llm: BaseChatModel,
    ):
        nonlocal _overlay_handler
        if _overlay_handler is None:
            from graph_agent.cartography.react_explorer.overlay_handler import (
                OverlayHandler,
            )

            _overlay_handler = OverlayHandler(browser_session, llm, controller=None)
        return _overlay_handler

    # ── scroll_horizontally ──────────────────────────────────────────

    @tools.action(description="Scroll the page or element horizontally", param_model=ScrollHorizontallyParams)
    async def scroll_horizontally(
        params: ScrollHorizontallyParams,
        browser_session: BrowserSession,
    ) -> str:
        try:
            page = await browser_session.get_current_page()
            if page is None:
                return "Cannot scroll: no active page"

            if params.index is not None:
                node = await browser_session.get_element_by_index(params.index)
                if node is None:
                    return f"Element [{params.index}] not found for horizontal scroll"
                dx = params.amount if params.direction == "right" else -params.amount
                ok = await page.evaluate(
                    f"""
                    (() => {{
                        const el = document.evaluate(
                            {json.dumps(node.xpath)}, document, null,
                            XPathResult.FIRST_ORDERED_NODE_TYPE, null
                        ).singleNodeValue;
                        if (!el) return false;
                        el.scrollBy({{left: {dx}, behavior: 'smooth'}});
                        return true;
                    }})()
                    """
                )
                if ok:
                    return f"Scrolled element [{params.index}] horizontally by {params.amount}px"
                return f"Element [{params.index}] not resolvable"

            dx = params.amount if params.direction == "right" else -params.amount
            await page.evaluate(f"window.scrollBy({{left: {dx}, behavior: 'smooth'}})")
            return f"Scrolled page horizontally {params.direction} by {params.amount}px"
        except Exception as e:
            return f"Horizontal scroll failed: {e}"

    # ── close_overlay ────────────────────────────────────────────────

    @tools.action(description="Close modal/popup/dialog overlay", param_model=NoParamsAction)
    async def close_overlay(
        browser_session: BrowserSession,
        page_extraction_llm: BaseChatModel,
    ) -> str:
        handler = _get_overlay_handler(browser_session, page_extraction_llm)
        return await handler.close_overlay()

    # ── query_knowledge ──────────────────────────────────────────────

    @tools.action(description="Query knowledge base for transition hints", param_model=QueryKnowledgeParams)
    async def query_knowledge(
        params: QueryKnowledgeParams,
    ) -> str:
        if not params.query_text.strip():
            return "Knowledge query skipped: empty query text"
        try:
            from graph_agent.neo4j_client.manager import GraphManager

            async with GraphManager() as manager:
                if params.target_type.strip() == "state":
                    rows = await manager.run_read(
                        """
                        MATCH (s:State)
                        WHERE toLower(coalesce(s.title, '') + ' ' + coalesce(s.url, ''))
                              CONTAINS toLower($q)
                        RETURN s.id AS id, s.url AS url, s.title AS title
                        ORDER BY coalesce(s.last_visited, '') DESC
                        LIMIT 5
                        """,
                        q=params.query_text.strip(),
                    )
                    if not rows:
                        return f"No historical states matched: {params.query_text}"
                    preview = "; ".join(
                        f"{r.get('title') or r.get('id')}" for r in rows
                    )
                    return f"Historical state hints: {preview}"

                rows = await manager.run_read(
                    """
                    MATCH (t:Transition)
                    WHERE toLower(coalesce(t.selector, '') + ' ' + coalesce(t.thought, ''))
                          CONTAINS toLower($q)
                    RETURN t.id AS id, t.selector AS selector, t.confidence AS confidence
                    ORDER BY coalesce(t.confidence, 0.0) DESC
                    LIMIT 5
                    """,
                    q=params.query_text.strip(),
                )
                if not rows:
                    return f"No historical transitions matched: {params.query_text}"
                preview = "; ".join(
                    f"{r.get('selector')}({float(r.get('confidence') or 0.0):.2f})"
                    for r in rows
                )
                return f"Historical transition hints: {preview}"
        except Exception as e:
            return f"Knowledge query failed: {e}"

    # ── discover_zones ───────────────────────────────────────────────

    @tools.action(description="Trigger zone discovery analysis", param_model=NoParamsAction)
    async def discover_zones(
        browser_session: BrowserSession,
        page_extraction_llm: BaseChatModel,
    ) -> str:
        try:
            from graph_agent.cartography.llm_planning import analyze_page_with_llm
        except ImportError as e:
            return f"Zone discovery failed: cannot import analyzer — {e}"

        try:
            bss = await browser_session.get_browser_state_summary(
                include_screenshot=False, include_recent_events=False
            )
            dom_text = bss.dom_state.llm_representation()
            current_url = bss.url or ""
            page_title = bss.title or ""
        except Exception as e:
            return f"Zone discovery failed: cannot get browser state — {e}"

        if not dom_text:
            return "Zone discovery skipped: no DOM content available"
        if not current_url:
            return "Zone discovery skipped: cannot determine current URL"

        analysis = await analyze_page_with_llm(
            llm=page_extraction_llm,
            dom_text=dom_text,
            current_url=current_url,
            page_title=page_title,
        )

        zones = analysis.functional_zones
        if not zones:
            return (
                f"No functional zones detected on this page "
                f"(page_type={analysis.page_type})."
            )

        nonlocal _last_zone_analysis
        _last_zone_analysis = analysis

        lines = [
            f"Discovered {len(zones)} functional zone(s) on {current_url} "
            f"(page_type={analysis.page_type}):",
        ]
        for i, z in enumerate(zones, 1):
            desc = f" — {z.description}" if z.description else ""
            lines.append(f"  {i}. [{z.zone_type}] {z.selector}{desc}")
        return "\n".join(lines)

    # ── extract_menu ─────────────────────────────────────────────────

    @tools.action(description="Extract navigation menu structure", param_model=NoParamsAction)
    async def extract_menu(
        browser_session: BrowserSession,
        page_extraction_llm: BaseChatModel,
    ) -> str:
        try:
            page = await browser_session.get_current_page()
            if page is None:
                return json.dumps({"items": [], "error": "no active page"})
        except Exception as e:
            return json.dumps({"items": [], "error": str(e)})

        # Phase 1: JS-based extraction
        try:
            js_result = await page.evaluate(_EXTRACT_MENU_JS)
        except Exception as e:
            logger.warning("JS menu extraction failed: %s", e)
            js_result = ""

        try:
            parsed = (
                json.loads(js_result) if isinstance(js_result, str) else js_result
            )
            if isinstance(parsed, dict) and parsed.get("items"):
                return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Phase 2: LLM fallback
        try:
            from browser_use.dom.markdown_extractor import extract_clean_markdown
            from browser_use.llm.messages import SystemMessage, UserMessage

            content, _ = await extract_clean_markdown(
                browser_session=browser_session,
                extract_links=True,
                extract_images=False,
            )

            MAX_CHARS = 60000
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS] + "\n...[truncated]"

            prompt = f"""Extract the main navigation menu structure from this webpage content.
Return ONLY a valid JSON object (no markdown fences, no extra text) with this exact structure:
{{"items": [{{"text": "...", "href": "...", "level": 1, "tag": "a", "children": [...]}}]}}

Rules:
- Only extract the PRIMARY navigation (navbar, sidebar, hamburger menu)
- SKIP: footer links, breadcrumbs, content body links, utility links (login, signup, settings, language switcher)
- "text": menu item text (trim whitespace, max 80 chars)
- "href": URL path only (strip origin, keep path+query+hash), empty string "" if no link
- "level": nesting depth (1=top-level menu, 2=dropdown sub-item, 3=deeper nested)
- "tag": HTML tag name, usually "a" or "button"
- "children": array of sub-menu items with same structure (empty array [] if no children)
- PRESERVE hierarchical parent-child relationships

<page_content>
{content}
</page_content>"""

            response = await page_extraction_llm.ainvoke([
                SystemMessage(content="You are an expert at extracting navigation structure from web pages. Return ONLY valid JSON, no other text."),
                UserMessage(content=prompt),
            ])

            raw = (
                response.completion
                if hasattr(response, "completion")
                else str(response)
            )
            return json.dumps(_parse_menu_json(raw))
        except Exception as e:
            logger.warning("LLM-based menu extraction fallback failed: %s", e)
            return json.dumps({"items": [], "error": f"LLM fallback failed: {e}"})

    # ── solve_captcha ────────────────────────────────────────────────

    @tools.action(description="Detect and solve captcha on the page", param_model=SolveCaptchaParams)
    async def solve_captcha(
        params: SolveCaptchaParams,
        browser_session: BrowserSession,
        page_extraction_llm: BaseChatModel,
    ) -> str:
        page = await browser_session.get_current_page()
        if page is None:
            return "CAPTCHA_FAILED no_active_page"

        solve_mode = resolve_captcha_solve_mode()
        logger.info(
            "[CAPTCHA_FLOW] mode=%s input_index=%s input_hint=%s",
            solve_mode, params.input_index, params.input_hint[:40] if params.input_hint else "",
        )

        # --- Manual mode ---
        if solve_mode == "manual":
            from graph_agent.cartography.captcha import prompt_manual_captcha_code

            _MAX_MANUAL_RETRIES = 5
            for _attempt in range(1, _MAX_MANUAL_RETRIES + 1):
                captcha_code = await prompt_manual_captcha_code(page, params.input_hint)
                if not captcha_code:
                    if _attempt < _MAX_MANUAL_RETRIES:
                        continue
                    return f"CAPTCHA_MANUAL_EMPTY mode={solve_mode} attempts={_attempt}"
                fill_result = await _fill_captcha(
                    browser_session, page, captcha_code, params.input_index, params.input_hint
                )
                if (
                    "CAPTCHA_OK" in fill_result.upper()
                    or _attempt >= _MAX_MANUAL_RETRIES
                ):
                    return f"{fill_result} mode={solve_mode} attempts={_attempt}"
                await asyncio.sleep(0.3)
            return f"CAPTCHA_MANUAL_EXHAUSTED mode={solve_mode}"

        # --- Auto mode ---
        captcha_code = await solve_captcha_from_page(
            page, None,
            page_extraction_llm,
            input_hint=params.input_hint,
        )
        if not captcha_code:
            return f"CAPTCHA_EMPTY_CODE mode={solve_mode}"

        fill_result = await _fill_captcha(
            browser_session, page, captcha_code, params.input_index, params.input_hint
        )
        return f"{fill_result} mode={solve_mode}"

    # ── request_human_help ───────────────────────────────────────────

    @tools.action(description="Pause exploration and request human help", param_model=HumanHelpParams)
    async def request_human_help(
        params: HumanHelpParams,
        browser_session: BrowserSession,
    ) -> str:
        page = await browser_session.get_current_page()
        current_url = ""
        if page is not None:
            try:
                current_url = await page.get_url() or ""
            except Exception:
                pass

        from graph_agent.cartography.config import resolve_human_help_mode

        mode = resolve_human_help_mode()
        if mode == "auto":
            return (
                f"Human help skipped (auto mode). Reason: {params.reason}. "
                f"Try an alternative approach."
            )

        prompt = (
            f"\n{'=' * 60}\n"
            f"[HUMAN_HELP] Agent needs your intervention.\n"
            f"URL: {current_url}\n"
            f"Reason: {params.reason}\n"
            f"{'=' * 60}\n"
            f"Resolve the issue, then press Enter to continue..."
        )
        await asyncio.to_thread(input, prompt)
        return f"Human help completed. Reason: {params.reason}"

    return tools


# ------------------------------------------------------------------
# Captcha fill helper (standalone, not an action itself)
# ------------------------------------------------------------------


async def _fill_captcha(
    browser_session: BrowserSession,
    page: Any,
    captcha_code: str,
    input_index: int | None,
    input_hint: str,
) -> str:
    """Fill captcha code into target input using browser-use event bus."""

    # Path A: index-based fill via TypeTextEvent
    if input_index is not None:
        node = await browser_session.get_element_by_index(input_index)
        if node is not None:
            event = browser_session.event_bus.dispatch(
                TypeTextEvent(node=node, text=captcha_code, clear=True)
            )
            await event
            return (
                f"CAPTCHA_OK filled_index={input_index} "
                f"code_len={len(captcha_code)}"
            )
        return f"CAPTCHA_FILL_FAILED index={input_index} not_found"

    # Path B: semantic matching (no index → find by captcha hint pattern)
    import re as _re

    hint_pattern = (
        _re.escape(input_hint)
        if input_hint
        else r"captcha|验证码|verify.*code|auth.*code"
    )
    fill_script = """
    (...args) => {{
        const [captchaCode, hintPattern] = args;
        const re = new RegExp(hintPattern, 'i');
        const inputs = document.querySelectorAll('input');
        for (const inp of inputs) {{
            const t = inp.type || 'text';
            if (t === 'password') continue;
            const sig = ((inp.name || '') + (inp.id || '') + (inp.placeholder || '') + (inp.className || '') + (inp.getAttribute('aria-label') || '')).toLowerCase();
            if (re.test(sig)) {{
                inp.focus();
                const proto = Object.getPrototypeOf(inp);
                const desc = proto && Object.getOwnPropertyDescriptor(proto, 'value');
                if (desc && desc.set) desc.set.call(inp, captchaCode); else inp.value = captchaCode;
                inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return true;
            }}
        }}
        return false;
    }}
    """
    filled = bool(await page.evaluate(fill_script, captcha_code, hint_pattern))
    if filled:
        return f"CAPTCHA_OK filled_by_semantic_match code_len={len(captcha_code)}"
    return "CAPTCHA_FILL_FAILED target_input_not_found"

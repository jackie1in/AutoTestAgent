"""PageActions — first-class action handlers for the ReAct explorer.

Each handler is a named async method, composable and independently testable.
The ``register()`` method injects them into a browser-use ``Registry`` so
``BaseAgent._execute_action → Tools.act() → registry.execute_action()``
dispatches to these handlers automatically.

Handler method names use descriptive Python names (e.g. ``input_text``);
``register()`` overrides ``fn.__name__`` to the LLM-facing action_type string
(e.g. ``"input"``) before registration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import TYPE_CHECKING

from browser_use.tools.registry.service import Registry

from graph_agent.cartography.captcha import (
    resolve_captcha_solve_mode,
    solve_captcha_from_page,
)
from graph_agent.cartography.config import (
    resolve_agent_marker,
    resolve_auto_marker_enabled,
)
from graph_agent.cartography.react_explorer.overlay_handler import OverlayHandler
from graph_agent.lib.page_controller import PageController

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.llm.base import BaseChatModel
    from graph_agent.cartography.llm_planning import LLMPageAnalysis

logger = logging.getLogger(__name__)

_PageActionsSharedWait = list[float]  # mutable reference: [total_wait_time]

# Fields that get the agent marker auto-prepended (configurable via env)
_NAME_FIELD_RE = re.compile(
    (os.getenv("CARTOGRAPHY_NAME_FIELD_PATTERNS") or r"name|title|名称|标题|姓名"),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PageActions
# ---------------------------------------------------------------------------


class PageActions:
    """Action handler registry backed by PageController (W3C event simulation).

    Delegates complex handlers:
    - OverlayHandler → close_overlay (5-phase dismissal)
    - captcha/ package → solve_captcha
    """

    def __init__(
        self,
        controller: PageController | None,
        browser: "BrowserSession | None",
        llm: "BaseChatModel",
        total_wait_time_ref: _PageActionsSharedWait | None = None,
    ):
        self._controller = controller
        self._browser = browser
        self._llm = llm
        self._total_wait_time_ref: _PageActionsSharedWait = (
            total_wait_time_ref if total_wait_time_ref is not None else [0.0]
        )
        self._last_zone_analysis: LLMPageAnalysis | None = None
        self._overlay_handler = OverlayHandler(browser, llm, controller)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, registry: Registry, supported_actions: set[str]) -> None:
        """Wire custom action handlers into *registry* and update *supported_actions*.

        Only registers actions that browser-use does NOT provide or
        that need custom logic on top of browser-use built-ins:
        - W3C click (overrides browser-use CDP click)
        - input (overrides: auto-prepends agent marker to name/title fields)
        - dropdown_options (overrides: detects non-standard dropdowns)
        - scroll_horizontally, close_overlay
        - query_knowledge, discover_zones, extract_menu, solve_captcha

        All others (select_dropdown, scroll, wait, send_keys, go_back,
        evaluate, done, etc.) use browser-use built-ins.
        """
        _self = self

        # -- click (W3C pointer events override) --
        async def click(index: int) -> str:
            return await _self.click(index)

        click.__name__ = "click"
        registry.action(description="Click element by index using W3C pointer events")(
            click
        )
        supported_actions.add("click")

        # -- input (override: auto-prepend agent marker to name/title fields) --
        async def input(index: int, text: str) -> str:
            return await _self.input_text(index, text)

        input.__name__ = "input"
        registry.action(
            description="Click and type text into an input element; auto-prepends agent marker for name/title fields"
        )(input)
        supported_actions.add("input")

        # -- dropdown_options (override: detect non-standard dropdowns) --
        async def dropdown_options(index: int) -> str:
            return await _self._dropdown_options(index)

        dropdown_options.__name__ = "dropdown_options"
        registry.action(
            description="Get dropdown option values. Only works on native <select> elements — returns guidance for custom dropdowns."
        )(dropdown_options)
        supported_actions.add("dropdown_options")

        # -- scroll_horizontally --
        async def scroll_horizontally(
            direction: str = "right", amount: int = 300, index: int | None = None
        ) -> str:
            return await _self.scroll_horizontally(direction, amount, index)

        scroll_horizontally.__name__ = "scroll_horizontally"
        registry.action(description="Scroll the page or element horizontally")(
            scroll_horizontally
        )
        supported_actions.add("scroll_horizontally")

        # -- close_overlay --
        async def close_overlay() -> str:
            return await _self.close_overlay()

        close_overlay.__name__ = "close_overlay"
        registry.action(description="Close modal/popup/dialog overlay")(close_overlay)
        supported_actions.add("close_overlay")

        # -- query_knowledge --
        async def query_knowledge(query_text: str, target_type: str = "all") -> str:
            return await _self.query_knowledge(query_text, target_type)

        query_knowledge.__name__ = "query_knowledge"
        registry.action(description="Query knowledge base for transition hints")(
            query_knowledge
        )
        supported_actions.add("query_knowledge")

        # -- discover_zones --
        async def discover_zones() -> str:
            return await _self.discover_zones()

        discover_zones.__name__ = "discover_zones"
        registry.action(description="Trigger zone discovery analysis")(discover_zones)
        supported_actions.add("discover_zones")

        # -- extract_menu --
        async def extract_menu() -> str:
            return await _self.extract_menu()

        extract_menu.__name__ = "extract_menu"
        registry.action(description="Extract navigation menu structure")(extract_menu)
        supported_actions.add("extract_menu")

        # -- solve_captcha --
        async def solve_captcha(
            input_index: int | None = None, input_hint: str = ""
        ) -> str:
            return await _self.solve_captcha(input_index, input_hint)

        solve_captcha.__name__ = "solve_captcha"
        registry.action(description="Detect and solve captcha on the page")(
            solve_captcha
        )
        supported_actions.add("solve_captcha")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def click(self, index: int) -> str:
        ctrl = self._require_controller()
        r = await ctrl.click_element(index)
        return r.message

    async def input_text(self, index: int, text: str) -> str:
        """Type text into an input element, auto-prepending agent marker for name/title fields."""
        ctrl = self._require_controller()
        if resolve_auto_marker_enabled():
            marker = resolve_agent_marker()
            if marker:
                elem_desc = ctrl._element_text_map.get(index, "")
                if _NAME_FIELD_RE.search(elem_desc) and marker not in text:
                    text = f"{marker} {text}"
        r = await ctrl.input_text(index, text)
        return r.message

    async def _dropdown_options(self, index: int) -> str:
        """Check element type and return dropdown options or guidance."""
        ctrl = self._require_controller()
        node = ctrl.selector_map.get(index)
        tag = (getattr(node, "tag_name", "") or "").lower() if node else ""

        if tag != "select":
            return (
                f"Element [{index}] is <{tag}>, NOT a native <select>. "
                f"Skip dropdown_options/select_dropdown. Instead: "
                f"click this element to open the dropdown panel, "
                f"then click the desired option directly."
            )

        # Standard <select> — extract options via JS
        browser = self._browser
        if browser is None:
            return "Cannot read dropdown options: no browser"

        try:
            page = await browser.get_current_page()
            if page is None:
                return "Cannot read dropdown options: no active page"

            element = await ctrl._get_element(index)
            if element is None:
                return f"Element [{index}] not found"

            raw = await element.evaluate(
                """(el) => {
                    const opts = [];
                    for (const o of el.options || []) {
                        opts.push({index: o.index, text: o.text, value: o.value});
                    }
                    return JSON.stringify(opts);
                }"""
            )
            options = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if not options:
                return f"No options found in <select> [{index}]."

            lines = [f"Dropdown options for <{tag}> [{index}]:"]
            for o in options:
                lines.append(f"  {o['index']}: {o['text'][:60]}")
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to read dropdown options: {e}"

    async def scroll_horizontally(
        self, direction: str = "right", amount: int = 300, index: int | None = None
    ) -> str:
        ctrl = self._require_controller()
        r = await ctrl.scroll_horizontally(direction, amount, index)
        return r.message

    async def close_overlay(self) -> str:
        return await self._overlay_handler.close_overlay()

    async def query_knowledge(self, query_text: str, target_type: str = "all") -> str:
        return await self._query_knowledge(query_text.strip(), target_type.strip())

    async def discover_zones(self) -> str:
        """Analyze the current page's functional zones via LLM.

        Returns a structured zone list the agent can use to focus exploration
        on high-value interactive areas (forms, tables, action_bars, etc.).
        """
        try:
            from graph_agent.cartography.llm_planning import analyze_page_with_llm
        except ImportError as e:
            return f"Zone discovery failed: cannot import analyzer — {e}"

        ctrl = self._controller
        browser = self._browser
        if ctrl is None:
            return "Zone discovery failed: no page controller"

        dom_text = ctrl.simplified_html or ""
        if not dom_text:
            return "Zone discovery skipped: no DOM content available"

        current_url = ""
        page_title = ""
        if browser is not None:
            try:
                current_url = await browser.get_current_page_url() or ""
                page = await browser.get_current_page()
                if page is not None:
                    page_title = await page.get_title() or ""
            except Exception:
                pass

        if not current_url:
            return "Zone discovery skipped: cannot determine current URL"

        analysis = await analyze_page_with_llm(
            llm=self._llm,
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

        lines = [
            f"Discovered {len(zones)} functional zone(s) on {current_url} "
            f"(page_type={analysis.page_type}):",
        ]
        for i, z in enumerate(zones, 1):
            desc = f" — {z.description}" if z.description else ""
            lines.append(f"  {i}. [{z.zone_type}] {z.selector}{desc}")

        self._last_zone_analysis = analysis
        return "\n".join(lines)

    async def extract_menu(self) -> str:
        ctrl = self._require_controller()
        # 1. Try JS-based extraction first (fast, zero API cost)
        js_result = await ctrl.extract_menu_structure()
        try:
            parsed = json.loads(js_result)
            if parsed.get("items"):
                return js_result
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. JS returned empty — fall back to LLM-based extraction
        browser = self._browser
        if browser is None:
            return js_result

        try:
            from browser_use.dom.markdown_extractor import extract_clean_markdown
            from browser_use.llm.messages import SystemMessage, UserMessage

            content, _ = await extract_clean_markdown(
                browser_session=browser, extract_links=True, extract_images=False
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

            response = await self._llm.ainvoke([
                SystemMessage(content="You are an expert at extracting navigation structure from web pages. Return ONLY valid JSON, no other text."),
                UserMessage(content=prompt),
            ])

            raw = response.completion if hasattr(response, 'completion') else str(response)
            parsed = _parse_menu_json(raw)
            return json.dumps(parsed)
        except Exception as e:
            logger.warning("LLM-based menu extraction fallback failed: %s", e)
            return js_result

    async def solve_captcha(
        self, input_index: int | None = None, input_hint: str = ""
    ) -> str:
        from graph_agent.cartography.captcha import (
            fill_captcha_code,
            prompt_manual_captcha_code,
        )

        browser = self._browser
        if browser is None:
            return "CAPTCHA_FAILED no browser"
        page = await browser.get_current_page()
        if page is None:
            return "CAPTCHA_FAILED no_active_page"
        ctrl = self._require_controller()
        solve_mode = resolve_captcha_solve_mode()
        input_hint_present = bool(input_hint.strip())
        logger.info(
            "[CAPTCHA_FLOW] mode=%s input_hint_present=%s input_index=%s",
            solve_mode, input_hint_present, input_index,
        )
        if solve_mode == "manual":
            _MAX_MANUAL_RETRIES = 5
            for _attempt in range(1, _MAX_MANUAL_RETRIES + 1):
                captcha_code = await prompt_manual_captcha_code(page, input_hint)
                if not captcha_code:
                    if _attempt < _MAX_MANUAL_RETRIES:
                        continue
                    return f"CAPTCHA_MANUAL_EMPTY mode={solve_mode} attempts={_attempt}"
                fill_result = await fill_captcha_code(
                    page=page, controller=ctrl,
                    captcha_code=captcha_code, input_index=input_index, input_hint=input_hint,
                )
                if "CAPTCHA_OK" in fill_result.upper() or _attempt >= _MAX_MANUAL_RETRIES:
                    return f"{fill_result} mode={solve_mode} attempts={_attempt}"
                await asyncio.sleep(0.3)
            return f"CAPTCHA_MANUAL_EXHAUSTED mode={solve_mode}"
        else:
            captcha_code = await solve_captcha_from_page(
                page, None, self._llm, input_hint=input_hint,
            )
            if not captcha_code:
                return f"CAPTCHA_EMPTY_CODE mode={solve_mode}"
        fill_result = await fill_captcha_code(
            page=page, controller=ctrl,
            captcha_code=captcha_code, input_index=input_index, input_hint=input_hint,
        )
        return f"{fill_result} mode={solve_mode}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_controller(self) -> PageController:
        ctrl = self._controller
        if ctrl is None:
            raise RuntimeError("PageController not set — call explore_page() first")
        return ctrl

    async def _query_knowledge(self, query_text: str, target_type: str) -> str:
        if not query_text:
            return "Knowledge query skipped: empty query text"
        try:
            from graph_agent.neo4j_client.manager import GraphManager

            async with GraphManager() as manager:
                if target_type == "state":
                    rows = await manager.run_read(
                        """
                        MATCH (s:State)
                        WHERE toLower(coalesce(s.title, '') + ' ' + coalesce(s.url, ''))
                              CONTAINS toLower($q)
                        RETURN s.id AS id, s.url AS url, s.title AS title
                        ORDER BY coalesce(s.last_visited, '') DESC
                        LIMIT 5
                        """,
                        q=query_text,
                    )
                    if not rows:
                        return f"No historical states matched: {query_text}"
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
                    q=query_text,
                )
                if not rows:
                    return f"No historical transitions matched: {query_text}"
                preview = "; ".join(
                    f"{r.get('selector')}({float(r.get('confidence') or 0.0):.2f})"
                    for r in rows
                )
                return f"Historical transition hints: {preview}"
        except Exception as e:
            return f"Knowledge query failed: {e}"

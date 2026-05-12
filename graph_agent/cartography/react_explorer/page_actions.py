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
import logging
import re
import time
from typing import TYPE_CHECKING

from browser_use.tools.registry.service import Registry

from graph_agent.cartography.captcha import (
    normalize_manual_captcha_code,
    resolve_captcha_solve_mode,
    solve_captcha_from_page,
)
from graph_agent.lib.page_controller import PageController

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.llm.base import BaseChatModel
    from playwright.async_api import Page as PlaywrightPage

logger = logging.getLogger(__name__)

_PageActionsSharedWait = list[float]  # mutable reference: [total_wait_time]


# ---------------------------------------------------------------------------
# PageActions
# ---------------------------------------------------------------------------


class PageActions:
    """Action handler registry backed by PageController (W3C event simulation).

    Each public async method is an action handler.  ``register()`` wires them
    into a browser-use ``Registry`` so they become available to the LLM via
    the auto-generated ``ActionModel``.
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

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, registry: Registry, supported_actions: set[str]) -> None:
        """Wire all action handlers into *registry* and update *supported_actions*.

        Creates thin wrapper closures with the correct parameter signatures
        and action-type names.  Delegates to named handler methods so each
        handler remains independently testable.
        """
        _self = self

        # -- click --
        async def click(index: int) -> str:
            return await _self.click(index)

        click.__name__ = "click"
        registry.action(
            description="Click element by index using W3C pointer events"
        )(click)
        supported_actions.add("click")

        # -- input --
        async def input(index: int, text: str) -> str:
            return await _self.input_text(index, text)

        input.__name__ = "input"
        registry.action(description="Type text into input field by index")(input)
        supported_actions.add("input")

        # -- select_dropdown --
        async def select_dropdown(index: int, option_text: str = "") -> str:
            return await _self.select_dropdown(index, option_text)

        select_dropdown.__name__ = "select_dropdown"
        registry.action(description="Select dropdown option by index")(
            select_dropdown
        )
        supported_actions.add("select_dropdown")

        # -- scroll --
        async def scroll(
            down: bool = True, pages: float = 1.0, index: int | None = None
        ) -> str:
            return await _self.scroll(down, pages, index)

        scroll.__name__ = "scroll"
        registry.action(description="Scroll the page or element vertically")(scroll)
        supported_actions.add("scroll")

        # -- scroll_horizontally --
        async def scroll_horizontally(
            direction: str = "right", amount: int = 300, index: int | None = None
        ) -> str:
            return await _self.scroll_horizontally(direction, amount, index)

        scroll_horizontally.__name__ = "scroll_horizontally"
        registry.action(
            description="Scroll the page or element horizontally"
        )(scroll_horizontally)
        supported_actions.add("scroll_horizontally")

        # -- wait --
        async def wait(seconds: int = 1) -> str:
            return await _self.wait(seconds)

        wait.__name__ = "wait"
        registry.action(description="Wait for specified seconds")(wait)
        supported_actions.add("wait")

        # -- close_overlay --
        async def close_overlay() -> str:
            return await _self.close_overlay()

        close_overlay.__name__ = "close_overlay"
        registry.action(description="Close modal/popup/dialog overlay")(
            close_overlay
        )
        supported_actions.add("close_overlay")

        # -- execute_javascript --
        async def execute_javascript(script: str) -> str:
            return await _self.execute_javascript(script)

        execute_javascript.__name__ = "execute_javascript"
        registry.action(description="Execute JavaScript in the page")(
            execute_javascript
        )
        supported_actions.add("execute_javascript")

        # -- query_knowledge --
        async def query_knowledge(query_text: str, target_type: str = "all") -> str:
            return await _self.query_knowledge(query_text, target_type)

        query_knowledge.__name__ = "query_knowledge"
        registry.action(
            description="Query knowledge base for transition hints"
        )(query_knowledge)
        supported_actions.add("query_knowledge")

        # -- discover_zones --
        async def discover_zones() -> str:
            return await _self.discover_zones()

        discover_zones.__name__ = "discover_zones"
        registry.action(description="Trigger zone discovery analysis")(
            discover_zones
        )
        supported_actions.add("discover_zones")

        # -- extract_menu --
        async def extract_menu() -> str:
            return await _self.extract_menu()

        extract_menu.__name__ = "extract_menu"
        registry.action(description="Extract navigation menu structure")(
            extract_menu
        )
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
        ctrl = self._require_controller()
        r = await ctrl.input_text(index, text)
        return r.message

    async def select_dropdown(self, index: int, option_text: str = "") -> str:
        ctrl = self._require_controller()
        r = await ctrl.select_option(index, option_text)
        return r.message

    async def scroll(self, down: bool = True, pages: float = 1.0, index: int | None = None) -> str:
        ctrl = self._require_controller()
        direction = "down" if down else "up"
        amount = int(pages * 500)
        r = await ctrl.scroll(direction, amount, index)
        return r.message

    async def scroll_horizontally(
        self, direction: str = "right", amount: int = 300, index: int | None = None
    ) -> str:
        ctrl = self._require_controller()
        r = await ctrl.scroll_horizontally(direction, amount, index)
        return r.message

    async def wait(self, seconds: int = 1) -> str:
        ctrl = self._require_controller()
        seconds = min(seconds, 10)
        self._total_wait_time_ref[0] += seconds
        last = await ctrl.get_last_update_time()
        elapsed = time.time() - last if last > 0 else 0
        actual = max(0, seconds - elapsed)
        await asyncio.sleep(actual)
        return f"Waited {seconds}s (actual {actual:.1f}s)"

    async def go_back(self) -> str:
        browser = self._browser
        if browser is None:
            return "Go back failed: no browser"
        page = await browser.get_current_page()
        if page is None:
            return "Go back failed: no active page"
        await page.go_back()
        return "Navigated back"

    async def close_overlay(self) -> str:
        browser = self._browser
        if browser is None:
            return "Close overlay failed: no browser"
        page = await browser.get_current_page()
        if page is None:
            return "Close overlay failed: no active page"
        closed = await self._close_overlays(page)
        return f"Closed {closed} overlay(s)"

    async def execute_javascript(self, script: str) -> str:
        ctrl = self._require_controller()
        r = await ctrl.execute_javascript(script)
        return r.message

    async def query_knowledge(self, query_text: str, target_type: str = "all") -> str:
        return await self._query_knowledge(query_text.strip(), target_type.strip())

    async def discover_zones(self) -> str:
        return "Zone discovery delegated to pipeline analysis"

    async def extract_menu(self) -> str:
        return "Menu extraction delegated to pipeline analysis"

    async def send_keys(self, keys: str) -> str:
        ctrl = self._require_controller()
        parts = keys.split("+")
        mods = {
            "Control": "ctrlKey", "Ctrl": "ctrlKey",
            "Alt": "altKey", "Shift": "shiftKey", "Meta": "metaKey",
        }
        key = parts[-1]
        opts = ",".join(f"{mods[m]}:true" for m in parts[:-1] if m in mods)
        script = (
            f"(function(){{var e=new KeyboardEvent('keydown',{{key:'{key}',{opts}}});"
            f"document.dispatchEvent(e);"
            f"e=new KeyboardEvent('keyup',{{key:'{key}',{opts}}});"
            f"document.dispatchEvent(e);}})()"
        )
        r = await ctrl.execute_javascript(script)
        return f"Sent keys: {keys}"

    async def solve_captcha(self, input_index: int | None = None, input_hint: str = "") -> str:
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
                pause_started = time.monotonic()
                captcha_code = await self._prompt_manual_captcha_code(page, input_hint)
                manual_wait_ms = int((time.monotonic() - pause_started) * 1000)
                if not captcha_code:
                    if _attempt < _MAX_MANUAL_RETRIES:
                        continue
                    return f"CAPTCHA_MANUAL_EMPTY mode={solve_mode} attempts={_attempt}"
                fill_result = await self._fill_captcha_code(
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
        fill_result = await self._fill_captcha_code(
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

    async def _close_overlays(self, page: "PlaywrightPage") -> int:
        try:
            result = await page.evaluate(
                """() => {
                    let closed = 0;
                    document.querySelectorAll(
                        '.ant-drawer-close, .ant-modal-close, '
                        + '.el-drawer__close-btn, .el-dialog__close'
                    ).forEach(btn => { btn.click(); closed++; });
                    if (closed === 0) {
                        const mask = document.querySelector(
                            '.ant-drawer-mask, .ant-modal-mask, .ant-modal-wrap'
                        );
                        if (mask && getComputedStyle(mask).display !== 'none') {
                            document.dispatchEvent(
                                new KeyboardEvent('keydown',
                                    {key: 'Escape', keyCode: 27, bubbles: true})
                            );
                            closed++;
                        }
                    }
                    return closed;
                }"""
            )
            return int(result) if result else 0
        except Exception:
            return 0

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

    async def _prompt_manual_captcha_code(
        self, page: "PlaywrightPage", input_hint: str
    ) -> str:
        current_url = ""
        try:
            current_url = await page.get_url() or ""
        except Exception:
            current_url = ""
        hint = input_hint.strip() or "<none>"
        prompt = (
            "\n[CAPTCHA][MANUAL] ReAct paused for manual captcha input.\n"
            f"URL: {current_url or '<unknown>'}\n"
            f"input_hint: {hint}\n"
            "Please type captcha code and press Enter: "
        )
        try:
            raw = await asyncio.to_thread(input, prompt)
        except EOFError:
            return ""
        return normalize_manual_captcha_code(raw)

    async def _fill_captcha_code(
        self,
        *,
        page: "PlaywrightPage",
        controller: PageController,
        captcha_code: str,
        input_index: int | None,
        input_hint: str,
    ) -> str:
        if input_index is not None:
            input_result = await controller.input_text(input_index, captcha_code)
            return (
                f"CAPTCHA_OK filled_index={input_index} "
                f"code_len={len(captcha_code)} result={input_result.message}"
            )

        hint_pattern = (
            re.escape(input_hint) if input_hint else r"captcha|验证码|verify.*code|auth.*code"
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
            return "CAPTCHA_OK filled_by_semantic_match code_len={}".format(len(captcha_code))
        return "CAPTCHA_FILL_FAILED target_input_not_found"

    @staticmethod
    def _extract_captcha_result_code(result_text: str) -> str:
        match = re.search(r"(CAPTCHA_[A-Z_]+)", str(result_text or "").upper())
        return match.group(1) if match else "CAPTCHA_UNKNOWN"

    @staticmethod
    def _extract_captcha_fill_path(result_text: str) -> str:
        text = str(result_text or "").lower()
        if "filled_index=" in text:
            return "input_index"
        if "filled_by_semantic_match" in text:
            return "semantic_match"
        if "fill_failed" in text:
            return "fill_failed"
        return "none"

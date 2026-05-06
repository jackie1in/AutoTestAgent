"""ReAct-based page explorer — inherits BaseAgent ReAct loop.

Eliminates duplicated loop code by reusing BaseAgent's observe→think→act
infrastructure, while keeping PageController for browser interactions and
CartographyResult for output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import cast
from urllib.parse import parse_qsl, urlparse

from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography.base_agent import BaseAgent
from graph_agent.cartography.captcha import solve_captcha_from_page
from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    infer_transition_semantics,
)
from graph_agent.cartography.react_prompts import (
    build_system_prompt,
    build_user_prompt,
)
from graph_agent.cartography.react_schema import AgentOutput
from graph_agent.graph.merger import CartographyResult
from graph_agent.lib.page_controller import PageController
from graph_agent.llm import get_llm
from graph_agent.models import (
    ActionType,
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointTiming,
    Severity,
    State,
    Transition,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 500
_LLM_MAX_RETRIES = 3
_LLM_TIMEOUT_MS = 60_000


def _extract_spa_route(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.fragment or parsed.path or "/"


def _build_data_signature(url: str) -> str:
    parsed = urlparse(url or "")
    query_keys = sorted(k for k, _ in parse_qsl(parsed.query, keep_blank_values=True))
    raw = "|".join(query_keys)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _build_state_identity(url: str, spa_route: str, view_fingerprint: str) -> str:
    """Build a stable state identity for cross-session reuse."""
    parsed = urlparse(url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    route = (spa_route or "/").strip() or "/"
    view = (view_fingerprint or "").strip() or "no-view-fp"
    digest = hashlib.md5(f"{origin}|{route}|{view}".encode("utf-8")).hexdigest()[:16]
    return f"state:{digest}"


def _rank_selector_chain(
    selector: str, attrs: dict[str, object], tag: str
) -> list[str]:
    chain: list[str] = []
    test_id = attrs.get("data-testid") or attrs.get("data-test") or attrs.get("data-qa")
    if test_id:
        chain.append(f'[data-testid="{test_id}"]')
    element_id = attrs.get("id")
    if element_id:
        chain.append(f"#{element_id}")
    role = attrs.get("role")
    name = attrs.get("name")
    if role and name:
        chain.append(f'{tag}[role="{role}"][name="{name}"]')
    elif role:
        chain.append(f'{tag}[role="{role}"]')
    if name:
        chain.append(f'{tag}[name="{name}"]')
    if selector and selector not in chain:
        chain.append(selector)
    if not chain:
        chain.append(f"[selector:{selector or '?'}]")
    return chain


class ReActExplorer(BaseAgent):
    """Lightweight page/zone explorer using BaseAgent's ReAct loop.

    Overrides:
    - Prompt construction (via react_prompts builders)
    - Action execution (via PageController instead of browser-use Tools)
    - State-change recording (transitions + intent inference)
    - History format (evaluation/memory/next_goal fields)
    """

    def __init__(
        self,
        max_steps: int = _DEFAULT_MAX_STEPS,
        browser_session: "Browser | None" = None,
        initial_actions: list[dict[str, object]] | None = None,
        initial_history: list[dict[str, object]] | None = None,
        extra_system_prompt: str = "",
        use_vision: bool | str = "auto",
        vision_detail_level: str = "auto",
    ):
        llm = get_llm()
        env_use_vision = (os.getenv("BROWSER_USE_USE_VISION") or "").strip()
        env_vision_detail = (os.getenv("BROWSER_USE_VISION_DETAIL_LEVEL") or "").strip()
        super().__init__(
            task="Explore the page and record all interactive elements and transitions.",
            llm=llm,
            browser=browser_session,
            max_steps=max_steps,
            total_max_steps=max_steps,
            initial_history=initial_history,
            initial_actions=initial_actions,
            start_url="",
            use_vision=env_use_vision or use_vision,
            vision_detail_level=env_vision_detail or vision_detail_level,
        )
        self._browser_session = browser_session
        self._controller: PageController | None = None
        self._explored_indices: set[int] = set()
        self._total_wait_time = 0.0
        self._last_url = ""
        self._page_title = ""
        self._state_id = ""
        self._result = CartographyResult()
        self._extra_system_prompt = extra_system_prompt
        self._semantic_conflict_count = 0

        # Extra actions supported by PageController but not in BaseAgent defaults
        self._supported_actions.update(
            {
                "scroll_horizontally",
                "close_overlay",
                "execute_javascript",
                "query_knowledge",
                "discover_zones",
                "extract_menu",
                "solve_captcha",
            }
        )
        self._dynamic_action_model = self._build_dynamic_action_model()

    # ------------------------------------------------------------------
    # Prompt hooks
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        prompt = build_system_prompt(max_steps=self.max_steps)
        if self._extra_system_prompt:
            prompt += f"\n\n{self._extra_system_prompt}"
        return prompt

    def _build_user_prompt(
        self,
        dom_text: str,
        history: list[dict],
        step: int,
        current_url: str,
        page_title: str = "",
    ) -> str:
        observations: list[str] = []

        if self._total_wait_time >= 3:
            observations.append(
                f"You have waited {self._total_wait_time:.0f} seconds accumulatively. "
                "DO NOT wait any longer unless you have a good reason."
            )

        if current_url != self._last_url:
            if self._last_url:
                observations.append(f"Page navigated to → {current_url}")
            self._last_url = current_url

        remaining = self.total_max_steps - step
        if remaining == 5:
            observations.append(
                f"Only {remaining} steps remaining. "
                "Consider wrapping up or calling done with partial results."
            )
        elif remaining == 2:
            observations.append(
                f"Critical: Only {remaining} steps left! "
                "You must finish the task or call done immediately."
            )

        return build_user_prompt(
            browser_state_text=dom_text,
            history=history,
            explored_indices=self._explored_indices,
            step=step,
            max_steps=self.total_max_steps,
            page_title=page_title,
            observations=observations,
        )

    # ------------------------------------------------------------------
    # History hook — capture LLM reasoning fields
    # ------------------------------------------------------------------

    def _make_history_entry(
        self,
        step: int,
        action_type: str,
        result_text: str,
        output: AgentOutput | None = None,
    ) -> dict[str, object]:
        return {
            "evaluation_previous_goal": (
                getattr(output, "evaluation_previous_goal", "") if output else ""
            ),
            "memory": getattr(output, "memory", "") if output else "",
            "next_goal": getattr(output, "next_goal", "") if output else "",
            "action_name": action_type,
            "action_result": result_text,
        }

    # ------------------------------------------------------------------
    # Observation — use PageController as the single source of truth so
    # the DOM the LLM sees and the selector_map we click against come
    # from the SAME update_tree() call. Otherwise PageController's
    # _is_indexed flag is never set and every click fails with
    # "DOM tree not indexed. Call update_tree() first.".
    # ------------------------------------------------------------------

    async def _get_browser_snapshot(self) -> tuple[str, str, dict]:
        controller = self._controller
        if controller is None:
            return await super()._get_browser_snapshot()
        try:
            dom_text = await controller.update_tree()
        except Exception as e:
            logger.warning(
                "PageController.update_tree() failed, falling back to browser_use snapshot: %s",
                e,
            )
            return await super()._get_browser_snapshot()

        title = ""
        try:
            _browser = self.browser
            if _browser is not None:
                page = await _browser.get_current_page()
                if page is not None:
                    title = await page.get_title() or ""
        except Exception:
            pass

        selector_map = getattr(controller, "selector_map", {}) or {}
        return dom_text, title, dict(selector_map)

    # ------------------------------------------------------------------
    # Action execution — delegate to PageController
    # ------------------------------------------------------------------

    async def _execute_action(self, action_type: str, params: dict[str, object]) -> str:
        controller = self._controller
        if controller is None:
            return "Controller not initialized"

        browser = self.browser
        assert browser is not None, (
            "BrowserSession must be set before executing actions"
        )

        # Track explored indices for click/input/select
        if action_type in ("click", "input", "select_dropdown"):
            _raw_idx = params.get("index")
            if _raw_idx is not None:
                self._explored_indices.add(cast(int, _raw_idx))

        try:
            match action_type:
                case "click":
                    idx = cast(int, params.get("index", 0))
                    r = await controller.click_element(idx)
                    return r.message
                case "input":
                    idx = cast(int, params.get("index", 0))
                    text = cast(str, params.get("text", ""))
                    r = await controller.input_text(idx, text)
                    return r.message
                case "select_dropdown":
                    idx = cast(int, params.get("index", 0))
                    opt = cast(str, params.get("option_text", params.get("text", "")))
                    r = await controller.select_option(idx, opt)
                    return r.message
                case "scroll":
                    # Schema uses down (bool) + pages (float)
                    down = params.get("down", True)
                    pages = cast(float, params.get("pages", 1.0))
                    direction = "down" if down else "up"
                    amount = int(pages * 500)
                    _idx_raw = params.get("index")
                    idx: int | None = (
                        cast(int, _idx_raw) if _idx_raw is not None else None
                    )
                    r = await controller.scroll(direction, amount, idx)
                    return r.message
                case "scroll_horizontally":
                    direction = cast(str, params.get("direction", "right"))
                    amount = cast(int, params.get("amount", params.get("pixels", 300)))
                    _idx_raw2 = params.get("index")
                    idx2: int | None = (
                        cast(int, _idx_raw2) if _idx_raw2 is not None else None
                    )
                    r = await controller.scroll_horizontally(direction, amount, idx2)
                    return r.message
                case "wait":
                    seconds = min(cast(int, params.get("seconds", 1)), 10)
                    self._total_wait_time += seconds
                    last_update = await controller.get_last_update_time()
                    elapsed = time.time() - last_update if last_update > 0 else 0
                    actual = max(0, seconds - elapsed)
                    await asyncio.sleep(actual)
                    return f"Waited {seconds}s (actual {actual:.1f}s)"
                case "go_back":
                    try:
                        page = await browser.get_current_page()
                        if page is None:
                            return "Go back failed: no active page"
                        await page.go_back()
                        return "Navigated back"
                    except Exception as e:
                        return f"Go back failed: {e}"
                case "close_overlay":
                    try:
                        page = await browser.get_current_page()
                        closed = await self._close_overlays(page)
                        return f"Closed {closed} overlay(s)"
                    except Exception as e:
                        return f"Close overlay failed: {e}"
                case "execute_javascript":
                    script = cast(str, params.get("script", ""))
                    r = await controller.execute_javascript(script)
                    return r.message
                case "query_knowledge":
                    query_text = cast(str, params.get("query_text") or "").strip()
                    target_type = cast(str, params.get("target_type") or "all").strip()
                    return await self._query_knowledge(query_text, target_type)
                case "discover_zones":
                    return "Zone discovery delegated to pipeline analysis"
                case "extract_menu":
                    return "Menu extraction delegated to pipeline analysis"
                case "solve_captcha":
                    page = await browser.get_current_page()
                    if page is None:
                        return "CAPTCHA_FAILED no_active_page"
                    _input_index_raw = params.get("input_index")
                    input_index: int | None = (
                        cast(int, _input_index_raw)
                        if _input_index_raw is not None
                        else None
                    )
                    input_hint = str(params.get("input_hint") or "")
                    # Pass None as login_info — solve_captcha_from_page will infer
                    # image scope from the DOM directly, anchored by input_index/input_hint
                    # rather than assuming a password-form login context.
                    captcha_code = await solve_captcha_from_page(
                        page, None, self.llm, input_hint=input_hint
                    )
                    if captcha_code and input_index is not None:
                        input_result = await controller.input_text(
                            input_index, captcha_code
                        )
                        return (
                            f"CAPTCHA_OK filled_index={input_index} "
                            f"code_len={len(captcha_code)} result={input_result.message}"
                        )
                    if captcha_code:
                        # Fallback: locate the target input by hint or general captcha keywords
                        hint_pattern = (
                            re.escape(input_hint)
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
                        filled = bool(
                            await page.evaluate(fill_script, captcha_code, hint_pattern)
                        )
                        if filled:
                            return (
                                "CAPTCHA_OK filled_by_semantic_match "
                                f"code_len={len(captcha_code)}"
                            )
                        return "CAPTCHA_FILL_FAILED target_input_not_found"
                    return "CAPTCHA_EMPTY_CODE"
                case _:
                    return f"Unknown action: {action_type}"
        except Exception as e:
            return f"Action {action_type} failed: {e}"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def _on_before_step(self, step: int) -> None:
        # Health check every 10 steps
        if step % 10 == 0:
            try:
                browser = self.browser
                if browser is None:
                    return
                page = await browser.get_current_page()
                if page is not None:
                    await page.evaluate("() => 1")
            except Exception as e:
                logger.warning("Browser health check failed at step %d: %s", step, e)

    async def _on_after_step(
        self,
        step: int,
        action_type: str,
        result_text: str,
        changed: bool,
        url_before: str,
        url_after: str,
    ) -> None:
        if action_type != "wait":
            self._total_wait_time = 0

    async def _on_state_changed(
        self,
        step: int,
        url_before: str,
        url_after: str,
        action_type: str,
        output: AgentOutput,
        fp_before: str = "",
        fp_after: str = "",
    ) -> None:
        """Record transition for CartographyResult when state changes."""
        if action_type not in ("click", "input", "select_dropdown"):
            return

        try:
            (
                dom_text_after,
                title_after,
                selector_map_after,
            ) = await self._get_browser_snapshot()
            fp_after = self._compute_page_fingerprint(
                dom_text_after, title_after, selector_map_after
            )
        except Exception:
            return

        from_state_id = _build_state_identity(
            url_before,
            _extract_spa_route(url_before),
            fp_before or self._compute_dom_fingerprint(url_before),
        )
        to_state_id = _build_state_identity(
            url_after,
            _extract_spa_route(url_after),
            fp_after or self._compute_dom_fingerprint(url_after),
        )

        from_state = State(
            id=from_state_id,
            url=url_before,
            title=self._page_title or title_after,
            spa_route=_extract_spa_route(url_before),
            fingerprint=fp_before,
            view_fingerprint=fp_before,
            data_signature=_build_data_signature(url_before),
        )
        to_state = State(
            id=to_state_id,
            url=url_after,
            fingerprint=fp_after,
            title=f"{self._page_title or title_after} after {action_type}",
            spa_route=_extract_spa_route(url_after),
            view_fingerprint=fp_after,
            data_signature=_build_data_signature(url_after),
        )

        act_type = {
            "click": ActionType.CLICK,
            "input": ActionType.FILL,
            "select_dropdown": ActionType.SELECT,
        }.get(action_type, ActionType.CLICK)

        idx = getattr(output.action, "index", None)
        selector = f"[{idx}]" if idx is not None else "[?]"
        selector_chain = [selector]
        element_snapshot_json: str | None = None
        if idx is not None and self._controller is not None:
            node = self._controller.selector_map.get(idx)
            if node is not None:
                attrs = getattr(node, "attributes", {}) or {}
                tag = (getattr(node, "tag_name", "") or "").strip() or "*"
                node_selector = (
                    getattr(node, "css_selector", "")
                    or getattr(node, "xpath", "")
                    or selector
                )
                selector_chain = _rank_selector_chain(node_selector, attrs, tag)
                selector = selector_chain[0]
                element_snapshot_json = json.dumps(
                    {
                        "selector": selector,
                        "xpath": getattr(node, "xpath", None),
                        "css_selector": getattr(node, "css_selector", None),
                        "name": attrs.get("name"),
                        "id": attrs.get("id"),
                        "class_name": attrs.get("class"),
                        "type": attrs.get("type"),
                        "tag_name": tag,
                        "text_content": getattr(node, "node_value", ""),
                        "attributes": attrs,
                        "frame_path": [],
                    },
                    ensure_ascii=False,
                )
        thought_text = output.next_goal or ""

        transition = Transition(
            id=f"t:{self._state_id or 'root'}:react-{action_type}-{step}",
            selector=selector,
            selector_chain=selector_chain,
            semantic_action_key=f"{act_type.value}:{selector}",
            action=act_type,
            thought=thought_text,
            intent=None,
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            step_index=step,
            element_snapshot=element_snapshot_json,
            evidence_ids=[
                f"evidence:{self._state_id or 'root'}:{step}:url_change",
                f"evidence:{self._state_id or 'root'}:{step}:dom_after",
            ],
        )
        try:
            neighbor_steps: list[dict[str, str]] | None = None
            if self._result.history:
                neighbor_steps = [
                    {
                        "action": str(h.get("action_name") or h.get("action") or ""),
                        "selector": str(h.get("selector") or ""),
                        "source_url": str(h.get("url_before") or ""),
                        "target_url": str(h.get("url_after") or ""),
                        "thought": str(h.get("next_goal") or ""),
                    }
                    for h in self._result.history[-3:]
                    if isinstance(h, dict)
                ]
            page_signals = {"title": self._page_title or title_after, "url": url_before}
            semantic = await infer_transition_semantics(
                SemanticInferenceInput(
                    source_type="auto",
                    operator_id="agent",
                    action=act_type,
                    selector=selector,
                    selector_chain_hint=selector_chain,
                    source_url=url_before,
                    target_url=url_after,
                    param_name=None,
                    thought_text=thought_text,
                    neighbor_steps=neighbor_steps,
                    page_signals=page_signals,
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    step_index=step,
                    transition_id=transition.id,
                    existing_semantic_keys=cast(
                        "set[str]",
                        {
                            t.semantic_action_key
                            for t in self._result.transitions
                            if getattr(t, "semantic_action_key", None) is not None
                        },
                    ),
                )
            )
            transition.intent = semantic.transition_patch.intent
            transition.intent_failure_reason = (
                semantic.transition_patch.intent_failure_reason
            )
            transition.selector_chain = semantic.transition_patch.selector_chain
            transition.semantic_action_key = (
                semantic.transition_patch.semantic_action_key
            )
            if semantic.transition_patch.confidence_hint > 0:
                transition.confidence = semantic.transition_patch.confidence_hint
            if semantic.conflict_flags:
                self._semantic_conflict_count += 1
            extra_checkpoints = semantic.checkpoints
        except Exception as e:
            logger.debug("    → Semantic inference skipped: %s", e)
            extra_checkpoints = []

        cp = Checkpoint(
            id=f"cp:{transition.id}:after",
            layer=CheckpointLayer.STRUCTURAL,
            timing=CheckpointTiming.AFTER,
            expect=CheckpointExpect.SHOULD_PASS,
            severity=Severity.MAJOR,
            rule_type="url_changed" if url_after != url_before else "dom_changed",
            description=thought_text or f"After {action_type}, page should change",
        )

        self._result.states.extend([from_state, to_state])
        self._result.transitions.append(transition)
        self._result.checkpoints.append(cp)
        self._result.checkpoints.extend(extra_checkpoints)
        self._result.checkpoint_transition_map[cp.id] = transition.id
        for extra in extra_checkpoints:
            self._result.checkpoint_transition_map[extra.id] = transition.id
        self._result.semantic_conflict_count = self._semantic_conflict_count
        logger.info("    → Transition recorded: %s", transition.id)

        if url_after != url_before:
            try:
                _browser = self.browser
                if _browser is not None:
                    page = await _browser.get_current_page()
                    if page is not None:
                        await page.go_back()
            except Exception:
                pass
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Run override — return CartographyResult
    # ------------------------------------------------------------------

    async def explore_page(
        self,
        session: "Browser | None" = None,
        state_id: str = "",
        page_title: str = "",
        *,
        start_url: str = "",
        max_steps: int | None = None,
    ) -> CartographyResult:
        """Explore a page/zone and return CartographyResult.

        Supports two calling conventions:
        1. ``explore_page(session, state_id, page_title)`` — used by mapping pipeline.
        2. ``explore_page(start_url=url, max_steps=n)`` — used by maximal explorer.
        """
        # Handle maximal-explorer keyword convention
        if start_url:
            if self._browser_session is None:
                raise RuntimeError(
                    "ReActExplorer requires a BrowserSession when start_url is given"
                )
            bs = self._browser_session
            # Navigate to start_url if provided
            try:
                await bs.navigate_to(start_url)
            except Exception as e:
                logger.warning("Failed to navigate to start_url %s: %s", start_url, e)
            self._state_id = f"state:{start_url}"
        else:
            if session is None and self._browser_session is None:
                raise RuntimeError("ReActExplorer requires a BrowserSession")
            bs = session or self._browser_session
            self._state_id = state_id

        assert bs is not None, "BrowserSession must not be None at this point"
        self._controller = PageController(bs)
        self._page_title = page_title
        self._result = CartographyResult()
        self._explored_indices.clear()
        self._total_wait_time = 0.0
        self._last_url = ""
        self._semantic_conflict_count = 0

        # Temporarily override max_steps if requested
        original_max_steps = self.max_steps
        if max_steps is not None:
            self.max_steps = max_steps
            self.total_max_steps = max_steps

        # Reset token tracker
        from graph_agent.lib.token_tracker import reset_global_tracker

        reset_global_tracker()

        try:
            # Run the BaseAgent loop
            agent_result = await self.run()

            # Attach history to result
            self._result.history = agent_result.history
        finally:
            # Restore max_steps
            if max_steps is not None:
                self.max_steps = original_max_steps
                self.total_max_steps = original_max_steps

        # Log token usage
        from graph_agent.lib.token_tracker import get_global_tracker

        get_global_tracker().log_summary()

        logger.info(
            "ReAct exploration complete: %d steps, %d transitions, %d checkpoints",
            len(agent_result.history),
            len(self._result.transitions),
            len(self._result.checkpoints),
        )

        # Dispose controller
        if self._controller:
            try:
                self._controller.dispose()
            except Exception as e:
                logger.warning("Failed to dispose controller: %s", e)

        return self._result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _close_overlays(self, page) -> int:
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
        """Query historical graph knowledge to guide next exploration step."""
        if not query_text:
            return "Knowledge query skipped: empty query text"
        try:
            from graph_agent.neo4j_client.manager import GraphManager

            async with GraphManager() as manager:
                if target_type == "state":
                    rows = await manager._run_read(
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

                rows = await manager._run_read(
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

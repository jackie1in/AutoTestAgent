"""ReActExplorerBase — single-inheritance explorer using BaseAgent's ReAct loop.

Merges the former _ReActExplorerCore and _ReActExplorerExtended into one class
that directly inherits BaseAgent.  Action handlers live in PageActions (imported),
eliminating the old diamond-inheritance pattern.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography.base_agent import BaseAgent
from graph_agent.cartography.captcha import normalize_manual_captcha_code
from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    infer_transition_semantics,
)
from graph_agent.cartography.react_explorer.page_actions import PageActions
from graph_agent.cartography.react_explorer.selector_utils import _rank_selector_chain
from graph_agent.cartography.react_explorer.transition_utils import (
    _infer_param_name_from_snapshot,
    _should_commit_transition,
    _transition_dedupe_key,
)
from graph_agent.cartography.react_explorer.url_utils import (
    _build_data_signature,
    _build_state_identity,
    _extract_spa_route,
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
    TransitionStep,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 500


class ReActExplorerBase(BaseAgent):
    """Page/zone explorer using BaseAgent's ReAct loop with PageController actions.

    Overrides:
    - Prompt construction (via react_prompts builders)
    - Action execution (via PageActions → PageController W3C events)
    - State-change recording (transitions + intent inference)
    - History format (evaluation/memory/next_goal fields)
    """

    llm: Any  # BaseChatModel from get_llm(); incompatible library types

    def __init__(
        self,
        max_steps: int = _DEFAULT_MAX_STEPS,
        browser_session: "Browser | None" = None,
        initial_actions: list[dict[str, object]] | None = None,
        initial_history: list[dict[str, object]] | None = None,
        extra_system_prompt: str = "",
        use_vision: bool | str = "auto",
        vision_detail_level: str = "auto",
        target_zone_selectors: list[str] | None = None,
    ):
        llm = get_llm()
        env_use_vision = (os.getenv("BROWSER_USE_USE_VISION") or "").strip()
        env_vision_detail = (
            os.getenv("BROWSER_USE_VISION_DETAIL_LEVEL") or ""
        ).strip()
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
        self._page_actions: PageActions | None = None
        self._explored_indices: set[int] = set()

        # _total_wait_time is a mutable list so PageActions can mutate it.
        self._total_wait_time: list[float] = [0.0]
        self._last_url = ""
        self._page_title = ""
        self._state_id = ""
        self._result = CartographyResult()
        self._extra_system_prompt = extra_system_prompt
        self._semantic_conflict_count = 0
        self._target_zone_selectors: list[str] | None = (
            list(target_zone_selectors) if target_zone_selectors else None
        )

        # Intent-level transition accumulation
        self._pending_steps: list[TransitionStep] = []
        self._pending_from_state_id = ""
        self._pending_from_url = ""
        self._pending_from_fp = ""
        self._explore_start_url = ""

        # Actions are registered lazily in explore_page() once the controller
        # and browser are wired.
        # self._register_actions() is called from explore_page().

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

        if self._total_wait_time[0] >= 3:
            observations.append(
                f"You have waited {self._total_wait_time[0]:.0f} seconds accumulatively. "
                "DO NOT wait any longer unless you have a good reason."
            )

        if current_url != self._last_url:
            if self._last_url:
                observations.append(f"Page navigated to → {current_url}")
            self._last_url = current_url

        remaining_pct = (self.total_max_steps - step) / max(self.total_max_steps, 1)
        remaining = self.total_max_steps - step
        if remaining_pct < 0.1:
            observations.append(
                f"Only {remaining} steps remaining. You MUST finish NOW."
            )

        # Invalidate zone_filter when page has navigated away from the start URL.
        effective_zone_selectors = self._target_zone_selectors
        if effective_zone_selectors and self._explore_start_url:
            _current_clean = _extract_spa_route(current_url) or current_url.split("?")[0].split("#")[0]
            _start_clean = _extract_spa_route(self._explore_start_url) or self._explore_start_url.split("?")[0].split("#")[0]
            if _current_clean != _start_clean:
                effective_zone_selectors = None

        return build_user_prompt(
            browser_state_text=dom_text,
            history=history,
            explored_indices=self._explored_indices,
            step=step,
            max_steps=self.total_max_steps,
            page_title=page_title,
            observations=observations,
            target_zone_selectors=effective_zone_selectors,
        )

    # ------------------------------------------------------------------
    # History hook
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
    # Observation — use PageController as the single source of truth
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

        # Keep _page_title in sync for state recording
        if title:
            self._page_title = title

        selector_map = getattr(controller, "selector_map", {}) or {}
        return dom_text, title, dict(selector_map)

    # ------------------------------------------------------------------
    # Action registration — via PageActions
    # ------------------------------------------------------------------

    def _register_actions(self) -> None:
        """Register PageController-backed actions into browser-use registry.

        Called from explore_page() after the controller is wired.
        """
        self._page_actions = PageActions(
            controller=self._controller,
            browser=self.browser,
            llm=self.llm,
            total_wait_time_ref=self._total_wait_time,
        )
        self._page_actions.register(self.tools.registry, self._supported_actions)
        self._dynamic_action_model = self._build_dynamic_action_model()

    # ------------------------------------------------------------------
    # Proxy methods — delegate to PageActions for test monkeypatch surface
    # ------------------------------------------------------------------

    async def _prompt_manual_captcha_code(self, page, input_hint: str) -> str:
        pa = self._page_actions
        if pa is None:
            return ""
        return await pa._prompt_manual_captcha_code(page, input_hint)

    async def _fill_captcha_code(
        self,
        *,
        page,
        controller: PageController,
        captcha_code: str,
        input_index: int | None,
        input_hint: str,
    ) -> str:
        pa = self._page_actions
        if pa is None:
            return "CAPTCHA_FILL_FAILED no_page_actions"
        return await pa._fill_captcha_code(
            page=page,
            controller=controller,
            captcha_code=captcha_code,
            input_index=input_index,
            input_hint=input_hint,
        )

    @staticmethod
    def _extract_captcha_result_code(result_text: str) -> str:
        return PageActions._extract_captcha_result_code(result_text)

    @staticmethod
    def _extract_captcha_fill_path(result_text: str) -> str:
        return PageActions._extract_captcha_fill_path(result_text)

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
            self._total_wait_time[0] = 0.0

    # ------------------------------------------------------------------
    # Transition building
    # ------------------------------------------------------------------

    def _build_transition_step(
        self,
        step: int,
        action_type: str,
        output: AgentOutput,
    ) -> TransitionStep:
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

        param_name = _infer_param_name_from_snapshot(element_snapshot_json)
        thought_text = output.next_goal or ""

        return TransitionStep(
            action=act_type,
            selector=selector,
            selector_chain=selector_chain,
            param_name=param_name,
            action_value=None,
            element_snapshot=element_snapshot_json,
            thought=thought_text,
            step_index=step,
            semantic_action_key=f"{act_type.value}:{selector}",
        )

    async def _flush_pending_transition(self) -> None:
        """Commit accumulated pending steps as an intent-level Transition."""
        if not self._pending_steps:
            return

        steps = list(self._pending_steps)
        self._pending_steps = []

        last_step = steps[-1]
        first_step = steps[0]
        overall_action = last_step.action
        overall_selector = last_step.selector
        overall_selector_chain = last_step.selector_chain
        overall_thought = last_step.thought or first_step.thought or ""
        overall_element_snapshot = last_step.element_snapshot
        overall_param_name = last_step.param_name or first_step.param_name

        try:
            dom_text_after, title_after, selector_map_after = (
                await self._get_browser_snapshot()
            )
            fp_after = self._compute_page_fingerprint(
                dom_text_after, title_after, selector_map_after
            )
        except Exception:
            fp_after = ""

        current_url = ""
        _browser = self.browser
        if _browser is not None:
            try:
                page = await _browser.get_current_page()
                if page is not None:
                    current_url = page.url
            except Exception:
                pass

        from_state_id = self._pending_from_state_id
        to_state_id = _build_state_identity(
            current_url or self._pending_from_url,
            _extract_spa_route(current_url or self._pending_from_url),
            fp_after or "no-view-fp",
        )

        from_state = State(
            id=from_state_id,
            url=self._pending_from_url,
            title=self._page_title or "",
            spa_route=_extract_spa_route(self._pending_from_url),
            fingerprint=self._pending_from_fp,
            view_fingerprint=self._pending_from_fp,
            data_signature=_build_data_signature(self._pending_from_url),
        )
        to_state = State(
            id=to_state_id,
            url=current_url or self._pending_from_url,
            title=self._page_title or "",
            spa_route=_extract_spa_route(current_url or self._pending_from_url),
            fingerprint=fp_after,
            view_fingerprint=fp_after,
            data_signature=_build_data_signature(
                current_url or self._pending_from_url
            ),
        )

        state_digest = (
            from_state_id.replace("state:", "", 1) if from_state_id else "root"
        )
        semantic_suffix = f"react-{overall_action.value}-{first_step.step_index}"
        transition_id = f"t:{state_digest}:{semantic_suffix}"

        transition = Transition(
            id=transition_id,
            selector=overall_selector,
            selector_chain=overall_selector_chain,
            semantic_action_key=f"{overall_action.value}:{overall_selector}",
            action=overall_action,
            thought=overall_thought,
            intent=None,
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            step_index=first_step.step_index or 0,
            element_snapshot=overall_element_snapshot,
            param_name=overall_param_name,
            steps=steps,
            evidence_ids=[
                f"evidence:{state_digest}:{first_step.step_index}:url_change",
                f"evidence:{state_digest}:{first_step.step_index}:dom_after",
            ],
        )

        # Run semantic inference with param_name
        try:
            neighbor_steps: list[dict[str, str]] | None = None
            if self._result.history:
                neighbor_steps = [
                    {
                        "action": str(
                            h.get("action_name") or h.get("action") or ""
                        ),
                        "selector": str(h.get("selector") or ""),
                        "source_url": str(h.get("url_before") or ""),
                        "target_url": str(h.get("url_after") or ""),
                        "thought": str(h.get("next_goal") or ""),
                    }
                    for h in self._result.history[-3:]
                    if isinstance(h, dict)
                ]
            page_signals = {
                "title": self._page_title or "",
                "url": self._pending_from_url,
            }
            semantic = await infer_transition_semantics(
                SemanticInferenceInput(
                    source_type="auto",
                    operator_id="agent",
                    action=overall_action,
                    selector=overall_selector,
                    selector_chain_hint=overall_selector_chain,
                    source_url=self._pending_from_url,
                    target_url=current_url or self._pending_from_url,
                    param_name=overall_param_name,
                    thought_text=overall_thought,
                    neighbor_steps=neighbor_steps,
                    page_signals=page_signals,
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    step_index=first_step.step_index or 0,
                    transition_id=transition.id,
                    existing_semantic_keys=cast(
                        "set[str]",
                        {
                            t.semantic_action_key
                            for t in self._result.transitions
                            if getattr(t, "semantic_action_key", None)
                            is not None
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
                transition.confidence = (
                    semantic.transition_patch.confidence_hint
                )
            if semantic.conflict_flags:
                self._semantic_conflict_count += 1
            extra_checkpoints = semantic.checkpoints
        except Exception as e:
            logger.debug("    → Semantic inference skipped: %s", e)
            extra_checkpoints = []

        if transition.intent and transition.intent.key:
            intent_suffix = transition.intent.key
            transition.id = f"t:{state_digest}:{intent_suffix}"

        cp = Checkpoint(
            id=f"cp:{transition.id}:after",
            layer=CheckpointLayer.STRUCTURAL,
            timing=CheckpointTiming.AFTER,
            expect=CheckpointExpect.SHOULD_PASS,
            severity=Severity.MAJOR,
            rule_type=(
                "url_changed"
                if current_url != self._pending_from_url
                else "dom_changed"
            ),
            description=overall_thought
            or "After intent-level transition, page should change",
        )

        self._result.states.extend([from_state, to_state])
        dedupe_key = _transition_dedupe_key(transition)
        existing_keys = {
            _transition_dedupe_key(item)
            for item in self._result.transitions[-30:]
        }
        if dedupe_key in existing_keys:
            logger.info("    → Transition deduped: %s", dedupe_key)
            return
        self._result.transitions.append(transition)
        self._result.checkpoints.append(cp)
        self._result.checkpoints.extend(extra_checkpoints)
        self._result.checkpoint_transition_map[cp.id] = transition.id
        for extra in extra_checkpoints:
            self._result.checkpoint_transition_map[extra.id] = transition.id
        self._result.semantic_conflict_count = self._semantic_conflict_count

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
        if action_type not in ("click", "input", "select_dropdown"):
            return

        from_state_id = _build_state_identity(
            url_before,
            _extract_spa_route(url_before),
            fp_before or self._compute_dom_fingerprint(url_before),
        )

        if not self._pending_steps:
            self._pending_from_state_id = from_state_id
            self._pending_from_url = url_before
            self._pending_from_fp = fp_before or ""

        ts = self._build_transition_step(step, action_type, output)
        self._pending_steps.append(ts)

        should_commit = _should_commit_transition(
            url_before, url_after, action_type
        )
        if should_commit:
            await self._flush_pending_transition()

    # ------------------------------------------------------------------
    # Public entry point
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
        if start_url:
            if self._browser_session is None:
                raise RuntimeError(
                    "ReActExplorer requires a BrowserSession when start_url is given"
                )
            bs = self._browser_session
            try:
                await bs.navigate_to(start_url)
            except Exception as e:
                logger.warning(
                    "Failed to navigate to start_url %s: %s", start_url, e
                )
            self._state_id = _build_state_identity(
                start_url, _extract_spa_route(start_url), ""
            )
        else:
            if session is None and self._browser_session is None:
                raise RuntimeError("ReActExplorer requires a BrowserSession")
            bs = session or self._browser_session
            self._state_id = state_id

        assert bs is not None, "BrowserSession must not be None at this point"
        self._controller = PageController(bs)

        # Wire actions now that controller is ready.
        self._register_actions()

        self._page_title = page_title
        self._result = CartographyResult()
        self._explored_indices.clear()
        self._total_wait_time[0] = 0.0
        self._last_url = ""
        self._semantic_conflict_count = 0
        self._pending_steps = []
        self._pending_from_state_id = ""
        self._pending_from_url = ""
        self._pending_from_fp = ""
        self._explore_start_url = ""
        try:
            _start_page = await bs.get_current_page()
            if _start_page is not None:
                self._explore_start_url = await _start_page.get_url() or ""
        except Exception:
            pass

        original_max_steps = self.max_steps
        if max_steps is not None:
            self.max_steps = max_steps
            self.total_max_steps = max_steps

        from graph_agent.lib.token_tracker import reset_global_tracker

        reset_global_tracker()

        try:
            agent_result = await self.run()
            self._result.history = agent_result.history
        finally:
            if self._pending_steps:
                await self._flush_pending_transition()
            if max_steps is not None:
                self.max_steps = original_max_steps
                self.total_max_steps = original_max_steps

        from graph_agent.lib.token_tracker import get_global_tracker

        get_global_tracker().log_summary()

        logger.info(
            "ReAct exploration complete: %d steps, %d transitions, %d checkpoints",
            len(agent_result.history),
            len(self._result.transitions),
            len(self._result.checkpoints),
        )

        if self._controller:
            try:
                self._controller.dispose()
            except Exception as e:
                logger.warning("Failed to dispose controller: %s", e)

        return self._result


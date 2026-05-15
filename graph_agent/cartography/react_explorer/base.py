"""ExplorerAgent — browser-use Agent subclass for web application exploration.

Inherits browser-use Agent's battle-tested infrastructure (error recovery, step
timeout, MessageManager, SignalHandler) and injects project-specific behaviors:
- Custom actions (6 total) via PageActions → Tools.registry
- DOM patch 3 (required fields) via on_step_start callback
- Runtime budget + should_stop checks via on_step_start callback
- State change detection + fingerprint + transition building via on_step_end callback
- Semantic inference for intent labeling
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    infer_transition_semantics,
)
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
from graph_agent.cartography.runtime_watchdog import RuntimeWatchdog
from graph_agent.graph.merger import CartographyResult
from graph_agent.lib.page_controller.js_snippets import _PATCH_REQUIRED_FIELDS_JS
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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Effectively unlimited — runtime budget is the real governor
_UNLIMITED_STEPS = 10**8

_EXPLORATION_SYSTEM_RULES = """\
<goals>
You are an AI agent that systematically explores a web application to build a
complete map (graph) of its pages, interactive elements, and transitions.
1. Explore the current page: discover interactive elements, click them, observe changes.
2. Extract the navigation menu structure so you can navigate to other pages.
3. Follow menu items in depth-first order: explore one branch fully, then the next.
4. Report all discovered transitions when you've exhausted the page.
When you believe you have discovered all meaningful interactive elements and
transitions on the current page, call done. Do not wait for a step limit —
you decide when exploration is complete.
</goals>

<menu_discovery>
Navigation menus in SPAs often use internal routing. Explore ALL menu branches
using this depth-first pattern:
  1. Open the menu (send_keys shortcut or click menu trigger).
  2. Call extract_menu to get the full tree.
  3. BEFORE exploring any items, write a plan file (todo.md) listing every menu
     item as a checkbox `- [ ]`. This is your exploration checklist — you MUST
     create it before the first menu click. Update it as you progress.
  4. Explore items in depth-first order from top to bottom. After exploring one
     page fully, check it off (`- [x]`), return to HOME, re-open the menu,
     and proceed to the next unchecked sibling.
  5. Only call done when ALL items in your plan are checked off.
</menu_discovery>

<zone_discovery>
Call discover_zones as the FIRST action on every new page to understand the
page's functional layout. It identifies zones by type (form, table, action_bar,
filter, modal, tabs, etc.) with CSS selectors. Use these to prioritize
exploration: forms > action_bars > tables > tabs > content.
</zone_discovery>

<form_exploration>
Forms are HIGH-VALUE exploration targets — each input field is a transition
that will be used to generate test scripts. Even forms with 50+ fields must be
fully explored. Do NOT abandon a form because it is "too complex" or taking many
steps — every field you fill is a recorded transition, and there is no step limit.
When you open a form:
  1. Systematically fill EVERY visible input, select, checkbox, and radio ONE AT A TIME.
  2. After filling ALL fields, click the submit/save/confirm button.
     → If save succeeds: observe the result page (list updated, toast message).
     → If save fails (page stays, form still visible, no success message):
       a. Scan for unfilled required fields: elements with required attribute,
          aria-required="true", red border-color, red asterisk *.
          Fill EVERY one you find — do NOT skip any.
       b. Click submit again. Repeat this retry loop up to 3 times.
       c. Only after 3 failed submit attempts, give up — click back/cancel.
       d. IMPORTANT: do NOT stop exploring the form after one failed submit.
          "不确定" or "uncertain" evaluation means RETRY, not give up.
  3. Skip submit only if the form is clearly destructive (delete, remove, reset password).
  4. Use dropdown_options before select_dropdown to discover available options.
     If dropdown_options says the element is NOT a native <select>, do NOT use
     select_dropdown — instead click the element to open the dropdown panel,
     then click the desired option directly.
</form_exploration>

<dropdown_handling>
Before using dropdown_options or select_dropdown, understand the dropdown type:
1. STANDARD (<select> tag): dropdown_options lists options, select_dropdown selects one.
2. NON-STANDARD (anything not <select>):
   a. Click the trigger element to open the dropdown panel
   b. Click the specific option within that panel
   c. For tree selects: expand parent nodes first, then click the leaf
   d. For searchable selects: input text to filter, then click the match
   e. AFTER selecting an option, the dropdown panel may stay open — the value
      might NOT have been applied yet. Click somewhere else on the page
      (a blank area, form body, or press Escape) to close the dropdown and
      commit the selection. Then observe the field to verify it was filled.
      If the value is still empty, the click may have only expanded a tree
      node — try clicking the child item or a checkbox next to the option.
</dropdown_handling>

<data_safety>
Before destructive operations (delete/remove/reset):
  1. SCAN for [AUTO] records in data table. If none, skip destructive test.
  2. Only operate on ONE record at a time.
Form fields with name/title automatically receive an "[AUTO]" prefix by the system.
</data_safety>

<rules>
- Only interact with elements that have a numeric [index].
- After clicking, observe the result. Close any drawer/modal/overlay that appeared.
- Before starting DFS exploration, write a todo.md plan listing every menu item as a
  checkbox. Mark items `- [x]` as you complete them. Always know what's next.
- Skip elements that only reload data (e.g. "刷新", "导出", "重置").
- When you encounter an input with placeholder containing "搜索", first input "测试" then observe the result.
- Switch to EVERY tab panel to discover hidden content.
- Use send_keys for keyboard shortcuts (e.g. Alt+Z) if menus are hidden behind icons.
- Use scroll_horizontally to reveal hidden columns in wide tables/carousels.
</rules>

<language>
Respond in 中文 for evaluation/memory/next_goal fields.
</language>
"""


def _build_extend_system_message(extra_system_prompt: str = "") -> str:
    """Build the extend_system_message string with exploration rules."""
    if extra_system_prompt:
        return _EXPLORATION_SYSTEM_RULES + "\n\n" + extra_system_prompt
    return _EXPLORATION_SYSTEM_RULES


class ExplorerAgent(Agent):
    """Exploration agent inheriting browser-use Agent's loop + infrastructure.

    Injects custom actions, state change detection, transition building,
    and semantic inference via on_step_start / on_step_end callbacks.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        browser_session: "Browser | None" = None,
        initial_actions: list[dict[str, object]] | None = None,
        initial_history: list[dict[str, object]] | None = None,
        extra_system_prompt: str = "",
        use_vision: bool | str = "auto",
        vision_detail_level: str = "auto",
        target_zone_selectors: list[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        llm = get_llm()

        # Resolve vision settings
        env_use_vision = (os.getenv("BROWSER_USE_USE_VISION") or "").strip()
        env_vision_detail = (os.getenv("BROWSER_USE_VISION_DETAIL_LEVEL") or "").strip()
        # Resolve vision to browser-use's expected types (bool | Literal['auto'])
        _raw_vision = (env_use_vision or str(use_vision)).strip().lower()
        resolved_vision: bool | Literal["auto"] = (
            True
            if _raw_vision == "true"
            else False
            if _raw_vision == "false"
            else "auto"
        )
        _raw_detail = (env_vision_detail or vision_detail_level).strip().lower()
        resolved_vision_detail: Literal["auto", "low", "high"] = (
            "low"
            if _raw_detail == "low"
            else "high"
            if _raw_detail == "high"
            else "auto"
        )

        # Build custom tools via browser-use's documented @tools.action() pattern
        from graph_agent.cartography.react_explorer.page_actions import (
            create_explorer_tools,
        )

        tools = create_explorer_tools()

        super().__init__(
            task=(
                "Explore the page: interact with EVERY element (click, fill, select), "
                "submit forms, and map all transitions."
            ),
            llm=llm,
            tools=tools,
            browser_session=browser_session,
            max_actions_per_step=1,
            max_failures=30,
            final_response_after_failure=False,
            use_judge=False,
            enable_planning=False,
            message_compaction=False,
            include_attributes=[
                "title",
                "type",
                "checked",
                "id",
                "name",
                "role",
                "value",
                "placeholder",
                "data-date-format",
                "alt",
                "aria-label",
                "aria-expanded",
                "data-state",
                "aria-checked",
                "aria-valuemin",
                "aria-valuemax",
                "aria-valuenow",
                "aria-placeholder",
                "pattern",
                "min",
                "max",
                "minlength",
                "maxlength",
                "step",
                "accept",
                "multiple",
                "inputmode",
                "required",
                "aria-required",  # patch 3 markers
            ],
            extend_system_message=_build_extend_system_message(extra_system_prompt),
            initial_actions=cast(Any, initial_actions),
            use_vision=resolved_vision,
            vision_detail_level=resolved_vision_detail,
            step_timeout=600,
        )

        # ── Project-specific state ──────────────────────────────────
        self._browser_session = browser_session
        self._explored_indices: set[int] = set()
        self._total_wait_time: float = 0.0
        self._last_url = ""
        self._page_title = ""
        self._state_id = ""
        self._result = CartographyResult()
        self._extra_system_prompt = extra_system_prompt
        self._semantic_conflict_count = 0
        self._target_zone_selectors: list[str] | None = (
            list(target_zone_selectors) if target_zone_selectors else None
        )
        self._should_stop = should_stop
        self._initial_history = initial_history or []

        # Runtime budget
        self._max_runtime_sec = self._resolve_max_runtime_sec()
        self._run_start_time: float = 0.0

        # Intent-level transition accumulation
        self._pending_steps: list[TransitionStep] = []
        self._pending_from_state_id = ""
        self._pending_from_url = ""
        self._pending_from_fp = ""
        self._explore_start_url = ""

        # Pre-step tracking for state change detection (set in on_step_end)
        self._url_before = ""
        self._fp_before = ""
        self._selector_map_before: dict = {}

        # Runtime watchdog (records observations; stop handled by browser-use)
        self.runtime_watchdog = RuntimeWatchdog()

    # ------------------------------------------------------------------
    # Runtime budget
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_max_runtime_sec() -> float:
        raw = (os.getenv("CARTOGRAPHY_AGENT_MAX_RUNTIME_SEC") or "").strip()
        if not raw:
            return 3600.0
        try:
            val = float(raw)
        except ValueError:
            return 3600.0
        return max(30.0, min(4 * 3600.0, val))

    # ------------------------------------------------------------------
    # Fingerprinting utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _url_origin(url: str) -> str:
        from urllib.parse import urlparse

        p = urlparse(url or "")
        return f"{p.scheme}://{p.netloc}"

    def _is_same_origin(self, url: str) -> bool:
        if not self._explore_start_url:
            return True
        return self._url_origin(url) == self._url_origin(self._explore_start_url)

    @staticmethod
    def _compute_dom_fingerprint(dom_text: str) -> str:
        text = dom_text or ""
        text = re.sub(r"\[\d+\]", "[]", text)
        text = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "DATE", text)
        text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?\b", "TIME", text)
        text = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "UUID",
            text,
        )
        text = re.sub(r"\b\d{4,}\b", "NUM", text)
        text = re.sub(r"\b[0-9a-fA-F]{8,}\b", "HEX", text)
        text = re.sub(r"\s+", " ", text).strip()
        return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def _compute_structural_fingerprint(selector_map: dict) -> str:
        from collections import Counter

        counts: Counter[str] = Counter()
        for el in (selector_map or {}).values():
            tag = getattr(el, "tag_name", "") or ""
            role = (getattr(el, "attributes", {}) or {}).get("role", "")
            if tag:
                counts[f"tag:{tag}"] += 1
            if role:
                counts[f"role:{role}"] += 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        raw = "|".join(parts)
        return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]

    def _compute_page_fingerprint(
        self,
        dom_text: str,
        title: str,
        selector_map: dict,
        layout_fingerprint: str | None = None,
    ) -> str:
        dom_fp = self._compute_dom_fingerprint(dom_text)
        struct_fp = self._compute_structural_fingerprint(selector_map)
        title_part = hashlib.md5(
            (title or "").encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        composite = f"{dom_fp}:{struct_fp}:{title_part}"
        include_layout = (
            (os.getenv("CARTOGRAPHY_LAYOUT_INCLUDE_IN_PAGE_FP") or "").strip().lower()
        )
        if layout_fingerprint and include_layout in {"1", "true", "yes", "on"}:
            composite = f"{composite}:{layout_fingerprint}"
        return hashlib.md5(
            composite.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:20]

    # ------------------------------------------------------------------
    # Action registration
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # on_step_start callback (runtime budget + patch 3 + health check)
    # ------------------------------------------------------------------

    @staticmethod
    async def _on_step_start(agent: "ExplorerAgent") -> None:
        # --- Runtime budget check (hard stop) ---
        elapsed = time.monotonic() - agent._run_start_time
        if elapsed >= agent._max_runtime_sec:
            logger.warning(
                "Stopping agent: runtime budget %.1fs exhausted", agent._max_runtime_sec
            )
            agent.state.stopped = True
            return

        # --- External should_stop signal ---
        if agent._should_stop and agent._should_stop():
            logger.info("Stopping agent: external shutdown request")
            agent.state.stopped = True
            return

        # --- DOM patch 3: required fields ---
        try:
            page = await agent.browser_session.get_current_page()
            if page is not None:
                await page.evaluate(_PATCH_REQUIRED_FIELDS_JS)
        except Exception:
            pass

        # --- Browser health check every 10 steps ---
        if agent.state.n_steps % 10 == 0:
            try:
                page = await agent.browser_session.get_current_page()
                if page is not None:
                    await page.evaluate("() => 1")
            except Exception as e:
                logger.warning(
                    "Browser health check failed at step %d: %s",
                    agent.state.n_steps,
                    e,
                )

    # ------------------------------------------------------------------
    # on_step_end callback (fingerprint + state change + transition)
    # ------------------------------------------------------------------

    @staticmethod
    async def _on_step_end(agent: "ExplorerAgent") -> None:
        model_output = agent.state.last_model_output
        if model_output is None:
            return

        # 1. Get current browser state
        bss = await agent.browser_session.get_browser_state_summary(
            include_screenshot=False, include_recent_events=False
        )
        dom_text = bss.dom_state.llm_representation()
        selector_map: dict = getattr(bss.dom_state, "selector_map", {}) or {}
        title = bss.title or ""
        url_after = bss.url or ""

        # 2. Compute fingerprint
        fp_after = agent._compute_page_fingerprint(dom_text, title, selector_map)

        # 3. Detect state change
        changed = (url_after != agent._url_before) or (fp_after != agent._fp_before)

        # 4. Extract action info from model output
        actions = model_output.action
        if not actions:
            return

        action = actions[0]
        params = action.model_dump(exclude_unset=True, exclude_none=True)
        action_names = list(params.keys())
        action_type = action_names[0] if action_names else ""

        if action_type == "done":
            return

        # 5. Build transition step if state changed and action is relevant
        if changed and action_type in ("click", "input", "select_dropdown"):
            # Set up pending from-state (only on first step of a new transition)
            from_state_id = _build_state_identity(
                agent._url_before,
                _extract_spa_route(agent._url_before),
                agent._fp_before or agent._compute_dom_fingerprint(agent._url_before),
            )
            if not agent._pending_steps:
                agent._pending_from_state_id = from_state_id
                agent._pending_from_url = agent._url_before
                agent._pending_from_fp = agent._fp_before or ""

            ts = agent._build_transition_step(
                step=agent.state.n_steps,
                action_type=action_type,
                action=action,
                model_output=model_output,
                selector_map=selector_map,
            )
            agent._pending_steps.append(ts)

        # 6. Flush pending transition if needed
        if changed and _should_commit_transition(
            agent._url_before, url_after, action_type
        ):
            await agent._flush_pending_transition()

        # 7. Update tracking for next step comparison
        agent._url_before = url_after
        agent._fp_before = fp_after
        agent._selector_map_before = selector_map

        # 8. Update tracking state
        if url_after != agent._last_url:
            agent._last_url = url_after
        if title:
            agent._page_title = title

        # 9. Reset wait time tracking
        if action_type != "wait":
            agent._total_wait_time = 0.0

    # ------------------------------------------------------------------
    # Transition building
    # ------------------------------------------------------------------

    def _build_transition_step(
        self,
        step: int,
        action_type: str,
        action: Any,  # browser-use ActionModel
        model_output: Any,  # browser-use AgentOutput
        selector_map: dict,
    ) -> TransitionStep:
        """Build a TransitionStep from browser-use model output.

        Uses selector_map from browser-state (EnhancedDOMTreeNode) instead
        of the old PageController.selector_map — they share the same source
        and have identical fields (tag_name, attributes, css_selector, xpath,
        node_value).
        """
        act_type = {
            "click": ActionType.CLICK,
            "input": ActionType.FILL,
            "select_dropdown": ActionType.SELECT,
        }.get(action_type, ActionType.CLICK)

        idx = action.get_index() if hasattr(action, "get_index") else None
        selector = f"[{idx}]" if idx is not None else "[?]"
        selector_chain = [selector]
        element_snapshot_json: str | None = None

        if idx is not None and selector_map:
            node = selector_map.get(idx)
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

                is_required = (
                    attrs.get("required") is not None
                    or attrs.get("aria-required") == "true"
                    or bool(
                        re.search(
                            r"\b(required|is-required|mandatory|must)\b",
                            str(attrs.get("class", "")),
                        )
                    )
                )

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
                        "required": is_required,
                        "attributes": attrs,
                        "frame_path": [],
                    },
                    ensure_ascii=False,
                )

        param_name = _infer_param_name_from_snapshot(element_snapshot_json)
        thought_text = getattr(model_output, "next_goal", "") or ""
        text_value = None
        try:
            params = action.model_dump(exclude_unset=True, exclude_none=True)
            if action_type in ("input", "select_dropdown"):
                text_value = params.get("text")
        except Exception:
            pass

        return TransitionStep(
            action=act_type,
            selector=selector,
            selector_chain=selector_chain,
            param_name=param_name,
            action_value=text_value,
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

        # Get current browser state for after-state fingerprint
        try:
            bss = await self.browser_session.get_browser_state_summary(
                include_screenshot=False, include_recent_events=False
            )
            dom_text_after = bss.dom_state.llm_representation()
            selector_map_after: dict = getattr(bss.dom_state, "selector_map", {}) or {}
            title_after = bss.title or ""
            fp_after = self._compute_page_fingerprint(
                dom_text_after, title_after, selector_map_after
            )
        except Exception:
            fp_after = ""
            dom_text_after = ""
            title_after = ""
            selector_map_after = {}

        # Get current URL
        current_url = ""
        try:
            page = await self.browser_session.get_current_page()
            if page is not None:
                current_url = await page.get_url() or ""
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
        resolved_to_title = (title_after or "").strip() or self._page_title or ""
        to_state = State(
            id=to_state_id,
            url=current_url or self._pending_from_url,
            title=resolved_to_title,
            spa_route=_extract_spa_route(current_url or self._pending_from_url),
            fingerprint=fp_after,
            view_fingerprint=fp_after,
            data_signature=_build_data_signature(
                current_url or self._pending_from_url,
                dom_text=dom_text_after or "",
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

        # Run semantic inference
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
            _transition_dedupe_key(item) for item in self._result.transitions[-30:]
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

    # ------------------------------------------------------------------
    # History extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_history(agent_history: AgentHistoryList) -> list[dict[str, Any]]:
        """Convert browser-use AgentHistoryList to project's history format."""
        history: list[dict[str, Any]] = []
        for h in agent_history.history:
            if h.model_output and h.model_output.action:
                action = h.model_output.action[0]
                dumped = action.model_dump(exclude_unset=True, exclude_none=True)
                action_names = list(dumped.keys())
                action_name = action_names[0] if action_names else ""
                result_text = h.result[0].extracted_content if h.result else ""
                history.append(
                    {
                        "evaluation_previous_goal": (
                            h.model_output.evaluation_previous_goal or ""
                        ),
                        "memory": h.model_output.memory or "",
                        "next_goal": h.model_output.next_goal or "",
                        "action_name": action_name,
                        "action_result": str(result_text or ""),
                    }
                )
        return history

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
        1. ``explore_page(session, state_id, page_title)`` — mapping pipeline.
        2. ``explore_page(start_url=url, max_steps=n)`` — maximal explorer.
        """
        bs = session or self.browser_session

        if start_url:
            if bs is None:
                raise RuntimeError(
                    "ReActExplorer requires a BrowserSession when start_url is given"
                )
            try:
                await bs.navigate_to(start_url)
            except Exception as e:
                logger.warning("Failed to navigate to start_url %s: %s", start_url, e)
            self._state_id = _build_state_identity(
                start_url, _extract_spa_route(start_url), ""
            )
        else:
            if bs is None:
                raise RuntimeError("ReActExplorer requires a BrowserSession")
            self._state_id = state_id

        # Initialize per-run state
        self._page_title = page_title
        self._result = CartographyResult()
        self._explored_indices.clear()
        self._total_wait_time = 0.0
        self._last_url = ""
        self._url_before = ""
        self._fp_before = ""
        self._selector_map_before = {}
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

        # Token tracking
        from graph_agent.lib.token_tracker import reset_global_tracker

        reset_global_tracker()
        self._run_start_time = time.monotonic()

        # Run browser-use Agent loop.  max_steps is effectively unlimited;
        # runtime budget (checked in _on_step_start) is the real governor.
        try:
            agent_history = await self.run(
                max_steps=_UNLIMITED_STEPS,
                on_step_start=ExplorerAgent._on_step_start,  # type: ignore[arg-type]
                on_step_end=ExplorerAgent._on_step_end,  # type: ignore[arg-type]
            )
            self._result.history = self._extract_history(agent_history)
        finally:
            if self._pending_steps:
                await self._flush_pending_transition()

        from graph_agent.lib.token_tracker import get_global_tracker

        get_global_tracker().log_summary()

        logger.info(
            "ReAct exploration complete: %d steps, %d transitions, %d checkpoints",
            len(self._result.history),
            len(self._result.transitions),
            len(self._result.checkpoints),
        )

        return self._result

"""Base ReAct agent — observe → think → act loop.

Subclasses override hook methods to customize behavior:
- _build_system_prompt() → str
- _build_user_prompt(...) → str
- _execute_action(action_type, params) → str
- _on_before_step(step) → None
- _on_after_step(step, action_type, result, changed) → None
- _on_state_changed(...) → None
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections import deque
from typing import TYPE_CHECKING, Awaitable, Callable

from pydantic import BaseModel, create_model

from browser_use.agent.views import ActionModel
from browser_use.browser.session import BrowserSession as Browser
from browser_use.llm.base import BaseChatModel
from browser_use.tools.service import Tools

from graph_agent.cartography.react_schema import AgentAction, AgentOutput
from graph_agent.llm.utils import ainvoke_structured
from graph_agent.models import ActionType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from playwright.async_api import Page, Request


def _get_spa_route(url: str) -> str:
    """Extract SPA route from URL hash or path."""
    from urllib.parse import urlparse

    parsed = urlparse(url or "")
    return parsed.fragment or parsed.path or ""


class AgentResult:
    """Base result container for agent runs.

    Subclasses can extend this with additional fields.
    """

    def __init__(self) -> None:
        self.states: list[object] = []
        self.transitions: list[object] = []
        self.history: list[dict[str, object]] = []


class BaseAgent:
    """Base ReAct agent with observe → think → act loop.

    Provides the core infrastructure:
    - Browser state observation and fingerprinting
    - LLM structured output via AgentOutput schema
    - Action dispatch through browser-use Tools
    - History tracking and step callbacks
    - initial_actions execution before the LLM loop

    Subclasses override hooks to customize:
    - Prompt construction (_build_system_prompt, _build_user_prompt)
    - Action execution (_execute_action)
    - Lifecycle hooks (_on_before_step, _on_after_step, _on_state_changed)
    """

    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser: Browser,
        max_steps: int = 100,
        total_max_steps: int | None = None,
        step_callback: Callable[[dict[str, object]], object | Awaitable[object]] | None = None,
        should_stop: Callable[[], bool] | None = None,
        initial_history: list[dict[str, object]] | None = None,
        initial_actions: list[dict[str, object]] | None = None,
        start_step: int = 0,
        start_url: str = "",
        use_vision: bool | str = "auto",
        vision_detail_level: str = "auto",
    ):
        self.task = task
        self.llm = llm
        self.browser = browser
        self.tools: Tools = Tools()
        self.max_steps = max_steps
        self.total_max_steps = total_max_steps or max_steps
        self.step_callback = step_callback
        self.should_stop = should_stop
        self.initial_history = initial_history or []
        self.initial_actions = initial_actions or []
        self.start_step = start_step
        self._start_url = start_url
        self._start_origin = self._url_origin(start_url) if start_url else ""
        self.session_id = "base-agent"
        self._last_history: list[dict[str, object]] = []

        # Page load tracking
        self._page_load_issue_note: str | None = None
        self._request_failure_log: list[dict[str, object]] = []
        self._use_vision = self._resolve_use_vision_mode(use_vision)
        self._vision_detail_level = self._resolve_vision_detail_level(vision_detail_level)

        # Supported actions from browser-use registry
        self._supported_actions: set[str] = {
            "click",
            "input",
            "select_dropdown",
            "scroll",
            "wait",
            "go_back",
            "done",
        }
        # Keep screenshot action available in auto/true mode.
        if self._use_vision != "false":
            self._supported_actions.add("screenshot")
        self._dynamic_action_model = self._build_dynamic_action_model()

    @staticmethod
    def _resolve_use_vision_mode(value: bool | str) -> str:
        """Resolve use_vision mode: auto | true | false."""
        raw = str(value).strip().lower() if not isinstance(value, bool) else ("true" if value else "false")
        if raw in {"true", "false", "auto"}:
            return raw
        env_raw = (os.getenv("BROWSER_USE_USE_VISION") or "").strip().lower()
        if env_raw in {"true", "false", "auto"}:
            return env_raw
        return "auto"

    @staticmethod
    def _resolve_vision_detail_level(value: str) -> str:
        """Resolve vision detail level: auto | low | high."""
        raw = (value or "").strip().lower()
        if raw in {"auto", "low", "high"}:
            return raw
        env_raw = (os.getenv("BROWSER_USE_VISION_DETAIL_LEVEL") or "").strip().lower()
        if env_raw in {"auto", "low", "high"}:
            return env_raw
        return "auto"

    # ------------------------------------------------------------------
    # Hook methods for subclasses to override
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build system prompt with available actions and output schema.

        Subclasses override this to add domain-specific instructions.
        """
        action_descriptions = []
        for name in sorted(self._supported_actions):
            if name not in self.tools.registry.registry.actions:
                continue
            action = self.tools.registry.registry.actions[name]
            action_descriptions.append(f"- {name}: {action.description}")

        actions_text = "\n".join(action_descriptions)

        from graph_agent.llm.utils import get_format_instructions

        format_instr = get_format_instructions(AgentOutput)

        return f"""\
You are an AI agent that explores a web application.

<available_actions>
{actions_text}
</available_actions>

{format_instr}
"""

    def _build_user_prompt(
        self,
        dom_text: str,
        history: list[dict],
        step: int,
        current_url: str,
        page_title: str = "",
    ) -> str:
        """Build user prompt with current browser state and history.

        Subclasses override this to inject additional context.
        """
        remaining = self.total_max_steps - step
        history_entries = []
        for i, h in enumerate(history[-10:], start=max(0, len(history) - 10)):
            history_entries.append(
                f"Step {h.get('step', i)}: {h.get('action', 'unknown')} -> {h.get('result', '')}"
            )
        history_text = "\n".join(history_entries) if history_entries else "No previous actions."

        note_section = ""
        if self._page_load_issue_note:
            note_section = f"\n<system_note>\n{self._page_load_issue_note}\n</system_note>\n"

        return f"""\
<task>
{self.task}
</task>{note_section}

<current_state>
URL: {current_url}
Title: {page_title}
Step: {step + 1} / {self.total_max_steps}
Remaining steps: {remaining}
</current_state>

<browser_state>
{dom_text}
</browser_state>

<history>
{history_text}
</history>
"""

    async def _execute_action(self, action_type: str, params: dict[str, object]) -> str:
        """Execute a single action and return a result description.

        Subclasses override this to handle custom actions.
        Default implementation delegates to browser-use Tools.

        Returns:
            Human-readable result string for history tracking.
        """
        if action_type not in self._supported_actions:
            return f"Unsupported action type: {action_type}"

        # Convert params to AgentAction then to browser-use ActionModel
        action_cls = self._dynamic_action_model
        action_model = action_cls(**{action_type: params})
        action_result = await self.tools.act(
            action_model, browser_session=self.browser
        )
        return action_result.extracted_content or str(action_result)

    async def _on_before_step(self, step: int) -> None:
        """Hook called at the start of each step (after observe).

        Subclasses can use this for waypoint checks, scope updates, etc.
        """
        pass

    async def _on_after_step(
        self,
        step: int,
        action_type: str,
        result_text: str,
        changed: bool,
        url_before: str,
        url_after: str,
    ) -> None:
        """Hook called after action execution and transition recording.

        Subclasses can use this for periodic tasks (e.g. rediscovery).
        """
        pass

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
        """Hook called when a state change is detected (URL or fingerprint changed).

        Subclasses override this to record states and transitions.
        """
        pass

    def _make_history_entry(
        self,
        step: int,
        action_type: str,
        result_text: str,
        output: AgentOutput | None = None,
    ) -> dict[str, object]:
        """Build a history entry dict after each step.

        Subclasses override this to capture additional fields from ``output``
        (e.g. evaluation_previous_goal, memory, next_goal).
        """
        return {"step": step, "action": action_type, "result": result_text}

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def run(self) -> AgentResult:
        """Execute the ReAct loop.

        Returns an AgentResult. Subclasses can override to return a richer result type.
        """
        result = AgentResult()
        history: deque[dict] = deque(self.initial_history, maxlen=50)

        url_before = ""
        fp_before = ""
        title_before = ""
        selector_map_before: dict = {}

        # Setup network request failure listener
        await self._attach_request_failure_listener()

        # Execute initial_actions before the LLM loop so they appear in history.
        for ia_step, ia in enumerate(self.initial_actions):
            action_type = ia.get("action_type", "")
            result_text = await self._execute_initial_action(action_type, ia)
            history.append({
                "step": f"init-{ia_step}",
                "action": action_type,
                "result": result_text,
            })
            logger.info(
                "[INITIAL_ACTION] step=init-%d action=%s result=%s",
                ia_step,
                action_type,
                result_text[:80],
            )

        for step in range(self.start_step, self.start_step + self.max_steps):
            if self.should_stop and self.should_stop():
                logger.info("Stopping agent loop due to shutdown request")
                break

            # 1. Observe
            try:
                dom_text, page_title, selector_map = await self._get_browser_snapshot()
                current_url = await self.browser.get_current_page_url() or ""
            except Exception as e:
                logger.warning("Failed to get browser state at step %d: %s", step, e)
                break

            # Detect page load failure (empty/minimal DOM)
            dom_len = len(dom_text or "")
            if dom_len < 50:
                logger.warning(
                    "[SKIP] Page appears not loaded (DOM length=%d, url=%s).",
                    dom_len,
                    current_url,
                )
                self._page_load_issue_note = (
                    "The previous action resulted in a page that appears not fully loaded. "
                    "Try clicking a different tab, menu item, or interactive element instead."
                )
                history.append({
                    "step": step,
                    "action": "load_issue",
                    "result": f"Page not loaded (DOM={dom_len}), trying alternative action.",
                })
                continue

            url_before = current_url or url_before
            title_before = page_title or title_before
            selector_map_before = selector_map or selector_map_before
            fp_before = self._compute_page_fingerprint(dom_text, page_title, selector_map)

            # Subclass hook: before step
            await self._on_before_step(step)

            # 2. Think
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                dom_text, list(history), step, url_before, page_title
            )

            try:
                output: AgentOutput = await ainvoke_structured(
                    self.llm,
                    system_prompt,
                    user_prompt,
                    AgentOutput,
                    max_retries=2,
                )
            except Exception as e:
                logger.warning("LLM structured output failed at step %d: %s", step, e)
                history.append({
                    "step": step,
                    "action": "error",
                    "result": f"LLM error: {e}",
                })
                continue

            action_type = output.action.action_type
            logger.info(
                "  [Agent Step %d] action=%s | goal=%s",
                step,
                action_type,
                output.next_goal[:60],
            )

            # 3. Act
            if action_type == "done":
                history.append(
                    self._make_history_entry(
                        step, "done",
                        getattr(output.action, "text", "completed"),
                        output,
                    )
                )
                break

            params = output.action.model_dump(exclude={"action_type"})
            result_text = await self._execute_action(action_type, params)

            # Observation feedback: truncate but keep enough context to see
            # CDP error messages like "element index X not found" that the
            # LLM uses to decide whether to retry.
            _result_snippet = (result_text or "").strip().replace("\n", " ")
            if len(_result_snippet) > 200:
                _result_snippet = _result_snippet[:200] + "..."
            logger.info(
                "  [Agent Step %d] result=%s",
                step,
                _result_snippet or "<empty>",
            )

            history.append(
                self._make_history_entry(step, action_type, result_text, output)
            )

            if self.step_callback:
                try:
                    cb_result = self.step_callback({
                        "step": step,
                        "action_type": action_type,
                        "output": output,
                        "result": result_text,
                        "url": url_before,
                    })
                    if cb_result and hasattr(cb_result, "__await__"):
                        await cb_result
                except Exception as e:
                    logger.debug("Step callback error: %s", e)

            # 4. Record transition if state changed
            try:
                page = await self.browser.get_current_page()
            except Exception:
                page = None

            if page:
                stable_result = await self._wait_for_page_stable(page)
                if stable_result.get("has_cors_failures"):
                    failed_count = len(stable_result.get("failed_requests", []))
                    logger.warning(
                        "[CORS] Page has %d CORS failures; trying alternative action.",
                        failed_count,
                    )
                    self._page_load_issue_note = (
                        f"The previous action caused {failed_count} network request failure(s) "
                        "(likely CORS / cross-origin blocking). Try clicking a different element."
                    )
                    history.append({
                        "step": step,
                        "action": "cors_failure",
                        "result": f"{failed_count} request failures detected; trying alternative.",
                    })
                    continue
                self._page_load_issue_note = None
            else:
                await asyncio.sleep(0.5)

            # Check for state change
            try:
                dom_text_after, title_after, selector_map_after = await self._get_browser_snapshot()
                url_after = await self.browser.get_current_page_url() or ""
                fp_after = self._compute_page_fingerprint(
                    dom_text_after, title_after, selector_map_after
                )
            except Exception:
                url_after = url_before
                fp_after = fp_before
                title_after = title_before
                selector_map_after = selector_map_before

            changed = (url_after != url_before) or (fp_after != fp_before)

            if changed:
                await self._on_state_changed(
                    step, url_before, url_after, action_type, output,
                    fp_before=fp_before, fp_after=fp_after,
                )

            await self._on_after_step(
                step, action_type, result_text, changed, url_before, url_after
            )

        self._last_history = list(history)
        result.history = self._last_history
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_initial_action(self, action_type: str, action: dict[str, object]) -> str:
        """Execute a single initial_action."""
        return await self._execute_action(action_type, action)

    async def _attach_request_failure_listener(self) -> None:
        """Attach Playwright requestfailed listener to detect CORS issues."""
        try:
            pw_page = await self.browser.get_current_page()
        except Exception:
            pw_page = None
        if pw_page and hasattr(pw_page, "on"):
            def _on_request_failed(request: "Request") -> None:
                failure = getattr(request, "failure", None)
                error_text = str(failure.get("errorText", "")) if failure else ""
                is_cors = any(
                    kw in error_text.lower()
                    for kw in ("cors", "cross-origin", "access-control")
                )
                self._request_failure_log.append({
                    "url": request.url,
                    "method": getattr(request, "method", "GET"),
                    "error": error_text,
                    "is_cors": is_cors,
                })

            pw_page.on("requestfailed", _on_request_failed)

    def _build_dynamic_action_model(self) -> type[ActionModel]:
        """Build a browser-use compatible ActionModel with supported actions."""
        fields: dict[str, tuple[type[BaseModel] | None, None]] = {}
        for name in sorted(self._supported_actions):
            if name not in self.tools.registry.registry.actions:
                continue
            param_model = self.tools.registry.registry.actions[name].param_model
            fields[name] = (param_model | None, None)
        return create_model("DynamicAction", __base__=ActionModel, **fields)

    async def _get_browser_snapshot(self) -> tuple[str, str, dict]:
        """Get DOM text, page title, and selector map."""
        include_screenshot = self._use_vision == "true"
        browser_summary = await self.browser.get_browser_state_summary(
            include_screenshot=include_screenshot, include_recent_events=False
        )
        dom_text = browser_summary.dom_state.llm_representation()
        title = browser_summary.title or ""
        selector_map = getattr(browser_summary.dom_state, "selector_map", {}) or {}
        return dom_text, title, selector_map

    async def _wait_for_page_stable(
        self, page: "Page", timeout_ms: int = 5000
    ) -> dict[str, object]:
        """Wait for page to stabilize after an action."""
        result: dict[str, object] = {
            "stable": True,
            "has_cors_failures": False,
            "failed_requests": [],
        }

        # Check for accumulated request failures BEFORE waiting
        if self._request_failure_log:
            cors_failures = [
                r for r in self._request_failure_log if r.get("is_cors")
            ]
            if len(cors_failures) >= 2:
                result["stable"] = False
                result["has_cors_failures"] = True
                result["failed_requests"] = list(self._request_failure_log)
                self._request_failure_log.clear()
                return result

        # 1. Try networkidle first
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass

        # 2. Wait for DOM to stop changing significantly
        dom_text, title, selector_map = await self._get_browser_snapshot()
        start_fp = self._compute_page_fingerprint(dom_text, title, selector_map)
        stable_count = 0
        for _ in range(10):
            await asyncio.sleep(0.25)
            dom_text, title, selector_map = await self._get_browser_snapshot()
            current_fp = self._compute_page_fingerprint(dom_text, title, selector_map)
            if current_fp == start_fp:
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                start_fp = current_fp
                stable_count = 0

        # After waiting, check again for failures
        if self._request_failure_log:
            result["failed_requests"] = list(self._request_failure_log)
            cors_failures = [
                r for r in self._request_failure_log if r.get("is_cors")
            ]
            if cors_failures:
                result["has_cors_failures"] = True
            self._request_failure_log.clear()

        return result

    # ------------------------------------------------------------------
    # Fingerprinting utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _url_origin(url: str) -> str:
        """Extract scheme://host from a URL."""
        from urllib.parse import urlparse
        p = urlparse(url or "")
        return f"{p.scheme}://{p.netloc}"

    def _is_same_origin(self, url: str) -> bool:
        """Check if url shares the same origin as the start URL."""
        if not self._start_origin:
            return True
        return self._url_origin(url) == self._start_origin

    @staticmethod
    def _compute_dom_fingerprint(dom_text: str) -> str:
        """Compute a stable fingerprint from DOM representation."""
        text = dom_text or ""
        text = re.sub(r"\[\d+\]", "[]", text)
        text = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "DATE", text)
        text = re.sub(
            r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?\b", "TIME", text
        )
        text = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "UUID",
            text,
        )
        text = re.sub(r"\b\d{4,}\b", "NUM", text)
        text = re.sub(r"\b[0-9a-fA-F]{8,}\b", "HEX", text)
        text = re.sub(r"\s+", " ", text).strip()
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _compute_structural_fingerprint(selector_map: dict) -> str:
        """Compute a structural fingerprint from the interactive element map."""
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
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def _compute_page_fingerprint(
        self,
        dom_text: str,
        title: str,
        selector_map: dict,
        layout_fingerprint: str | None = None,
    ) -> str:
        """Composite fingerprint combining DOM text, title, and structure."""
        dom_fp = self._compute_dom_fingerprint(dom_text)
        struct_fp = self._compute_structural_fingerprint(selector_map)
        title_part = hashlib.md5((title or "").encode("utf-8")).hexdigest()[:8]
        composite = f"{dom_fp}:{struct_fp}:{title_part}"
        include_layout = (os.getenv("CARTOGRAPHY_LAYOUT_INCLUDE_IN_PAGE_FP") or "").strip().lower()
        if layout_fingerprint and include_layout in {"1", "true", "yes", "on"}:
            composite = f"{composite}:{layout_fingerprint}"
        return hashlib.md5(composite.encode("utf-8")).hexdigest()[:20]

    async def _detect_modal(self, page) -> bool:
        """Detect if a modal/dialog is currently visible."""
        try:
            return await page.evaluate(
                """
                () => {
                    const selectors = [
                        '.ant-modal-wrap:not(.ant-modal-hidden)',
                        '.el-dialog__wrapper:not(.is-hidden)',
                        '[role="dialog"]:not([aria-hidden="true"])',
                        '.modal.show',
                        '.drawer.open'
                    ];
                    return selectors.some(sel => document.querySelector(sel) !== null);
                }
                """
            )
        except Exception:
            return False

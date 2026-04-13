"""ReAct-based page explorer — LLM-guided observe→think→act loop.

Aligned with page-agent's PageAgentCore architecture:
- Structured output via Pydantic schema (equivalent to page-agent's Zod macro tool)
- System observations (wait time, URL changes, remaining steps)
- LLM retry with configurable attempts

All browser interactions go through browser-use's BrowserSession / Page / Element APIs.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession

from graph_agent.cartography.react_prompts import (
    build_system_prompt,
    build_user_prompt,
)
from graph_agent.cartography.react_schema import (
    AgentOutput,
    agent_output_to_dict,
)

# Import tools to ensure registration
try:
    from graph_agent import tools  # noqa: F401 - registers tools via @action
except ImportError:
    pass
from graph_agent.cartography.snapshot import capture_dom_fingerprint
from graph_agent.graph.merger import CartographyResult
from graph_agent.lib.page_controller import PageController
from graph_agent.intent.parser import infer_intent_progressive, distill_ui_thought
from graph_agent.llm import ainvoke_structured, get_llm
from graph_agent.lib.token_tracker import get_global_tracker
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
_LLM_TIMEOUT_MS = 60_000  # 增加到60秒，适应长上下文
_MAX_HISTORY_LENGTH = 20  # 限制历史记录长度，防止内存泄漏


class ReActExplorer:
    """LLM-driven ReAct explorer for a single page/zone.

    Architecture mirrors page-agent PageAgentCore:
    - observe (getBrowserState) → think (LLM with Pydantic schema) → act → loop
    - Structured output via ``AgentOutput`` Pydantic schema (≈ Zod macro tool)
    - System observations injected before each step
    - LLM retry on failure
    """

    def __init__(
        self,
        max_steps: int = _DEFAULT_MAX_STEPS,
        browser_session: "BrowserSession | None" = None,
    ):
        self._max_steps = max_steps
        self._llm = get_llm()
        self._browser_session = browser_session

    async def _health_check(self, page) -> bool:
        """Check if browser page is still healthy."""
        try:
            await page.evaluate("() => 1")
            return True
        except Exception as e:
            logger.warning("Browser health check failed: %s", e)
            return False

    async def explore_page(
        self,
        session: "BrowserSession",
        state_id: str,
        page_title: str = "",
    ) -> CartographyResult:
        bs = session or self._browser_session
        if bs is None:
            raise RuntimeError("ReActExplorer requires a BrowserSession")

        controller = PageController(bs)
        result = CartographyResult()
        
        # Reset global token tracker for this exploration session
        from graph_agent.lib.token_tracker import reset_global_tracker
        reset_global_tracker()
        token_tracker = get_global_tracker()

        try:
            bu_page = await bs.must_get_current_page()
            history: deque[dict] = deque(maxlen=_MAX_HISTORY_LENGTH)
            explored_indices: set[int] = set()
            observations: list[str] = []
            total_wait_time = 0.0
            last_url = ""
            last_checkpoint_time = time.monotonic()
            system_prompt = build_system_prompt(max_steps=self._max_steps)

            for step in range(self._max_steps):
                # Health check every 10 steps
                if step % 10 == 0:
                    if not await self._health_check(bu_page):
                        logger.error("Browser health check failed at step %d, stopping", step)
                        break

                # Observe
                try:
                    browser_state = await controller.get_browser_state()
                except Exception as e:
                    logger.warning("Failed to get browser state at step %d: %s", step, e)
                    break

                fp_before = await capture_dom_fingerprint(bu_page)
                url_before = await bu_page.get_url()
                observations.clear()

                if total_wait_time >= 3:
                    observations.append(
                        f"You have waited {total_wait_time:.0f} seconds accumulatively. "
                        "DO NOT wait any longer unless you have a good reason."
                    )

                if url_before != last_url:
                    if last_url:
                        observations.append(f"Page navigated to → {url_before}")
                    last_url = url_before
                    await asyncio.sleep(0.5)

                remaining = self._max_steps - step
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

                user_prompt = build_user_prompt(
                    browser_state_text=(
                        f"{browser_state.header}\n{browser_state.content}\n{browser_state.footer}"
                    ),
                    history=history,
                    explored_indices=explored_indices,
                    step=step,
                    max_steps=self._max_steps,
                    page_title=page_title,
                    observations=observations,
                )

                # Think
                parsed = await self._invoke_llm_with_retry(system_prompt, user_prompt)

                if parsed is None:
                    history.append({
                        "evaluation_previous_goal": "LLM failed after retries",
                        "memory": "",
                        "next_goal": "",
                        "action_name": "error",
                        "action_result": "LLM call failed after all retries",
                    })
                    continue

                action = parsed.get("action", {})
                action_name = next(iter(action), "done") if isinstance(action, dict) else "done"
                action_params = action.get(action_name, {}) if isinstance(action, dict) else {}
                if not isinstance(action_params, dict):
                    action_params = {}

                logger.info(
                    "  [Step %d] %s | goal: %s",
                    step,
                    action_name,
                    (parsed.get("next_goal") or "")[:60],
                )

                # Act
                if action_name == "done":
                    history.append({
                        **{k: parsed.get(k, "") for k in ("evaluation_previous_goal", "memory", "next_goal")},
                        "action_name": "done",
                        "action_result": action_params.get("text", "completed"),
                    })
                    break

                action_result = await self._execute_action(
                    action_name, action_params, controller, bu_page
                )

                if action_name == "wait":
                    total_wait_time += action_params.get("seconds", 1)
                else:
                    total_wait_time = 0

                if action_name in ("click_element_by_index", "input_text", "select_dropdown_option"):
                    idx = action_params.get("index")
                    if idx is not None:
                        explored_indices.add(idx)

                # Record transition
                await asyncio.sleep(0.5)
                fp_after = await capture_dom_fingerprint(bu_page)
                url_after = await bu_page.get_url()
                changed = fp_after != fp_before or url_after != url_before

                if changed and action_name in (
                    "click_element_by_index", "input_text", "select_dropdown_option"
                ):
                    label = f"react-{action_name}-step{step}"
                    from_state = State(
                        id=state_id,
                        url=url_before,
                        title=page_title,
                        fingerprint=fp_before,
                    )
                    to_state_id = f"{state_id}:react-{step}"
                    to_state = State(
                        id=to_state_id,
                        url=url_after,
                        fingerprint=fp_after,
                        title=f"{page_title} after {action_name}",
                    )
                    act_type = {
                        "click_element_by_index": ActionType.CLICK,
                        "input_text": ActionType.FILL,
                        "select_dropdown_option": ActionType.SELECT,
                    }.get(action_name, ActionType.CLICK)

                    thought_text = parsed.get("next_goal", "")
                    intent = None
                    try:
                        distilled_thought = await distill_ui_thought(
                            thought_text, act_type, f"[{action_params.get('index', '?')}]", "", ""
                        )
                        neighbor_steps = list(history)[-3:] if history else None
                        page_signals = {"title": page_title, "url": url_before}
                        intent, _reason, _level = await infer_intent_progressive(
                            action=act_type,
                            selector=f"[{action_params.get('index', '?')}]",
                            source_url=url_before,
                            target_url=url_after,
                            param_name=None,
                            thought_text=distilled_thought,
                            neighbor_steps=neighbor_steps,
                            page_signals=page_signals,
                        )
                        if intent:
                            logger.debug("    → Intent inferred: %s", intent.key)
                    except Exception as e:
                        logger.debug("    → Intent inference skipped: %s", e)

                    transition = Transition(
                        id=f"t:{state_id}:{label}",
                        selector=f"[{action_params.get('index', '?')}]",
                        action=act_type,
                        thought=thought_text,
                        intent=intent,
                        from_state_id=state_id,
                        to_state_id=to_state_id,
                        step_index=step,
                    )
                    cp = Checkpoint(
                        id=f"cp:{transition.id}:after",
                        layer=CheckpointLayer.STRUCTURAL,
                        timing=CheckpointTiming.AFTER,
                        expect=CheckpointExpect.SHOULD_PASS,
                        severity=Severity.MAJOR,
                        rule_type="url_changed" if url_after != url_before else "dom_changed",
                        rule=json.dumps({
                            "expected": url_after if url_after != url_before else "dom_fingerprint_changed"
                        }),
                        description=parsed.get("next_goal", f"After {action_name}, page should change"),
                    )
                    result.states.extend([from_state, to_state])
                    result.transitions.append(transition)
                    result.checkpoints.append(cp)
                    result.checkpoint_transition_map[cp.id] = transition.id
                    logger.info("    → Transition recorded: %s", label)

                    if url_after != url_before:
                        try:
                            await bu_page.go_back()
                        except Exception:
                            pass
                        await asyncio.sleep(0.5)

                history.append({
                    **{k: parsed.get(k, "") for k in ("evaluation_previous_goal", "memory", "next_goal")},
                    "action_name": action_name,
                    "action_result": action_result,
                })

                # Checkpoint every 30 minutes
                if time.monotonic() - last_checkpoint_time > 1800:
                    logger.info("  [Checkpoint] Step %d, transitions: %d", step, len(result.transitions))
                    last_checkpoint_time = time.monotonic()

        except Exception as e:
            logger.error("  ReAct exploration failed: %s", e, exc_info=True)
            raise
        finally:
            logger.info(
                "  ReAct exploration complete: %d steps, %d transitions, %d checkpoints",
                len(history),
                len(result.transitions),
                len(result.checkpoints),
            )
            # Log token usage summary
            token_tracker.log_summary()
            try:
                controller.dispose()
            except Exception as e:
                logger.warning("  Failed to dispose controller: %s", e)

        return result

    async def _invoke_llm_with_retry(
        self, system_prompt: str, user_prompt: str
    ) -> dict | None:
        """Invoke the LLM with Pydantic schema validation and retry."""
        for attempt in range(1, _LLM_MAX_RETRIES + 1):
            try:
                output: AgentOutput = await ainvoke_structured(
                    self._llm,
                    system_prompt,
                    user_prompt,
                    AgentOutput,
                    timeout_ms=_LLM_TIMEOUT_MS,
                )
                return agent_output_to_dict(output)
            except asyncio.TimeoutError:
                logger.warning("LLM timeout (attempt %d/%d)", attempt, _LLM_MAX_RETRIES)
            except Exception as e:
                logger.warning("LLM error (attempt %d/%d): %s", attempt, _LLM_MAX_RETRIES, str(e)[:150])
            if attempt < _LLM_MAX_RETRIES:
                await asyncio.sleep(1.0 * attempt)
        return None

    async def _execute_action(
        self,
        name: str,
        params: dict,
        controller: PageController,
        bu_page,
    ) -> str:
        try:
            match name:
                case "click_element_by_index":
                    idx = params.get("index", 0)
                    r = await controller.click_element(idx)
                    return r.message
                case "input_text":
                    idx = params.get("index", 0)
                    text = params.get("text", "")
                    r = await controller.input_text(idx, text)
                    return r.message
                case "select_dropdown_option":
                    idx = params.get("index", 0)
                    opt = params.get("option_text", params.get("text", ""))
                    r = await controller.select_option(idx, opt)
                    return r.message
                case "scroll":
                    direction = params.get("direction", "down")
                    amount = params.get("amount", 500)
                    idx = params.get("index")
                    r = await controller.scroll(direction, amount, idx)
                    return r.message
                case "scroll_horizontally":
                    direction = params.get("direction", "right")
                    amount = params.get("amount", params.get("pixels", 300))
                    idx = params.get("index")
                    r = await controller.scroll_horizontally(direction, amount, idx)
                    return r.message
                case "wait":
                    seconds = min(params.get("seconds", 1), 10)
                    last_update = await controller.get_last_update_time()
                    elapsed = time.time() - last_update if last_update > 0 else 0
                    actual = max(0, seconds - elapsed)
                    await asyncio.sleep(actual)
                    return f"Waited {seconds}s (actual {actual:.1f}s)"
                case "go_back":
                    try:
                        await bu_page.go_back()
                        return "Navigated back"
                    except Exception as e:
                        return f"Go back failed: {e}"
                case "close_overlay":
                    closed = await self._close_overlays(bu_page)
                    return f"Closed {closed} overlay(s)"
                case "execute_javascript":
                    script = params.get("script", "")
                    r = await controller.execute_javascript(script)
                    return r.message
                case _:
                    return f"Unknown action: {name}"
        except Exception as e:
            return f"Action {name} failed: {e}"

    async def _close_overlays(self, bu_page) -> int:
        try:
            result = await bu_page.evaluate(
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

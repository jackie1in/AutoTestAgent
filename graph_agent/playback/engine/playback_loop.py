from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from graph_agent.models import ActionType, TabActionType
from graph_agent.playback.engine.helpers import (
    _ensure_frame_attached,
    _extract_login_error_message_from_page_text,
    _format_replay_error,
    _generate_value_from_constraints,
    _is_closed_context_error,
    _is_login_like_url,
    _lookup_transition_evidence_summary,
    _page_for_tab,
    _resolve_navigate_url,
    _retry_action,
    _run_with_http_wait,
    _try_action_with_selector_fallback,
    _urls_same_page,
)

logger = logging.getLogger(__name__)




@dataclass
class _PlaybackLoopState:
    actual_url: str = ""
    last_action_page: Any = None
    login_error_message: str | None = None
    pages_by_tab_id: dict = field(default_factory=dict)
    fallback_tab_by_closed_tab_id: dict = field(default_factory=dict)
    redirected_tab_by_tab_id: dict = field(default_factory=dict)
    saw_auth_credentials: bool = False
    nav_guard_passed: bool = False


async def _run_playback_loop(
    *,
    page,
    ordered_edges: list,
    test_data: dict,
    wait_for_network: bool,
    timeout_ms: int,
    log_callback,
    st: _PlaybackLoopState,
) -> dict | None:
    """Extracted per-edge playback loop.

    Returns an error dict for early exit, or None on success.
    """
    ordered_edges = sorted(
        ordered_edges,
        key=lambda e: (
            e.step_index if e.step_index is not None else 10**9
        ),
    )
    for i, edge in enumerate(ordered_edges):
        selector = edge.selector
        action = edge.action
        page_for_edge = page

        # T7: 兼容 edge.intent is None，不依赖 intent.summary 必然存在
        intent_summary = (
            getattr(edge.intent, "summary", None) or ""
            if edge.intent
            else ""
        )
        log_entry: dict = {
            "step_index": i,
            "selector": selector,
            "action": action,
            "intent": intent_summary,
        }

        try:
            page_for_edge = _page_for_tab(
                st.pages_by_tab_id,
                edge.tab_id,
                st.fallback_tab_by_closed_tab_id,
                st.redirected_tab_by_tab_id,
            )

            if action == ActionType.NAVIGATE:
                navigate_url = _resolve_navigate_url(edge)
                if not navigate_url:
                    raise ValueError(
                        "Navigate step is missing a resolvable URL"
                    )
                assert navigate_url is not None
                if wait_for_network:
                    _nav_url: str = navigate_url  # type: ignore[assignment]

                    async def _goto() -> None:
                        await page_for_edge.goto(_nav_url)

                    await _run_with_http_wait(
                        page_for_edge,
                        _goto,
                        timeout_ms=timeout_ms,
                    )
                else:
                    await page_for_edge.goto(navigate_url)  # type: ignore[arg-type]
                st.actual_url = page_for_edge.url
                st.last_action_page = page_for_edge
                log_entry["success"] = True
                if log_callback:
                    res = log_callback(log_entry)
                    if res and hasattr(res, "__await__"):
                        await res
                continue

            if edge.tab_action == TabActionType.CLOSE:
                close_tab_id = edge.target_tab_id or edge.tab_id
                closing_page = _page_for_tab(
                    st.pages_by_tab_id,
                    close_tab_id,
                    st.fallback_tab_by_closed_tab_id,
                    st.redirected_tab_by_tab_id,
                )
                await closing_page.close()
                pages_by_tab_id.pop(close_tab_id, None)
                opener_tab_id = opener_by_tab_id.pop(close_tab_id, None)
                redirected_tab_by_tab_id.pop(close_tab_id, None)
                st.redirected_tab_by_tab_id = {
                    source_tab_id: target_tab_id
                    for source_tab_id, target_tab_id in redirected_tab_by_tab_id.items()
                    if target_tab_id != close_tab_id
                }
                if opener_tab_id is not None:
                    st.fallback_tab_by_closed_tab_id[close_tab_id] = (
                        opener_tab_id
                    )
                    st.last_action_page = _page_for_tab(
                        st.pages_by_tab_id,
                        opener_tab_id,
                        st.fallback_tab_by_closed_tab_id,
                        st.redirected_tab_by_tab_id,
                    )
                    st.actual_url = last_action_page.url
                else:
                    st.actual_url = page_for_edge.url
                log_entry["success"] = True
                if log_callback:
                    res = log_callback(log_entry)
                    if res and hasattr(res, "__await__"):
                        await res
                continue

            if edge.tab_action == TabActionType.SWITCH:
                switch_target_tab_id = edge.target_tab_id or edge.tab_id
                redirected_tab_by_tab_id.pop(switch_target_tab_id, None)
                st.redirected_tab_by_tab_id[edge.tab_id] = (
                    switch_target_tab_id
                )
                switched_page = _page_for_tab(
                    st.pages_by_tab_id,
                    switch_target_tab_id,
                    st.fallback_tab_by_closed_tab_id,
                    st.redirected_tab_by_tab_id,
                )
                st.last_action_page = switched_page
                st.actual_url = switched_page.url
                log_entry["success"] = True
                if log_callback:
                    res = log_callback(log_entry)
                    if res and hasattr(res, "__await__"):
                        await res
                continue

            if not selector:
                # Skip if no selector (e.g. pure navigation or wait)
                continue

            if edge.tab_action == TabActionType.OPEN:

                async def _do_open_click(loc: Any) -> None:
                    if wait_for_network:
                        await _run_with_http_wait(
                            page_for_edge,
                            loc.click,
                            timeout_ms=timeout_ms,
                        )
                    else:
                        await loc.click()

                async with page_for_edge.expect_popup() as popup_info:
                    await _try_action_with_selector_fallback(
                        page_for_edge, edge, _do_open_click
                    )
                popup_page = await popup_info.value
                popup_page.set_default_timeout(timeout_ms)
                if hasattr(popup_page, "wait_for_load_state"):
                    await popup_page.wait_for_load_state(
                        "domcontentloaded"
                    )
                popup_tab_id = edge.target_tab_id or edge.tab_id
                st.pages_by_tab_id[popup_tab_id] = popup_page
                st.fallback_tab_by_closed_tab_id[popup_tab_id] = edge.tab_id
                fallback_tab_by_closed_tab_id.pop(popup_tab_id, None)
                st.actual_url = popup_page.url or page_for_edge.url
                st.last_action_page = popup_page
                log_entry["success"] = True
                if log_callback:
                    res = log_callback(log_entry)
                    if res and hasattr(res, "__await__"):
                        await res
                continue

            if edge.frame_path:
                await _ensure_frame_attached(
                    page_for_edge, edge.frame_path, timeout_ms
                )

            if action == ActionType.FILL:
                value = ""

                # 1. Prefer runtime test data for deterministic replay credentials.
                if edge.param_name and edge.param_name in test_data:
                    value = str(test_data[edge.param_name])
                # 2. Fall back to recorded value from mapping.
                elif edge.action_value is not None:
                    value = str(edge.action_value)
                # 3. Try constraints generation.
                elif edge.constraints:
                    value = _generate_value_from_constraints(
                        edge.constraints
                    )
                # 4. Fallback
                else:
                    value = "test_value"

                async def _do_fill(loc: Any) -> None:
                    await loc.fill(value)

                await _retry_action(
                    lambda: _try_action_with_selector_fallback(
                        page_for_edge, edge, _do_fill
                    )
                )
                st.actual_url = page_for_edge.url
                st.last_action_page = page_for_edge
                param_name = (
                    (edge.param_name or "").lower()
                    if edge.param_name
                    else ""
                )
                intent_key = (
                    (getattr(edge.intent, "key", "") or "").lower()
                    if edge.intent
                    else ""
                )
                if param_name in {
                    "username",
                    "password",
                } or intent_key.startswith("auth.fill"):
                    st.saw_auth_credentials = True

            elif action == ActionType.RICH_TEXT:
                value = ""
                if edge.param_name and edge.param_name in test_data:
                    value = str(test_data[edge.param_name])
                elif edge.action_value is not None:
                    value = str(edge.action_value)
                elif edge.constraints:
                    value = _generate_value_from_constraints(
                        edge.constraints
                    )
                else:
                    value = "test_value"

                async def _do_rich_text(loc: Any) -> None:
                    await loc.click()
                    kb = page_for_edge.keyboard
                    await kb.press("Control+a")
                    await kb.type(value)

                await _retry_action(
                    lambda: _try_action_with_selector_fallback(
                        page_for_edge, edge, _do_rich_text
                    )
                )
                st.actual_url = page_for_edge.url
                st.last_action_page = page_for_edge

            elif action == ActionType.SELECT:
                value = ""
                if edge.param_name and edge.param_name in test_data:
                    value = str(test_data[edge.param_name])
                elif edge.action_value is not None:
                    value = str(edge.action_value)
                else:
                    value = ""

                async def _do_select(loc: Any) -> None:
                    await loc.select_option(value)

                await _retry_action(
                    lambda: _try_action_with_selector_fallback(
                        page_for_edge, edge, _do_select
                    )
                )
                st.actual_url = page_for_edge.url
                st.last_action_page = page_for_edge

            elif action == ActionType.CLICK:
                url_before_click = page_for_edge.url

                async def _do_click(loc: Any) -> None:
                    if wait_for_network:
                        await _run_with_http_wait(
                            page_for_edge,
                            loc.click,
                            timeout_ms=timeout_ms,
                        )
                    else:
                        await loc.click()

                await _retry_action(
                    lambda: _try_action_with_selector_fallback(
                        page_for_edge, edge, _do_click
                    )
                )

                # --- Post-click recovery & navigation assertion ---

                # 1. Tab closure recovery: the click may have closed
                #    the current tab (common in SSO / OAuth flows).
                if getattr(page_for_edge, "is_closed", lambda: False)():
                    open_pages = [
                        pg
                        for pg in browser_context.pages
                        if not getattr(pg, "is_closed", lambda: False)()
                    ]
                    if open_pages:
                        st.pages_by_tab_id[edge.tab_id] = open_pages[0]
                        page_for_edge = open_pages[0]

                # 2. General redirect wait: if the click triggered a
                #    navigation, wait for the new page's DOM **and**
                #    network to settle.  SPAs often fire
                #    DOMContentLoaded long before the JS framework
                #    finishes mounting components + event handlers,
                #    so we use "networkidle" first (with a short
                #    timeout to avoid hangs on polling/websocket
                #    apps) then fall back to "domcontentloaded".
                url_after_click = page_for_edge.url
                if (
                    url_after_click
                    and url_before_click
                    and url_after_click != url_before_click
                    and hasattr(page_for_edge, "wait_for_load_state")
                ):
                    try:
                        await page_for_edge.wait_for_load_state(
                            "networkidle",
                            timeout=min(timeout_ms, 15_000),
                        )
                    except Exception:
                        try:
                            await page_for_edge.wait_for_load_state(
                                "domcontentloaded",
                                timeout=min(timeout_ms, 10_000),
                            )
                        except Exception:
                            pass

                st.actual_url = page_for_edge.url
                st.last_action_page = page_for_edge

                # 3. General navigation assertion: the recording
                #    captured expected source/target URLs.  If the
                #    click was supposed to change the page but didn't,
                #    detect the failure early instead of letting
                #    downstream selectors time out one by one.
                expected_nav = (
                    not st.nav_guard_passed
                    and edge.source_url
                    and edge.target_url
                    and not _urls_same_page(
                        edge.source_url, edge.target_url
                    )
                )
                if (
                    expected_nav
                    and st.actual_url
                    and _urls_same_page(
                        st.actual_url,
                        edge.source_url,  # type: ignore[arg-type]
                    )
                ):
                    # Brief grace period for client-side JS redirects
                    # that fire after DOMContentLoaded.
                    await asyncio.sleep(2.0)
                    if getattr(
                        page_for_edge, "is_closed", lambda: False
                    )():
                        open_pages = [
                            pg
                            for pg in browser_context.pages
                            if not getattr(
                                pg, "is_closed", lambda: False
                            )()
                        ]
                        if open_pages:
                            st.pages_by_tab_id[edge.tab_id] = open_pages[0]
                            page_for_edge = open_pages[0]
                    st.actual_url = page_for_edge.url

                # Still on the source page after expected navigation?
                if (
                    expected_nav
                    and st.actual_url
                    and _urls_same_page(
                        st.actual_url,
                        edge.source_url,  # type: ignore[arg-type]
                    )
                ):
                    is_login = _is_login_like_url(st.actual_url)
                    if is_login and st.saw_auth_credentials:
                        guard_error = (
                            "Login flow did not leave login page after "
                            "credential submit/click; authentication "
                            "likely failed (credentials or selector drift)."
                        )
                        if not st.login_error_message:
                            try:
                                body_text = await page_for_edge.locator(
                                    "body"
                                ).inner_text()
                                st.login_error_message = _extract_login_error_message_from_page_text(
                                    body_text
                                )
                            except Exception:
                                st.login_error_message = None
                        if st.login_error_message:
                            guard_error = f"{guard_error} Server says: {st.login_error_message}."
                    else:
                        guard_error = (
                            f"Click was expected to navigate from "
                            f"{edge.source_url} to {edge.target_url} "
                            f"but page stayed on {st.actual_url}."
                        )
                    log_entry["success"] = False
                    log_entry["error"] = guard_error
                    if log_callback:
                        res = log_callback(log_entry)
                        if res and hasattr(res, "__await__"):
                            await res
                    return {
                        "success": False,
                        "st.actual_url": st.actual_url,
                        "error": guard_error,
                        "failed_step_index": i,
                        "failed_edge_id": edge.edge_id,
                    }

                # Navigation succeeded — skip future guard checks
                # for this flow to avoid false positives on subsequent
                # same-page clicks.
                if expected_nav:
                    st.nav_guard_passed = True

                # 4. Lookahead assertion: if this click has no
                #    frame_path but the NEXT step needs an iframe,
                #    verify the iframe appeared.  A silent
                #    no-op click (e.g. SPA not ready) would
                #    leave the page without the iframe, causing
                #    downstream failures.
                if not edge.frame_path and i + 1 < len(ordered_edges):
                    next_edge = ordered_edges[i + 1]
                    if next_edge.frame_path:
                        try:
                            await _ensure_frame_attached(
                                page_for_edge,
                                next_edge.frame_path,
                                timeout_ms=min(timeout_ms, 15_000),
                            )
                        except Exception:
                            lookahead_error = (
                                f"Click on '{edge.selector}' "
                                f"completed but the expected "
                                f"iframe for the next step did "
                                f"not appear; the click likely "
                                f"had no effect (SPA not ready "
                                f"or wrong element)."
                            )
                            log_entry["success"] = False
                            log_entry["error"] = lookahead_error
                            if log_callback:
                                res = log_callback(log_entry)
                                if res and hasattr(res, "__await__"):
                                    await res
                            return {
                                "success": False,
                                "st.actual_url": st.actual_url,
                                "error": lookahead_error,
                                "failed_step_index": i,
                                "failed_edge_id": edge.edge_id,
                            }

            else:
                # Unknown action, log warning but continue? Or fail?
                # For now, just log
                logger.warning(
                    "Unknown action type %s at step %d", action, i
                )

            log_entry["success"] = True
            if log_callback:
                res = log_callback(log_entry)
                if res and hasattr(res, "__await__"):
                    await res

        except Exception as step_error:  # noqa: BLE001
            # Closed-context recovery: when Playwright reports
            # the page/context/browser as closed (common after
            # SPA navigations that replace the page object),
            # try to recover to a surviving open page and
            # replay the failed step once more.
            if _is_closed_context_error(step_error):
                open_pages = [
                    p
                    for p in browser_context.pages
                    if not getattr(p, "is_closed", lambda: False)()
                ]
                if open_pages:
                    recovered = open_pages[0]
                    st.pages_by_tab_id[edge.tab_id] = recovered
                    page_for_edge = recovered
                    try:
                        if hasattr(recovered, "wait_for_load_state"):
                            await recovered.wait_for_load_state(
                                "domcontentloaded",
                                timeout=min(timeout_ms, 15_000),
                            )
                        if edge.frame_path:
                            await _ensure_frame_attached(
                                recovered,
                                edge.frame_path,
                                timeout_ms,
                            )
                        if action == ActionType.FILL:
                            value = ""
                            if (
                                edge.param_name
                                and edge.param_name in test_data
                            ):
                                value = str(test_data[edge.param_name])
                            elif edge.action_value is not None:
                                value = str(edge.action_value)
                            elif edge.constraints:
                                value = (
                                    _generate_value_from_constraints(
                                        edge.constraints
                                    )
                                )
                            else:
                                value = "test_value"

                            async def _do_fill_r(
                                loc: Any,
                            ) -> None:
                                await loc.fill(value)

                            await _retry_action(
                                lambda: (
                                    _try_action_with_selector_fallback(
                                        page_for_edge,
                                        edge,
                                        _do_fill_r,
                                    )
                                )
                            )
                        elif action == ActionType.RICH_TEXT:
                            value = ""
                            if (
                                edge.param_name
                                and edge.param_name in test_data
                            ):
                                value = str(test_data[edge.param_name])
                            elif edge.action_value is not None:
                                value = str(edge.action_value)
                            elif edge.constraints:
                                value = (
                                    _generate_value_from_constraints(
                                        edge.constraints
                                    )
                                )
                            else:
                                value = "test_value"

                            async def _do_rich_text_r(
                                loc: Any,
                            ) -> None:
                                await loc.click()
                                kb = page_for_edge.keyboard
                                await kb.press("Control+a")
                                await kb.type(value)

                            await _retry_action(
                                lambda: (
                                    _try_action_with_selector_fallback(
                                        page_for_edge,
                                        edge,
                                        _do_rich_text_r,
                                    )
                                )
                            )
                        elif action == ActionType.SELECT:
                            value = ""
                            if (
                                edge.param_name
                                and edge.param_name in test_data
                            ):
                                value = str(test_data[edge.param_name])
                            elif edge.action_value is not None:
                                value = str(edge.action_value)
                            else:
                                value = ""

                            async def _do_select_r(
                                loc: Any,
                            ) -> None:
                                await loc.select_option(value)

                            await _retry_action(
                                lambda: (
                                    _try_action_with_selector_fallback(
                                        page_for_edge,
                                        edge,
                                        _do_select_r,
                                    )
                                )
                            )
                        elif action == ActionType.CLICK:

                            async def _do_click_r(
                                loc: Any,
                            ) -> None:
                                await loc.click()

                            await _retry_action(
                                lambda: (
                                    _try_action_with_selector_fallback(
                                        page_for_edge,
                                        edge,
                                        _do_click_r,
                                    )
                                )
                            )
                        st.actual_url = page_for_edge.url
                        st.last_action_page = page_for_edge
                        log_entry["success"] = True
                        if log_callback:
                            res = log_callback(log_entry)
                            if res and hasattr(res, "__await__"):
                                await res
                        continue
                    except Exception:
                        pass

            formatted_error = _format_replay_error(
                step_error, edge, edge.tab_id
            )
            evidence_summary = (
                await _lookup_transition_evidence_summary(edge.edge_id)
            )
            if evidence_summary:
                formatted_error = (
                    f"{formatted_error} | evidence: {evidence_summary}"
                )
            log_entry["success"] = False
            log_entry["error"] = formatted_error
            if log_callback:
                res = log_callback(log_entry)
                if res and hasattr(res, "__await__"):
                    await res
            try:
                st.actual_url = pages_by_tab_id.get(
                    edge.tab_id, page_for_edge
                ).url
            except Exception:
                st.actual_url = st.actual_url or ""
            return {
                "success": False,
                "st.actual_url": st.actual_url,
                "error": formatted_error,
                "failed_step_index": i,
                "failed_edge_id": edge.edge_id,
            }

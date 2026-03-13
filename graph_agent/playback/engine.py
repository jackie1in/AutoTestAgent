"""Playwright playback engine: run edge list (async, no LLM)."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Awaitable
from time import monotonic
from typing import Any

from playwright.async_api import expect, async_playwright
from graph_agent.models import (
    ActionType,
    ElementConstraints,
    FrameLocatorSnapshot,
    GraphEdge,
    TabActionType,
)

DEFAULT_TIMEOUT_MS = 60_000
NETWORK_OBSERVE_MS = 150
NETWORK_POLL_MS = 25
PLAYBACK_RETRY_COUNT = 2
PLAYBACK_RETRY_DELAY_S = 0.3


def _is_login_like_url(url: str) -> bool:
    """Best-effort login page detection for early auth failure guard."""
    lowered = url.lower()
    return "login" in lowered or "signin" in lowered or "sign-in" in lowered


def _extract_login_error_message(payload_text: str) -> str | None:
    """Parse backend login response text and return a human-readable error message."""
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("serviceSuccess") is True:
        return None
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        return None
    msg = str(first.get("msg") or "").strip()
    return msg or None


def _extract_login_error_message_from_page_text(page_text: str) -> str | None:
    """Best-effort extraction from rendered login page messages."""
    if not page_text:
        return None
    candidates = (
        "帐号已过期",
        "账号已过期",
        "密码错误",
        "登录失败",
        "用户不存在",
    )
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in candidates):
            return stripped
    return None


def _generate_value_from_constraints(constraints: ElementConstraints) -> str:
    """Generate a dummy value based on constraints."""
    fmt = constraints.format
    if fmt == "email":
        return f"test-{uuid.uuid4()}@example.com"
    if fmt == "password":
        return "Test@1234"  # Static strong password
    if fmt == "phone":
        return "13800138000"
    return "test_value"


def _is_http_request(request: Any) -> bool:
    """Track only HTTP(S) requests and ignore websocket traffic."""
    url = str(getattr(request, "url", "") or "")
    if not url.startswith(("http://", "https://")):
        return False
    resource_type = str(getattr(request, "resource_type", "") or "").lower()
    return resource_type != "websocket"


def _resolve_navigate_url(edge: GraphEdge) -> str | None:
    """Best-effort URL resolution for explicit navigate steps."""
    for candidate in (edge.action_value, edge.target, edge.source):
        value = str(candidate or "").strip()
        if value.startswith(("http://", "https://", "data:")):
            return value
    return None


def _remove_listener(page: Any, event: str, handler: Callable[[Any], None]) -> None:
    """Detach event listener across Playwright versions and test doubles."""
    if hasattr(page, "off"):
        page.off(event, handler)
        return
    if hasattr(page, "remove_listener"):
        page.remove_listener(event, handler)


def _is_transient_error(exc: BaseException) -> bool:
    """Detect transient failures (async load, timing) that may succeed on retry."""
    msg = str(exc).lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    return any(
        token in msg
        for token in ("timed out", "timeout", "waiting for selector", "locator")
    )


async def _retry_action(
    action: Callable[[], Awaitable[None]],
    *,
    max_retries: int = PLAYBACK_RETRY_COUNT,
    retry_delay_s: float = PLAYBACK_RETRY_DELAY_S,
) -> None:
    """Retry an action on transient failures to improve playback stability."""
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            await action()
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _is_transient_error(exc):
                await asyncio.sleep(retry_delay_s)
                continue
            raise
    if last_exc is not None:
        raise last_exc


def _frame_selector_candidates(frame: FrameLocatorSnapshot, index: int) -> list[str]:
    """Return ordered list of selector candidates for a frame (stability fallback)."""
    candidates: list[str] = []
    primary = str(frame.selector or "").strip()
    if primary:
        candidates.append(primary)
    css = str(frame.css_selector or "").strip()
    if css and css != primary:
        candidates.append(css)
    xp = frame.xpath or frame.x_path
    xp_str = str(xp or "").strip()
    if xp_str and xp_str != primary:
        candidates.append(f"xpath={xp_str}" if not xp_str.startswith("xpath=") else xp_str)
    if not candidates:
        raise ValueError(f"Missing selector for iframe level {index}")
    return candidates


def _resolve_playback_context(page: Any, frame_path: list[FrameLocatorSnapshot]) -> Any:
    """Resolve the page or nested frame context for an edge."""
    context = page
    for index, frame in enumerate(frame_path, start=1):
        candidates = _frame_selector_candidates(frame, index)
        last_exc: Exception | None = None
        for frame_selector in candidates:
            try:
                context = context.frame_locator(frame_selector)
                break
            except Exception as exc:
                last_exc = exc
                continue
        else:
            raise ValueError(
                f"Failed to locate iframe level {index}: {candidates[0]}"
            ) from last_exc
    return context


def _locator_in_context(page: Any, selector: str, frame_path: list[FrameLocatorSnapshot]) -> Any:
    """Build a locator in the target page/frame context."""
    return _resolve_playback_context(page, frame_path).locator(selector)


def _format_replay_error(
    step_error: Exception,
    edge: GraphEdge,
    tab_id: str,
) -> str:
    """Format playback error with context for diagnosis (tab/iframe/selector/async_load)."""
    orig = str(step_error)
    # Tab errors already have clear message from _page_for_tab
    if "Target tab does not exist" in orig:
        return f"tab: {orig}"
    # Iframe errors from _resolve_playback_context
    if "iframe" in orig.lower() or "Failed to locate iframe" in orig:
        return f"iframe: {orig}"
    # Async load / timing: element or frame not ready yet
    if _is_transient_error(step_error):
        prefix = "async_load: "
    else:
        prefix = ""
    # Selector/element errors: add context for diagnosis
    parts = [f"tab={tab_id}"]
    if edge.frame_path:
        frame_summary = " > ".join(
            f"level{i}={f.selector}" for i, f in enumerate(edge.frame_path, 1)
        )
        parts.append(f"frame_path={frame_summary}")
    if edge.selector:
        parts.append(f"selector={edge.selector}")
    ctx = ", ".join(parts)
    return f"{prefix}selector: {orig} (context: {ctx})"


def _page_for_tab(
    pages_by_tab_id: dict[str, Any],
    tab_id: str,
    fallback_tab_by_closed_tab_id: dict[str, str] | None = None,
    redirected_tab_by_tab_id: dict[str, str] | None = None,
) -> Any:
    """Return the page for a tab or raise a readable error."""
    current_tab_id = tab_id
    visited_tab_ids: set[str] = set()
    fallback_tab_by_closed_tab_id = fallback_tab_by_closed_tab_id or {}
    redirected_tab_by_tab_id = redirected_tab_by_tab_id or {}

    while True:
        if current_tab_id in visited_tab_ids:
            break
        visited_tab_ids.add(current_tab_id)

        redirected_tab_id = redirected_tab_by_tab_id.get(current_tab_id)
        if redirected_tab_id is not None:
            current_tab_id = redirected_tab_id
            continue

        page = pages_by_tab_id.get(current_tab_id)
        if page is not None:
            return page
        fallback_tab_id = fallback_tab_by_closed_tab_id.get(current_tab_id)
        if fallback_tab_id is None:
            break
        current_tab_id = fallback_tab_id

    raise ValueError(f"Target tab does not exist: {tab_id}")


async def _run_with_http_wait(
    page: Any,
    action: Callable[[], Awaitable[None]],
    *,
    timeout_ms: int,
) -> None:
    """Run an action and wait for the resulting HTTP request batch to settle."""
    pending: set[int] = set()
    saw_request = False
    changed = asyncio.Event()

    def _on_request(request: Any) -> None:
        nonlocal saw_request
        if not _is_http_request(request):
            return
        saw_request = True
        pending.add(id(request))
        changed.set()

    def _on_request_done(request: Any) -> None:
        if not _is_http_request(request):
            return
        pending.discard(id(request))
        changed.set()

    page.on("request", _on_request)
    page.on("requestfinished", _on_request_done)
    page.on("requestfailed", _on_request_done)
    try:
        await action()

        observe_deadline = monotonic() + (NETWORK_OBSERVE_MS / 1000)
        while not saw_request and monotonic() < observe_deadline:
            timeout = max(0.0, min(observe_deadline - monotonic(), NETWORK_POLL_MS / 1000))
            if timeout == 0:
                break
            try:
                await asyncio.wait_for(changed.wait(), timeout=timeout)
            except TimeoutError:
                continue
            changed.clear()

        request_deadline = monotonic() + (timeout_ms / 1000)
        while saw_request and pending:
            timeout = max(0.0, min(request_deadline - monotonic(), NETWORK_POLL_MS / 1000))
            if timeout == 0:
                raise TimeoutError("Timed out waiting for HTTP requests to finish")
            try:
                await asyncio.wait_for(changed.wait(), timeout=timeout)
            except TimeoutError:
                continue
            changed.clear()
    finally:
        _remove_listener(page, "request", _on_request)
        _remove_listener(page, "requestfinished", _on_request_done)
        _remove_listener(page, "requestfailed", _on_request_done)


async def run_playback(
    edge_list: list[GraphEdge],
    test_data: dict[str, Any],
    start_url: str,
    expected_end_url: str | None = None,
    log_callback: Callable[[dict], Awaitable[None] | None] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    wait_for_network: bool = True,
) -> dict:
    """Run recorded edge list with Playwright (async).

    Args:
        edge_list: List of GraphEdge objects.
        test_data: Data for fill actions (e.g. {"username": "u1", "password": "p1"}).
        start_url: Initial URL to open.
        expected_end_url: If set, assert page URL after last step (expect(page).to_have_url(...)).
        log_callback: Called after each step with a structured log dict (step_index, selector, action, success, error if any).
        timeout_ms: Timeout for navigation and actions (default 60s).

    Returns:
        {"success": bool, "actual_url": str, "error": str | None}
    """
    actual_url = ""
    
    # Config from env
    channel = os.getenv("PLAYWRIGHT_CHANNEL", "chrome")  # Default to chrome if not set
    # Default to headless=False for playback visibility unless explicitly set to true
    headless = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true"

    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(channel=channel, headless=headless)
            try:
                page = await browser.new_page()
                page.set_default_timeout(timeout_ms)
                pages_by_tab_id: dict[str, Any] = {"tab-0": page}
                opener_by_tab_id: dict[str, str] = {}
                fallback_tab_by_closed_tab_id: dict[str, str] = {}
                redirected_tab_by_tab_id: dict[str, str] = {}
                last_action_page = page
                
                # Navigate to start URL
                try:
                    await page.goto(start_url)
                except Exception as e:
                     return {
                        "success": False,
                        "actual_url": "",
                        "error": f"Failed to navigate to start URL: {e}",
                    }
                
                actual_url = page.url

                started_on_login_url = _is_login_like_url(start_url)
                saw_auth_credentials = False
                login_guard_satisfied = not started_on_login_url
                login_error_message: str | None = None

                async def _capture_login_response(response: Any) -> None:
                    nonlocal login_error_message
                    if not started_on_login_url:
                        return
                    url = str(getattr(response, "url", "") or "").lower()
                    if "/project/login" not in url:
                        return
                    try:
                        body = await response.text()
                    except Exception:
                        return
                    parsed = _extract_login_error_message(body)
                    if parsed:
                        login_error_message = parsed

                def _on_response(response: Any) -> None:
                    asyncio.create_task(_capture_login_response(response))

                page.on("response", _on_response)

                try:
                    ordered_edges = sorted(edge_list, key=lambda e: (e.step_index if e.step_index is not None else 10**9))
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
                                pages_by_tab_id,
                                edge.tab_id,
                                fallback_tab_by_closed_tab_id,
                                redirected_tab_by_tab_id,
                            )

                            if action == ActionType.NAVIGATE:
                                navigate_url = _resolve_navigate_url(edge)
                                if not navigate_url:
                                    raise ValueError("Navigate step is missing a resolvable URL")
                                if wait_for_network:
                                    async def _goto() -> None:
                                        await page_for_edge.goto(navigate_url)

                                    await _run_with_http_wait(
                                        page_for_edge,
                                        _goto,
                                        timeout_ms=timeout_ms,
                                    )
                                else:
                                    await page_for_edge.goto(navigate_url)
                                actual_url = page_for_edge.url
                                last_action_page = page_for_edge
                                log_entry["success"] = True
                                if log_callback:
                                    res = log_callback(log_entry)
                                    if res and hasattr(res, "__await__"):
                                        await res
                                continue

                            if edge.tab_action == TabActionType.CLOSE:
                                close_tab_id = edge.target_tab_id or edge.tab_id
                                closing_page = _page_for_tab(
                                    pages_by_tab_id,
                                    close_tab_id,
                                    fallback_tab_by_closed_tab_id,
                                    redirected_tab_by_tab_id,
                                )
                                await closing_page.close()
                                pages_by_tab_id.pop(close_tab_id, None)
                                opener_tab_id = opener_by_tab_id.pop(close_tab_id, None)
                                redirected_tab_by_tab_id.pop(close_tab_id, None)
                                redirected_tab_by_tab_id = {
                                    source_tab_id: target_tab_id
                                    for source_tab_id, target_tab_id in redirected_tab_by_tab_id.items()
                                    if target_tab_id != close_tab_id
                                }
                                if opener_tab_id is not None:
                                    fallback_tab_by_closed_tab_id[close_tab_id] = opener_tab_id
                                    last_action_page = _page_for_tab(
                                        pages_by_tab_id,
                                        opener_tab_id,
                                        fallback_tab_by_closed_tab_id,
                                        redirected_tab_by_tab_id,
                                    )
                                    actual_url = last_action_page.url
                                else:
                                    actual_url = page_for_edge.url
                                log_entry["success"] = True
                                if log_callback:
                                    res = log_callback(log_entry)
                                    if res and hasattr(res, "__await__"):
                                        await res
                                continue

                            if edge.tab_action == TabActionType.SWITCH:
                                switch_target_tab_id = edge.target_tab_id or edge.tab_id
                                redirected_tab_by_tab_id.pop(switch_target_tab_id, None)
                                redirected_tab_by_tab_id[edge.tab_id] = switch_target_tab_id
                                switched_page = _page_for_tab(
                                    pages_by_tab_id,
                                    switch_target_tab_id,
                                    fallback_tab_by_closed_tab_id,
                                    redirected_tab_by_tab_id,
                                )
                                last_action_page = switched_page
                                actual_url = switched_page.url
                                log_entry["success"] = True
                                if log_callback:
                                    res = log_callback(log_entry)
                                    if res and hasattr(res, "__await__"):
                                        await res
                                continue

                            if not selector:
                                 # Skip if no selector (e.g. pure navigation or wait)
                                 continue

                            loc = _locator_in_context(page_for_edge, selector, edge.frame_path)

                            if edge.tab_action == TabActionType.OPEN:
                                async with page_for_edge.expect_popup() as popup_info:
                                    if wait_for_network:
                                        await _run_with_http_wait(
                                            page_for_edge,
                                            loc.click,
                                            timeout_ms=timeout_ms,
                                        )
                                    else:
                                        await loc.click()
                                popup_page = await popup_info.value
                                popup_page.set_default_timeout(timeout_ms)
                                if hasattr(popup_page, "wait_for_load_state"):
                                    await popup_page.wait_for_load_state("domcontentloaded")
                                popup_tab_id = edge.target_tab_id or edge.tab_id
                                pages_by_tab_id[popup_tab_id] = popup_page
                                opener_by_tab_id[popup_tab_id] = edge.tab_id
                                fallback_tab_by_closed_tab_id.pop(popup_tab_id, None)
                                actual_url = popup_page.url or page_for_edge.url
                                last_action_page = popup_page
                                log_entry["success"] = True
                                if log_callback:
                                    res = log_callback(log_entry)
                                    if res and hasattr(res, "__await__"):
                                        await res
                                continue
                            
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
                                    value = _generate_value_from_constraints(edge.constraints)
                                # 4. Fallback
                                else:
                                    value = "test_value"
                                
                                async def _do_fill() -> None:
                                    await loc.fill(value)

                                await _retry_action(_do_fill)
                                actual_url = page_for_edge.url
                                last_action_page = page_for_edge
                                param_name = (edge.param_name or "").lower() if edge.param_name else ""
                                intent_key = (getattr(edge.intent, "key", "") or "").lower() if edge.intent else ""
                                if param_name in {"username", "password"} or intent_key.startswith("auth.fill"):
                                    saw_auth_credentials = True
                                
                            elif action == ActionType.CLICK:
                                async def _do_click() -> None:
                                    if wait_for_network:
                                        await _run_with_http_wait(
                                            page_for_edge,
                                            loc.click,
                                            timeout_ms=timeout_ms,
                                        )
                                    else:
                                        await loc.click()

                                await _retry_action(_do_click)
                                actual_url = page_for_edge.url
                                last_action_page = page_for_edge
                                if (
                                    not login_guard_satisfied
                                    and saw_auth_credentials
                                    and _is_login_like_url(actual_url)
                                ):
                                    guard_error = (
                                        "Login flow did not leave login page after credential submit/click; "
                                        "authentication likely failed (credentials or selector drift)."
                                    )
                                    if not login_error_message:
                                        try:
                                            body_text = await page_for_edge.locator("body").inner_text()
                                            login_error_message = _extract_login_error_message_from_page_text(body_text)
                                        except Exception:
                                            login_error_message = None
                                    if login_error_message:
                                        guard_error = f"{guard_error} Server says: {login_error_message}."
                                    log_entry["success"] = False
                                    log_entry["error"] = guard_error
                                    if log_callback:
                                        res = log_callback(log_entry)
                                        if res and hasattr(res, "__await__"):
                                            await res
                                    return {
                                        "success": False,
                                        "actual_url": actual_url,
                                        "error": guard_error,
                                    }
                                if saw_auth_credentials:
                                    login_guard_satisfied = True
                                
                            else:
                                # Unknown action, log warning but continue? Or fail?
                                # For now, just log
                                print(f"Warning: Unknown action type {action} at step {i}")

                            log_entry["success"] = True
                            if log_callback:
                                res = log_callback(log_entry)
                                if res and hasattr(res, "__await__"):
                                    await res
                                
                        except Exception as step_error:  # noqa: BLE001
                            formatted_error = _format_replay_error(
                                step_error, edge, edge.tab_id
                            )
                            log_entry["success"] = False
                            log_entry["error"] = formatted_error
                            if log_callback:
                                res = log_callback(log_entry)
                                if res and hasattr(res, "__await__"):
                                    await res
                            actual_url = pages_by_tab_id.get(edge.tab_id, page_for_edge).url
                            return {
                                "success": False,
                                "actual_url": actual_url,
                                "error": formatted_error,
                            }
                finally:
                    _remove_listener(page, "response", _on_response)

                if expected_end_url is not None:
                    try:
                        await expect(last_action_page).to_have_url(expected_end_url)
                    except AssertionError as e:
                         return {
                            "success": False,
                            "actual_url": actual_url,
                            "error": f"Expected URL {expected_end_url}, got {actual_url}. Error: {e}",
                        }

                return {"success": True, "actual_url": actual_url, "error": None}
            finally:
                await browser.close()
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "actual_url": actual_url or "",
            "error": str(e),
        }

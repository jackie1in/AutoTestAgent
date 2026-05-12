from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from playwright.async_api import async_playwright, expect

from graph_agent.models import GraphEdge
from graph_agent.playback.engine.helpers import (
    DEFAULT_TIMEOUT_MS,
    _extract_login_error_message,
    _remove_listener,
)

logger = logging.getLogger(__name__)


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
        {
            "success": bool,
            "actual_url": str,
            "error": str | None,
            "failed_step_index": int | None,
            "failed_edge_id": str | None,
        }
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
                        "failed_step_index": None,
                        "failed_edge_id": None,
                    }

                actual_url = page.url

                login_error_message: str | None = None

                async def _capture_login_response(response: Any) -> None:
                    nonlocal login_error_message
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
                    from graph_agent.playback.engine.playback_loop import _PlaybackLoopState, _run_playback_loop

                    ordered_edges = sorted(
                        edge_list,
                        key=lambda e: (
                            e.step_index if e.step_index is not None else 10**9
                        ),
                    )

                    st = _PlaybackLoopState()
                    st.actual_url = actual_url
                    st.pages_by_tab_id = pages_by_tab_id
                    st.fallback_tab_by_closed_tab_id = fallback_tab_by_closed_tab_id
                    st.redirected_tab_by_tab_id = redirected_tab_by_tab_id

                    early = await _run_playback_loop(
                        page=page,
                        ordered_edges=ordered_edges,
                        test_data=test_data,
                        wait_for_network=wait_for_network,
                        timeout_ms=timeout_ms,
                        log_callback=log_callback,
                        st=st,
                    )
                    if early is not None:
                        return early

                    actual_url = st.actual_url
                    last_action_page = st.last_action_page
                    login_error_message = st.login_error_message
                    pages_by_tab_id = st.pages_by_tab_id
                    fallback_tab_by_closed_tab_id = st.fallback_tab_by_closed_tab_id
                    redirected_tab_by_tab_id = st.redirected_tab_by_tab_id
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
                            "failed_step_index": None,
                            "failed_edge_id": None,
                        }

                return {
                    "success": True,
                    "actual_url": actual_url,
                    "error": None,
                    "failed_step_index": None,
                    "failed_edge_id": None,
                }
            finally:
                await browser.close()
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "actual_url": actual_url or "",
            "error": str(e),
            "failed_step_index": None,
            "failed_edge_id": None,
        }

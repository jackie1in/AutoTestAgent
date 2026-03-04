"""Playwright playback engine: run edge list (async, no LLM)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Awaitable
from typing import Any

from playwright.async_api import expect, async_playwright
from graph_agent.models import ActionType, ElementConstraints, GraphEdge

DEFAULT_TIMEOUT_MS = 60_000


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


async def run_playback(
    edge_list: list[GraphEdge],
    test_data: dict[str, Any],
    start_url: str,
    expected_end_url: str | None = None,
    log_callback: Callable[[dict], Awaitable[None] | None] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
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

                for i, edge in enumerate(edge_list):
                    selector = edge.selector
                    action = edge.action
                    
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
                        if not selector:
                             # Skip if no selector (e.g. pure navigation or wait)
                             # But check if it's a navigation action?
                             if action == ActionType.NAVIGATE:
                                 # Navigation logic if needed, but usually handled by goto or clicks
                                 pass
                             continue

                        loc = page.locator(selector)
                        
                        if action == ActionType.FILL:
                            data_key = edge.data_key
                            value = ""
                            
                            # 1. Try test_data
                            if data_key and data_key in test_data:
                                value = str(test_data[data_key])
                            # 2. Try constraints generation
                            elif edge.constraints:
                                value = _generate_value_from_constraints(edge.constraints)
                            # 3. Fallback
                            else:
                                value = "test_value"
                            
                            await loc.fill(value)
                            
                        elif action == ActionType.CLICK:
                            await loc.click()
                            
                        elif action == ActionType.NAVIGATE:
                            # Usually handled by clicks, but if explicit navigate...
                            pass
                            
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
                        log_entry["success"] = False
                        log_entry["error"] = str(step_error)
                        if log_callback:
                            res = log_callback(log_entry)
                            if res and hasattr(res, "__await__"):
                                await res
                        actual_url = page.url
                        return {
                            "success": False,
                            "actual_url": actual_url,
                            "error": str(step_error),
                        }

                actual_url = page.url
                if expected_end_url is not None:
                    try:
                        await expect(page).to_have_url(expected_end_url)
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

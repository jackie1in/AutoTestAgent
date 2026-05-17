"""Intent replay via CLI — find paths for an intent key and replay actions only.

Usage (via runner CLI):
    python -m graph_agent.cartography.runner --mode replay-intent --intent "auth.login"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def run_replay_intent(
    intent_key: str,
    *,
    start_url: str = "",
    output_path: str = "",
    headless: bool = True,
) -> str:
    """Load edges from Neo4j, find paths for intent, replay selected path.

    Returns:
        Summary string of the replay result.
    """
    # 1. Connect to Neo4j
    from graph_agent.neo4j_client.driver import Neo4jDriver

    neo4j = Neo4jDriver()
    await neo4j.connect()
    driver = neo4j.driver

    try:
        # 2. List available intents (query Neo4j directly, no edge loading needed)
        if not intent_key:
            await _list_available_intents_from_neo4j(driver)
            intent_key = input("\nEnter intent key (or Chinese query): ").strip()
            if not intent_key:
                return "No intent key provided."
    
        # 3. Load edges from Neo4j
        from graph_agent.web.app.data_queries import _get_edges_from_neo4j
    
        edges = await _get_edges_from_neo4j(driver)
    
        if not edges:
            return "No edges found in Neo4j."
    
        # 4. Find all matching paths
        from graph_agent.graph.pathfinding import find_all_paths_for_intent
    
        paths = find_all_paths_for_intent(intent_key, edges)
    
        if not paths:
            # Try with partial match info
            print(f"\nNo exact match for '{intent_key}'. Try one of the listed intents, or use a shorter query.")
            return f"No paths found for intent: {intent_key}"
    
        # 4. Select path
        if len(paths) == 1:
            selected = paths[0]
            logger.info("1 path found, auto-selecting.")
        else:
            selected = _interactive_select(paths)
            if selected is None:
                return "User cancelled."
    
        # 4. Resolve start URL
        resolved_url = start_url
        if not resolved_url:
            for e in selected:
                if e.source_url:
                    resolved_url = e.source_url
                    break
        if not resolved_url:
            resolved_url = (os.getenv("MAPPING_URL") or "").strip()
        if not resolved_url:
            return "No start URL found. Set MAPPING_URL or pass --url."
    
        logger.info("Replaying %d edges for intent=%s from %s", len(selected), intent_key, resolved_url)
    
        # 5. Execute action-only replay
        result = await _replay_actions_only(selected, resolved_url, headless)
    
        # 6. Save result
        summary = json.dumps(result, ensure_ascii=False, indent=2)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(summary, encoding="utf-8")
        return summary
    finally:
        await neo4j.close()


# ---------------------------------------------------------------------------
# Intent listing
# ---------------------------------------------------------------------------


async def _list_available_intents_from_neo4j(driver: Any) -> None:
    """Query Neo4j directly for distinct intents from Intent nodes (with summary)."""
    query = """
        MATCH (i:Intent)
        RETURN i.key AS key, i.summary AS summary
        ORDER BY i.key
    """
    async with driver.session() as session:
        result = await session.run(query)
        records = [r async for r in result]

    if not records:
        print("No intents found in Neo4j.")
        return

    print(f"\nAvailable intents ({len(records)}):\n")
    for r in records:
        key = r.get("key", "")
        summary = r.get("summary", "")
        display = f"{key}  — {summary}" if summary else key
        print(f"  {display}")


# ---------------------------------------------------------------------------
# Interactive path selection
# ---------------------------------------------------------------------------


def _interactive_select(paths: list[list[Any]]) -> list[Any] | None:
    """List paths and let user choose."""
    print(f"\nFound {len(paths)} paths for this intent:\n")
    for i, path in enumerate(paths, 1):
        action_count = sum(1 for e in path if e.action and str(e.action) not in ("ActionType.NAVIGATE",))
        start = path[0].source_url if path else "?"
        print(f"  [{i}] {len(path)} edges, {action_count} actions  from={start[:80]}")
        for e in path:
            tag = _action_tag(e)
            print(f"       {tag}")

    while True:
        try:
            raw = input(f"\nSelect path [1-{len(paths)}] or q: ").strip()
            if raw.lower() == "q":
                return None
            idx = int(raw) - 1
            if 0 <= idx < len(paths):
                return paths[idx]
        except (ValueError, EOFError):
            pass
        print("Invalid selection.")


def _action_tag(edge: Any) -> str:
    action = str(edge.action) if edge.action else "?"
    sel = edge.selector or "?"
    label = ""
    if edge.intent and hasattr(edge.intent, "summary"):
        label = edge.intent.summary or edge.intent.key or ""
    fp = ""
    if edge.frame_path:
        fp = f" [iframe:{len(edge.frame_path)}]"
    return f"  {action} {sel[:50]} {label}{fp}"


# ---------------------------------------------------------------------------
# Action-only replay engine
# ---------------------------------------------------------------------------


async def _replay_actions_only(
    edges: list[Any],
    start_url: str,
    headless: bool,
) -> dict:
    """Replay only click/fill/select actions, skipping navigation/tab actions.

    Supports multi-iframe nesting via edge.frame_path.
    """
    from browser_use.browser.session import BrowserSession as Browser

    from graph_agent.models import ActionType, FrameLocatorSnapshot
    from graph_agent.playback.engine.helpers import _ensure_frame_attached

    result: dict[str, Any] = {
        "success": True,
        "total": len(edges),
        "executed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }

    channel = (os.getenv("PLAYWRIGHT_CHANNEL") or os.getenv("MAPPING_CHANNEL") or "").strip() or None
    headless_env = (os.getenv("PLAYWRIGHT_HEADLESS") or "").strip().lower()
    if headless_env:
        headless = headless_env in ("1", "true", "yes", "on")

    window_size_raw = (os.getenv("BROWSER_WINDOW_SIZE") or "1920x1080").strip()
    try:
        w, h = window_size_raw.split("x", 1)
        window_size = {"width": int(w), "height": int(h)}
    except (ValueError, TypeError):
        window_size = {"width": 1920, "height": 1080}

    browser = Browser(
        headless=headless,
        args=["--incognito"],
        channel=channel,
        window_size=window_size,
        minimum_wait_page_load_time=0.1,
        wait_between_actions=0.1,
    )

    try:
        await browser.start()
        page = await browser.must_get_current_page()
        await page.goto(start_url)
        await asyncio.sleep(1)

        for edge in edges:
            action_str = str(edge.action) if edge.action else ""
            action = edge.action

            # Skip navigation and tab actions
            if action in (ActionType.NAVIGATE, None):
                result["skipped"] += 1
                continue
            if hasattr(action, "value") and action.value in ("tab_open", "tab_switch", "tab_close"):
                result["skipped"] += 1
                continue

            # Resolve frame context
            frame_path: list[FrameLocatorSnapshot] = []
            if edge.frame_path:
                try:
                    raw_fp = edge.frame_path
                    if isinstance(raw_fp, str):
                        raw_fp = json.loads(raw_fp)
                    if isinstance(raw_fp, list):
                        frame_path = [
                            FrameLocatorSnapshot(**f) if isinstance(f, dict) else f
                            for f in raw_fp
                        ]
                except Exception:
                    pass

            # Wait for iframe if needed
            if frame_path:
                try:
                    await _ensure_frame_attached(page, frame_path, 5000)
                except Exception as e:
                    logger.warning("Frame not attached: %s", e)

            # Execute action
            try:
                await _execute_single_action(
                    page, edge, frame_path, action_str
                )
                result["executed"] += 1
            except Exception as e:
                logger.warning("Action failed: %s edge=%s", e, getattr(edge, "edge_id", "?"))
                result["failed"] += 1
                result["errors"].append({
                    "edge_id": getattr(edge, "edge_id", ""),
                    "selector": edge.selector,
                    "action": action_str,
                    "error": str(e),
                })

        result["success"] = result["failed"] == 0
    finally:
        try:
            await browser.kill()
        except Exception:
            await browser.stop()

    return result


async def _execute_single_action(
    page: Any,
    edge: Any,
    frame_path: list[Any],
    action_str: str,
) -> None:
    """Execute a single click/fill/select action with selector fallbacks."""
    from graph_agent.models import ActionType
    from graph_agent.playback.engine.helpers import _try_action_with_selector_fallback

    async def _do_action(locator: Any) -> None:
        action = edge.action
        if action == ActionType.CLICK:
            await locator.click(timeout=5000)
        elif action == ActionType.FILL:
            value = edge.action_value or "test_value"
            await locator.fill(str(value), timeout=5000)
        elif action == ActionType.SELECT:
            value = edge.action_value or ""
            await locator.select_option(str(value), timeout=5000)
        else:
            logger.info("Unsupported action type: %s", action_str)

    await _try_action_with_selector_fallback(
        page_for_edge=page,
        edge=edge,
        action_fn=_do_action,
    )

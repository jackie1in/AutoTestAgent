"""Run browser-use Agent for mapping: explore flow and save to Neo4j."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from browser_use.browser.session import BrowserSession as Browser
from browser_use.llm.base import BaseChatModel

from graph_agent.cartography import browser_lifecycle
from graph_agent.cartography import config as cartography_config
from graph_agent.cartography.manual_capture import ManualCaptureSession
from graph_agent.cartography.mapping_pipeline import (
    rank_warm_start_candidates as orch_rank_warm_start_candidates,
)
from graph_agent.cartography.mapping_pipeline import (
    run_orchestrated_mapping,
)
from graph_agent.cartography.persistence import persist_mapping_result
from graph_agent.cartography.types import (
    EvidenceBundleItem,
)
from graph_agent.lib.observability import initialize_laminar, observe
from graph_agent.llm import get_llm

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

# Global registry for active browser sessions (for cleanup on Ctrl+C)
_active_browsers: list[Browser] = []
_shutdown_requested = False


def _register_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.register_browser(browser)


def _unregister_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.unregister_browser(browser)


async def _cleanup_all_browsers() -> None:
    """Compatibility wrapper."""
    await browser_lifecycle.cleanup_all_browsers()


# Register signal handlers
browser_lifecycle.install_signal_handlers()


@asynccontextmanager
async def managed_browser(browser: Browser):
    """Context manager for browser lifecycle with cleanup on exit."""
    _register_browser(browser)
    try:
        yield browser
    finally:
        try:
            # Use kill() API for forceful cleanup (browser-use recommended)
            if hasattr(browser, "kill"):
                await browser.kill()
            elif hasattr(browser, "stop"):
                await browser.stop()
            elif hasattr(browser, "close"):
                await browser.close()  # type: ignore[union-attr]
        except Exception as e:
            print(f"[WARN] Browser cleanup error: {e}")
        finally:
            _unregister_browser(browser)


def _resolve_mapping_url(url: str | None) -> str:
    return cartography_config.resolve_mapping_url(url)


def _resolve_mapping_headless() -> bool:
    return cartography_config.resolve_mapping_headless()


def _resolve_mapping_channel() -> str | None:
    return cartography_config.resolve_mapping_channel()


def _setup_browser_use_timeouts():
    cartography_config.setup_browser_use_timeouts()


def _resolve_knowledge_on_demand_enabled() -> bool:
    return cartography_config.resolve_knowledge_on_demand_enabled()


def _resolve_knowledge_min_interval_sec() -> float:
    return cartography_config.resolve_knowledge_min_interval_sec()


def _resolve_knowledge_trigger_score_threshold() -> float:
    return cartography_config.resolve_knowledge_trigger_score_threshold()


def _resolve_knowledge_trigger_profile() -> str:
    return cartography_config.resolve_knowledge_trigger_profile()


def _resolve_knowledge_query_timeout_ms() -> int:
    return cartography_config.resolve_knowledge_query_timeout_ms()


def _resolve_knowledge_topk() -> int:
    return cartography_config.resolve_knowledge_topk()


def _resolve_knowledge_release_id() -> str:
    return cartography_config.resolve_knowledge_release_id()


def _load_inventory(inventory_path: str | Path) -> list[dict]:
    return cartography_config.load_inventory(inventory_path)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def rank_warm_start_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    return orch_rank_warm_start_candidates(candidates)


async def _run_orchestrated_mapping(
    browser: Browser,
    llm: BaseChatModel,
    start_url: str,
    current_url: str,
    app_id: str,
    session_id: str,
    max_steps: int,
    time_budget_ms: int = 600_000,
    warm_start_candidates: list[dict[str, object]] | None = None,
    knowledge_on_demand_enabled: bool | None = None,
    knowledge_min_interval_sec: float | None = None,
    knowledge_trigger_score_threshold: float | None = None,
    knowledge_trigger_profile: str | None = None,
    knowledge_query_timeout_ms: int | None = None,
    knowledge_topk: int | None = None,
    knowledge_release_id: str = "",
) -> "CartographyResult":
    return await run_orchestrated_mapping(
        browser=browser,
        llm=llm,
        start_url=start_url,
        current_url=current_url,
        app_id=app_id,
        session_id=session_id,
        max_steps=max_steps,
        time_budget_ms=time_budget_ms,
        warm_start_candidates=warm_start_candidates,
        knowledge_on_demand_enabled=knowledge_on_demand_enabled,
        knowledge_min_interval_sec=knowledge_min_interval_sec,
        knowledge_trigger_score_threshold=knowledge_trigger_score_threshold,
        knowledge_trigger_profile=knowledge_trigger_profile,
        knowledge_query_timeout_ms=knowledge_query_timeout_ms,
        knowledge_topk=knowledge_topk,
        knowledge_release_id=knowledge_release_id,
    )


@observe(
    name="cartography.run_mapping",
    metadata={"component": "cartography", "stage": "entry"},
)
async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,  # Kept for API compatibility, ignored
    task: str | None = None,
    max_steps: int = 100,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,  # Kept for API compatibility, ignored
) -> str:
    """Run LLM-first orchestrated exploration and store results in Neo4j.

    - url: Start URL (required via arg or MAPPING_URL env).
    - output_path: Kept for API compatibility, data now stored in Neo4j.
    - task: Kept for API compatibility, no longer used by the pipeline.
    - max_steps: Maximum agent steps.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the app_id of the newly created app in Neo4j.
    """
    if inventory_path is None or not str(inventory_path).strip():
        raise ValueError(
            "inventory_path is required. Run scout first, then pass --inventory to mapping."
        )
    initialize_laminar()
    inventory = _load_inventory(inventory_path)

    from browser_use import Browser

    resolved_url = _resolve_mapping_url(url)
    pkg_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        # Resolve default relative to package: graph_agent/data/graph.json
        output_path = str(pkg_root / "data" / "graph.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    channel = _resolve_mapping_channel()
    base_headless = _resolve_mapping_headless()
    base_args = ["--incognito"]  # Force incognito mode: no session cache
    if channel:
        try:
            browser = Browser(headless=base_headless, args=base_args, channel=channel)
        except TypeError:
            # Older browser-use versions may not support channel keyword.
            browser = Browser(headless=base_headless, args=base_args)
    else:
        browser = Browser(headless=base_headless, args=base_args)

    async with managed_browser(browser):
        llm = get_llm()
        print("[RUNNER] === Pre-navigation phase starting ===")
        initial_actions_log: list[dict[str, object]] = []

        # Navigate to the target URL and wait for the page to settle.
        # Login (including captcha) is handled entirely by the LLM explorer
        # via prompt injection (build_login_hint_from_env).
        current_url = resolved_url
        print(f"[PreNav] Starting browser and navigating to {resolved_url}...")
        try:
            await browser.start()
            await browser.navigate_to(resolved_url)
            current_url = await browser.get_current_page_url() or resolved_url
            # Give the page time to load / redirect.
            await asyncio.sleep(4)
            current_url = await browser.get_current_page_url() or current_url
        except Exception as e:
            print(f"[WARN] Pre-navigation failed: {e}. Agent will attempt recovery.")

        print(f"[PreNav] Browser is now at {current_url}.")

        # Generate app_id and session_id early (needed by the pipeline and Neo4j)
        from urllib.parse import urlparse

        parsed = urlparse(resolved_url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        app_name = domain or "unknown"
        app_id = (
            f"app:{app_name}:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        session_id = f"session:{app_id}:{datetime.now(timezone.utc).isoformat()}"
        warm_start_candidates: list[dict[str, object]] = []
        try:
            from graph_agent.neo4j_client.manager import GraphManager

            async with GraphManager() as warm_manager:
                warm_start_candidates = await warm_manager._run_read(
                    """
                    MATCH (a:App {name: $app_name})
                    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                    OPTIONAL MATCH (target)-[:HAS_ZONE]->(z:Zone)
                    RETURN t.id AS transition_id,
                           coalesce(t.confidence, 0.0) AS confidence,
                           target.url AS target_url,
                           count(z) = 0 AS zone_unexplored
                    ORDER BY confidence DESC
                    LIMIT 200
                    """,
                    app_name=app_name,
                )
        except Exception as e:
            print(f"[PIPELINE] Warm-start candidate query skipped: {e}")

        knowledge_on_demand_enabled = _resolve_knowledge_on_demand_enabled()
        knowledge_min_interval_sec = _resolve_knowledge_min_interval_sec()
        knowledge_trigger_score_threshold = _resolve_knowledge_trigger_score_threshold()
        knowledge_trigger_profile = _resolve_knowledge_trigger_profile()
        knowledge_query_timeout_ms = _resolve_knowledge_query_timeout_ms()
        knowledge_topk = _resolve_knowledge_topk()
        knowledge_release_id = _resolve_knowledge_release_id()

        # Check for shutdown request before running
        if browser_lifecycle.is_shutdown_requested():
            print("[INFO] Shutdown requested before agent run, cleaning up...")
            return ""

        print("[RUNNER] === LLM-first orchestrated mode ===")
        try:
            result = await _run_orchestrated_mapping(
                browser=browser,
                llm=llm,
                start_url=resolved_url,
                current_url=current_url or resolved_url,
                app_id=app_id,
                session_id=session_id,
                max_steps=max_steps,
                warm_start_candidates=warm_start_candidates,
                knowledge_on_demand_enabled=knowledge_on_demand_enabled,
                knowledge_min_interval_sec=knowledge_min_interval_sec,
                knowledge_trigger_score_threshold=knowledge_trigger_score_threshold,
                knowledge_trigger_profile=knowledge_trigger_profile,
                knowledge_query_timeout_ms=knowledge_query_timeout_ms,
                knowledge_topk=knowledge_topk,
                knowledge_release_id=knowledge_release_id,
            )
        except asyncio.CancelledError:
            print("[INFO] Orchestrated exploration cancelled, cleaning up...")
            raise

    await persist_mapping_result(
        app_id=app_id,
        app_name=app_name,
        session_id=session_id,
        resolved_url=resolved_url,
        current_url=current_url or resolved_url,
        inventory=inventory,
        initial_actions_log=initial_actions_log,
        result=result,
    )
    return app_id


async def run_manual_mapping(
    *,
    app_name: str,
    session_id: str,
    events: list[dict[str, object]],
    mode: str = "manual_raw",
    app_id: str | None = None,
    operator_id: str = "human:operator",
    start_state_hint: str = "",
    start_url: str = "",
) -> str:
    """Persist manual capture events through the same cartography pipeline."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resolved_app_id = app_id or f"app:{app_name}:{ts}"
    if mode == "manual_graph_assisted":
        manual_session = ManualCaptureSession.from_graph_context(
            session_id=session_id,
            operator_id=operator_id,
            app_id=resolved_app_id,
            start_state_hint=start_state_hint or start_url,
            start_url=start_url,
        )
    else:
        manual_session = ManualCaptureSession.from_raw(
            session_id=session_id,
            operator_id=operator_id,
            start_url=start_url,
        )
    for event in events:
        raw_evidence = event.get("evidence_bundle")
        evidence_bundle = raw_evidence if isinstance(raw_evidence, list) else []
        manual_session.record_action(
            action=str(event.get("action") or "unknown"),
            selector=str(event.get("selector") or "[manual]"),
            url_before=str(event.get("url_before") or start_url),
            url_after=str(event.get("url_after") or start_url),
            thought=str(event.get("thought") or ""),
            action_value=str(event.get("action_value") or ""),
            param_name=str(event.get("param_name") or ""),
            confidence_hint=_as_float(event.get("confidence_hint") or 0.9, 0.9),
            evidence_bundle=cast(list[EvidenceBundleItem], evidence_bundle),
            from_state_hint=str(event.get("from_state_hint") or ""),
            to_state_hint=str(event.get("to_state_hint") or ""),
        )
    result = await manual_session.to_cartography_result()
    await persist_mapping_result(
        app_id=resolved_app_id,
        app_name=app_name,
        session_id=session_id,
        resolved_url=start_url,
        current_url=start_url,
        inventory=[],
        initial_actions_log=[],
        result=result,
    )
    return resolved_app_id


def main() -> None:
    """CLI entry: run scout then mapping. Scout writes inventory, mapping uses it to build graph."""
    import argparse

    from dotenv import load_dotenv

    # Try loading from current directory first, then fallback to project root
    if not load_dotenv():
        # Fallback: try to find .env in project root
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded .env from {env_path}")
        else:
            print(f"Warning: .env not found at {env_path} or current directory")

    # Setup browser-use timeouts from MAPPING_TIMEOUT env
    _setup_browser_use_timeouts()
    initialize_laminar()

    pkg_root = Path(__file__).resolve().parent.parent
    default_output = str(pkg_root / "data" / "graph.json")
    default_inventory = str(pkg_root / "data" / "element_inventory.json")

    parser = argparse.ArgumentParser(
        description="Run scout (list page elements) then mapping (explore flow, build graph)."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MAPPING_URL", ""),
        help="Start URL for scout and mapping. If omitted, uses MAPPING_URL.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MAPPING_OUTPUT", default_output),
        help="Output graph JSON path (default: graph_agent/data/graph.json)",
    )
    parser.add_argument(
        "--inventory",
        default=os.getenv("MAPPING_INVENTORY", default_inventory),
        help="Scout inventory JSON path (default: graph_agent/data/element_inventory.json)",
    )
    parser.add_argument(
        "--scout-pages",
        default=os.getenv("SCOUT_PAGES", ""),
        help="Comma-separated extra pages for multi-page scout aggregation. Supports relative paths or absolute URLs.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly mapped graph with existing output file.",
    )
    parser.add_argument(
        "--mode",
        default=os.getenv("MAPPING_MODE", "auto"),
        choices=["auto", "manual_graph_assisted", "manual_raw"],
        help="Run mode: auto (default), manual_graph_assisted, manual_raw.",
    )
    parser.add_argument(
        "--manual-events",
        default=os.getenv("MAPPING_MANUAL_EVENTS", ""),
        help="Path to manual capture events JSON for manual modes.",
    )
    parser.add_argument(
        "--manual-operator",
        default=os.getenv("MAPPING_MANUAL_OPERATOR", "human:operator"),
        help="Operator id for manual mode.",
    )
    parser.add_argument(
        "--manual-start-state",
        default=os.getenv("MAPPING_MANUAL_START_STATE", ""),
        help="Start state hint for manual_graph_assisted mode.",
    )
    parser.add_argument(
        "--manual-session-id",
        default=os.getenv("MAPPING_MANUAL_SESSION_ID", ""),
        help="Optional manual session id. Auto generated when omitted.",
    )
    args = parser.parse_args()
    output = (args.output or "").strip() or default_output
    inventory = (args.inventory or "").strip() or default_inventory
    scout_pages_arg = (args.scout_pages or "").strip()
    _scout_pages = [item.strip() for item in scout_pages_arg.split(",") if item.strip()]
    merge_existing = bool(args.merge_existing)
    run_mode = (args.mode or "auto").strip()
    url = _resolve_mapping_url(args.url)

    async def _run() -> None:
        if browser_lifecycle.is_shutdown_requested():
            print("[INFO] Shutdown requested, exiting before mapping...")
            return

        # Scout module removed; create empty inventory for backward compat
        import json

        if not Path(inventory).exists():
            Path(inventory).parent.mkdir(parents=True, exist_ok=True)
            Path(inventory).write_text(json.dumps({"elements": []}), encoding="utf-8")

        if run_mode == "auto":
            print("Mapping (explore flow, build graph)...")
            try:
                _ = await run_mapping(
                    url=url,
                    output_path=output,
                    inventory_path=inventory,
                    merge_existing=merge_existing,
                )
            except asyncio.CancelledError:
                print("[INFO] Mapping cancelled")
                raise
            return

        manual_events_path = (args.manual_events or "").strip()
        if not manual_events_path:
            raise ValueError("--manual-events is required in manual mode.")
        manual_events_file = Path(manual_events_path)
        if not manual_events_file.exists():
            raise FileNotFoundError(
                f"Manual events file not found: {manual_events_file}"
            )
        raw_manual = json.loads(manual_events_file.read_text(encoding="utf-8"))
        events = (
            raw_manual if isinstance(raw_manual, list) else raw_manual.get("events", [])
        )
        if not isinstance(events, list):
            raise ValueError('manual events must be a JSON list or {"events": [...]}')
        manual_session_id = (args.manual_session_id or "").strip() or (
            f"session:manual:{datetime.now(timezone.utc).isoformat()}"
        )
        from urllib.parse import urlparse

        parsed = urlparse(url)
        app_name = (parsed.netloc or parsed.path.split("/")[0] or "manual").strip()
        _ = await run_manual_mapping(
            app_name=app_name,
            session_id=manual_session_id,
            events=events,
            mode=run_mode,
            operator_id=(args.manual_operator or "human:operator").strip(),
            start_state_hint=(args.manual_start_state or "").strip(),
            start_url=url,
        )

    # Use a custom event loop to handle signals and cleanup properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Add signal handlers for the main thread
    def _sigint_handler():
        """Handle Ctrl+C in the main thread."""
        global _shutdown_requested
        _shutdown_requested = True
        browser_lifecycle.request_shutdown()
        print("\n[INFO] SIGINT received, shutting down gracefully...")
        # Cancel all tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()

    # Use asyncio's signal handling (works on Unix and Windows with ProactorEventLoop)
    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
        loop.add_signal_handler(signal.SIGTERM, _sigint_handler)
    except NotImplementedError:
        # Fallback for Windows with SelectorEventLoop
        signal.signal(signal.SIGINT, lambda s, f: _sigint_handler())
        signal.signal(signal.SIGTERM, lambda s, f: _sigint_handler())

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, initiating shutdown...")
    except asyncio.CancelledError:
        print("\n[INFO] Tasks cancelled, cleaning up...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise
    finally:
        # Clean up browsers
        try:
            loop.run_until_complete(_cleanup_all_browsers())
            print("[INFO] Browser cleanup complete")
        except Exception as cleanup_error:
            print(f"[WARN] Cleanup error: {cleanup_error}")

        # Remove signal handlers
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass

        loop.close()
        print("[INFO] Shutdown complete")
        sys.exit(130 if browser_lifecycle.is_shutdown_requested() else 0)


if __name__ == "__main__":
    main()

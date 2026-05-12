from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from graph_agent.cartography import browser_lifecycle
from graph_agent.cartography.runner.mapping_ops import (
    _cleanup_all_browsers,
    _resolve_mapping_url,
    run_mapping,
    run_manual_mapping,
    run_post_mapping_maintenance,
)

logger = logging.getLogger(__name__)

_shutdown_requested = False

load_dotenv()


def main() -> None:
    """CLI entry: run scout then mapping. Scout writes inventory, mapping uses it to build graph."""
    from urllib.parse import urlparse

    from browser_use.utils import logger as bu_logger

    bu_logger.setLevel(logging.WARNING)

    default_output = os.getenv("MAPPING_OUTPUT", "mapping_output.json")
    default_inventory = os.getenv("MAPPING_INVENTORY", "mapping_inventory.json")

    parser = argparse.ArgumentParser(
        description="Map a web application: explore UI flow and build a Neo4j knowledge graph."
    )
    parser.add_argument("--url", default="", help="Override the start URL (default: env MAPPING_URL).")
    parser.add_argument(
        "--no-scout",
        action="store_true",
        help="Skip scouting and reuse existing inventory file.",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Path to save the mapping result JSON (default: {default_output}).",
    )
    parser.add_argument(
        "--inventory",
        default=default_inventory,
        help=f"Path to scouting inventory file (default: {default_inventory}).",
    )
    parser.add_argument(
        "--scout-pages",
        default="",
        help="Comma-separated initial page URLs to scout.",
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
    parser.add_argument(
        "--maintenance-after-run",
        action="store_true",
        help="Run semantic gate + retention flow after mapping.",
    )
    parser.add_argument(
        "--maintenance-retention-execute",
        action="store_true",
        help="Execute retention in maintenance flow (default is dry-run).",
    )
    parser.add_argument(
        "--maintenance-keep-releases",
        type=int,
        default=int(os.getenv("CARTOGRAPHY_RETENTION_KEEP_RELEASES", "5")),
        help="Retention keep release count used by maintenance flow.",
    )
    parser.add_argument(
        "--maintenance-min-age-days",
        type=int,
        default=int(os.getenv("CARTOGRAPHY_RETENTION_MIN_AGE_DAYS", "14")),
        help="Retention minimum age days used by maintenance flow.",
    )
    parser.add_argument(
        "--maintenance-batch-size",
        type=int,
        default=int(os.getenv("CARTOGRAPHY_RETENTION_BATCH_SIZE", "500")),
        help="Retention batch size used by maintenance flow.",
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
            logger.info("[RUNNER] Shutdown requested, exiting before mapping...")
            return

        if not Path(inventory).exists():
            Path(inventory).parent.mkdir(parents=True, exist_ok=True)
            Path(inventory).write_text(json.dumps({"elements": []}), encoding="utf-8")

        if run_mode == "auto":
            logger.info("Mapping (explore flow, build graph)...")
            try:
                app_id = await run_mapping(
                    url=url,
                    output_path=output,
                    inventory_path=inventory,
                    merge_existing=merge_existing,
                )
                if args.maintenance_after_run:
                    maintenance_report = await run_post_mapping_maintenance(
                        app_id=app_id,
                        retention_execute=bool(args.maintenance_retention_execute),
                        keep_releases=max(1, int(args.maintenance_keep_releases)),
                        min_age_days=max(0, int(args.maintenance_min_age_days)),
                        batch_size=max(1, int(args.maintenance_batch_size)),
                    )
                    logger.info(
                        "[RUNNER] maintenance report:\n"
                        + json.dumps(maintenance_report, ensure_ascii=False, indent=2)
                    )
            except asyncio.CancelledError:
                logger.info("[RUNNER] Mapping cancelled")
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sigint_handler():
        global _shutdown_requested
        _shutdown_requested = True
        browser_lifecycle.request_shutdown()
        logger.info("[RUNNER] SIGINT received, shutting down gracefully...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
        loop.add_signal_handler(signal.SIGTERM, _sigint_handler)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _sigint_handler())
        signal.signal(signal.SIGTERM, lambda s, f: _sigint_handler())

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        logger.info("[RUNNER] KeyboardInterrupt received, initiating shutdown...")
    except asyncio.CancelledError:
        logger.info("[RUNNER] Tasks cancelled, cleaning up...")
    except Exception as e:
        logger.exception("[RUNNER] Unexpected error: %s", e)
        raise
    finally:
        try:
            loop.run_until_complete(_cleanup_all_browsers())
            logger.info("[RUNNER] Browser cleanup complete")
        except Exception as cleanup_error:
            logger.warning("[RUNNER] Cleanup error: %s", cleanup_error)

        try:
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass
        try:
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass
        loop.close()

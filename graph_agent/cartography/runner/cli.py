from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import queue
import signal
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

# Prevent browser-use from hijacking root logger on import.
# We configure async-safe logging ourselves in _configure_logging().
os.environ["BROWSER_USE_SETUP_LOGGING"] = "false"

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


def _configure_logging() -> None:
    """Async-safe logging setup via QueueHandler + QueueListener.

    browser-use's default setup_logging() is skipped (BROWSER_USE_SETUP_LOGGING=false),
    so we install a QueueHandler on the root logger.  All log records pass through an
    in-process queue; a dedicated thread serialises writes to stderr, guaranteeing
    that each log line is atomic even when multiple asyncio tasks log concurrently.
    """
    raw = (os.getenv("LOG_LEVEL") or "").strip().upper()
    level: int = getattr(logging, raw, None) if raw else None  # type: ignore[assignment]
    if not isinstance(level, int):
        level = logging.INFO

    fmt = logging.Formatter("%(levelname)-8s [%(name)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    _log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(QueueHandler(_log_queue))

    _listener = QueueListener(_log_queue, console)
    _listener.start()
    atexit.register(_listener.stop)

    # Per-logger defaults: browser-use is verbose, suppress unless LOG_LEVEL is set
    if raw:
        logging.getLogger("browser_use").setLevel(level)
    else:
        logging.getLogger("browser_use").setLevel(logging.WARNING)

    logging.getLogger("graph_agent").setLevel(level)


def main() -> None:
    """CLI entry: run scout then mapping. Scout writes inventory, mapping uses it to build graph."""
    from urllib.parse import urlparse

    _configure_logging()

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
        choices=["auto", "manual_graph_assisted", "manual_raw", "replay-intent"],
        help="Run mode: auto (default), manual_graph_assisted, manual_raw, replay-intent.",
    )
    parser.add_argument(
        "--intent",
        default="",
        help="Intent key for replay-intent mode.",
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

        if run_mode == "replay-intent":
            from graph_agent.cartography.runner.replay_ops import run_replay_intent

            intent_key = (args.intent or "").strip()
            if not intent_key:
                intent_key = input("Enter intent key to replay: ").strip()
            if not intent_key:
                raise ValueError("--intent is required for replay-intent mode.")
            summary = await run_replay_intent(
                intent_key,
                start_url=url,
                output_path=output,
                headless=True,
            )
            logger.info("[RUNNER] Replay result:\n%s", summary)
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

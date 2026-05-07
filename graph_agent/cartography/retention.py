from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

from graph_agent.neo4j_client.manager import GraphManager

logger = logging.getLogger(__name__)


async def _run_retention(
    *,
    app_id: str,
    keep_releases: int,
    min_age_days: int,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    async with GraphManager() as manager:
        if dry_run:
            report = await manager.evaluate_retention_plan(
                app_id=app_id,
                keep_releases=keep_releases,
                min_age_days=min_age_days,
            )
        else:
            report = await manager.run_retention(
                app_id=app_id,
                keep_releases=keep_releases,
                min_age_days=min_age_days,
                batch_size=batch_size,
            )
    report["dry_run"] = dry_run
    report["duration_ms"] = int((time.time() - started) * 1000)
    warnings: list[str] = []
    active_releases = int(
        (report.get("protected_counts") or {}).get("active_releases") or 0
    )
    if active_releases != 1:
        warnings.append(f"active_release_count_expected_1_got_{active_releases}")
    report["warnings"] = warnings
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retention for graph cartography data by app_id."
    )
    parser.add_argument("--app-id", required=True, help="Stable app id, e.g. app:demo")
    parser.add_argument(
        "--keep-releases",
        type=int,
        default=5,
        help="How many latest releases to keep.",
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=14,
        help="Only clean data older than this age.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for delete operations.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate retention candidates only (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute retention cleanup.",
    )
    args = parser.parse_args()

    dry_run = not bool(args.execute)
    report = asyncio.run(
        _run_retention(
            app_id=str(args.app_id).strip(),
            keep_releases=max(1, int(args.keep_releases)),
            min_age_days=max(0, int(args.min_age_days)),
            dry_run=dry_run,
            batch_size=max(1, int(args.batch_size)),
        )
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    logger.info("%s", payload)
    sys.stdout.write(payload + "\n")
    if dry_run:
        sys.exit(0)
    has_warnings = bool(report.get("warnings"))
    sys.exit(1 if has_warnings else 0)


if __name__ == "__main__":
    main()

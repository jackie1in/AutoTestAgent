from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from graph_agent.neo4j_client.manager import GraphManager

logger = logging.getLogger(__name__)


async def _run_gate(
    *,
    app_id: str,
    window: int,
    min_score: float,
    avg_score: float,
    consecutive_pass_required: int,
    default_threshold: float,
) -> dict[str, object]:
    async with GraphManager() as manager:
        result = await manager.evaluate_semantic_stability_gate(
            app_id=app_id,
            window=window,
            min_score=min_score,
            avg_score=avg_score,
            consecutive_pass_required=consecutive_pass_required,
            default_threshold=default_threshold,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic stability gate for an app."
    )
    parser.add_argument("--app-id", required=True, help="Stable app id, e.g. app:demo")
    parser.add_argument("--window", type=int, default=10, help="Recent session window")
    parser.add_argument("--min-score", type=float, default=90.0, help="Min score gate")
    parser.add_argument("--avg-score", type=float, default=92.0, help="Avg score gate")
    parser.add_argument(
        "--consecutive-pass-required",
        type=int,
        default=3,
        help="Required consecutive passed sessions",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=90.0,
        help="Fallback threshold when session field missing",
    )
    args = parser.parse_args()

    result = asyncio.run(
        _run_gate(
            app_id=args.app_id.strip(),
            window=max(1, int(args.window)),
            min_score=float(args.min_score),
            avg_score=float(args.avg_score),
            consecutive_pass_required=max(1, int(args.consecutive_pass_required)),
            default_threshold=float(args.default_threshold),
        )
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    logger.info("%s", payload)
    sys.stdout.write(payload + "\n")
    sys.exit(0 if bool(result.get("passed")) else 1)


if __name__ == "__main__":
    main()

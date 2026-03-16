"""Auto repair loop: diagnose -> re-infer -> replay."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from graph_agent.acceptance.failure_chain import classify_root_cause_detail
from graph_agent.acceptance.playback_diagnostics import collect_playback_diagnostics
from graph_agent.graph.io import load_graph
from graph_agent.mapping.run import re_infer_with_feedback
from graph_agent.run_e2e_acceptance import _get_start_url_from_graph

REPAIRABLE_ROOT_CAUSES: set[str] = {"selector", "dependency", "async_load"}

_METRIC_KEYS = (
    "intent_missing_count",
    "intent_success_rate",
    "semantic_consistency_rate",
)


def _snapshot_graph_metrics(graph_path: Path) -> dict[str, Any]:
    """Read a subset of graph metadata relevant for before/after comparison."""
    g = load_graph(graph_path)
    meta = g.graph if hasattr(g, "graph") else {}
    return {k: meta.get(k) for k in _METRIC_KEYS}


async def run_auto_repair_loop(
    graph_path: str | Path,
    intent_query: str,
    *,
    max_rounds: int = 2,
    start_url: str | None = None,
    test_data: dict[str, Any] | None = None,
    inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run diagnose -> re-infer -> replay loop with bounded retries."""
    graph_obj = Path(graph_path)
    if not graph_obj.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_obj}")
    if max_rounds < 1:
        max_rounds = 1

    graph = load_graph(graph_obj)
    resolved_start_url = start_url or _get_start_url_from_graph(graph) or ""
    failure_path = graph_obj.parent / "playback_failures.json"

    rounds: list[dict[str, Any]] = []
    for round_idx in range(1, max_rounds + 1):
        diagnostics = await collect_playback_diagnostics(
            graph_path=graph_obj,
            intent_query=intent_query,
            start_url=resolved_start_url,
            test_data=test_data or {},
            failure_output_path=failure_path,
        )
        playback = diagnostics.get("playback", {})
        root_cause_primary, root_cause_detail = classify_root_cause_detail(
            playback.get("error")
        )
        round_report: dict[str, Any] = {
            "round": round_idx,
            "playback_success": bool(playback.get("success", False)),
            "playback_error": playback.get("error"),
            "root_cause": root_cause_primary.value,
            "root_cause_detail": root_cause_detail,
            "failed_edge_id": playback.get("failed_edge_id"),
            "failed_step_index": playback.get("failed_step_index"),
        }
        if playback.get("success", False):
            round_report["repair"] = {"total": 0, "succeeded": 0, "failed": 0}
            rounds.append(round_report)
            return {
                "success": True,
                "rounds": rounds,
                "final_playback": playback,
                "failure_output_path": str(failure_path),
            }

        if root_cause_primary.value not in REPAIRABLE_ROOT_CAUSES:
            round_report["repair"] = {"total": 0, "succeeded": 0, "failed": 0}
            round_report["skip_reason"] = (
                f"root_cause '{root_cause_primary.value}' not in REPAIRABLE_ROOT_CAUSES"
            )
            rounds.append(round_report)
            break

        before_metrics = _snapshot_graph_metrics(graph_obj)

        repair_stats = await re_infer_with_feedback(
            graph_path=graph_obj,
            feedback=failure_path,
            inventory_path=inventory_path,
        )

        after_metrics = _snapshot_graph_metrics(graph_obj)
        delta: dict[str, Any] = {}
        for k in _METRIC_KEYS:
            b, a = before_metrics.get(k), after_metrics.get(k)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta[k] = round(a - b, 6)
            else:
                delta[k] = None

        round_report["repair"] = repair_stats
        round_report["before"] = before_metrics
        round_report["after"] = after_metrics
        round_report["delta"] = delta
        rounds.append(round_report)
        if repair_stats.get("total", 0) == 0:
            break

    final_diag = await collect_playback_diagnostics(
        graph_path=graph_obj,
        intent_query=intent_query,
        start_url=resolved_start_url,
        test_data=test_data or {},
        failure_output_path=failure_path,
    )
    return {
        "success": bool(final_diag.get("playback", {}).get("success", False)),
        "rounds": rounds,
        "final_playback": final_diag.get("playback", {}),
        "failure_output_path": str(failure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto repair loop: diagnose -> re-infer -> replay"
    )
    parser.add_argument(
        "--graph",
        default=str(Path(__file__).resolve().parent / "data" / "graph.json"),
        help="Graph JSON path",
    )
    parser.add_argument(
        "--intent",
        required=True,
        help="Intent query to diagnose and auto-repair",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum auto repair rounds",
    )
    parser.add_argument("--url", default="", help="Optional playback start URL")
    parser.add_argument(
        "--inventory",
        default="",
        help="Optional inventory path for acceptance snapshot updates",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_auto_repair_loop(
            graph_path=args.graph,
            intent_query=args.intent,
            max_rounds=args.max_rounds,
            start_url=args.url or None,
            inventory_path=args.inventory or None,
        )
    )
    print(result)
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()

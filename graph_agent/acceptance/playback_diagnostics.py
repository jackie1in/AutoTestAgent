"""Helpers for diagnosing real playback failures against a stored graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph_agent.acceptance.failure_chain import FailureRootCause, classify_root_cause_detail
from graph_agent.graph.io import load_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.playback.engine import run_playback
from graph_agent.run_e2e_acceptance import _get_start_url_from_graph


def _classify_root_cause(error: str | None) -> FailureRootCause:
    """Map playback error text to FailureRootCause."""
    text = str(error or "").lower()
    if text.startswith("tab:"):
        return FailureRootCause.TAB
    if text.startswith("iframe:"):
        return FailureRootCause.IFRAME
    if text.startswith("async_load:"):
        return FailureRootCause.ASYNC_LOAD
    if text.startswith("selector:"):
        return FailureRootCause.SELECTOR
    return FailureRootCause.DEPENDENCY


def _default_failure_path(graph_path: str | Path) -> Path:
    graph_obj = Path(graph_path)
    return graph_obj.parent / "playback_failures.json"


def _append_failure_record(path: Path, record: dict[str, Any]) -> None:
    """Append a failure record into playback_failures.json."""
    payload: dict[str, Any] = {"failures": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {"failures": []}
    failures = payload.get("failures")
    if not isinstance(failures, list):
        failures = []
    failures.append(record)
    payload["failures"] = failures
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def collect_playback_diagnostics(
    graph_path: str | Path,
    intent_query: str,
    start_url: str | None = None,
    test_data: dict[str, Any] | None = None,
    failure_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a query to a path and capture the paired playback result."""
    graph = load_graph(graph_path)
    path = get_path_from_query(intent_query, graph)
    playback_result = await run_playback(
        path,
        test_data=test_data or {},
        start_url=start_url or _get_start_url_from_graph(graph) or "",
    )
    failed_step_index = playback_result.get("failed_step_index")
    failed_edge_id = playback_result.get("failed_edge_id")
    root_cause = _classify_root_cause(playback_result.get("error"))
    root_cause_obj, root_cause_detail = classify_root_cause_detail(
        playback_result.get("error")
    )
    report = {
        "intent_query": intent_query,
        "path_length": len(path),
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "selector": edge.selector,
                "action": edge.action.value,
                "tab_id": edge.tab_id,
                "target_tab_id": edge.target_tab_id,
                "tab_action": edge.tab_action.value if edge.tab_action else None,
                "frame_path": [frame.selector for frame in edge.frame_path],
            }
            for edge in path
        ],
        "playback": playback_result,
        "root_cause": root_cause.value,
        "root_cause_detail": root_cause_detail,
        "failed_step_index": failed_step_index,
        "failed_edge_id": failed_edge_id,
    }
    if not playback_result.get("success", False):
        failure_record = {
            "intent_query": intent_query,
            "root_cause": root_cause.value,
            "root_cause_detail": root_cause_detail,
            "error": playback_result.get("error"),
            "failed_step_index": failed_step_index,
            "failed_edge_id": failed_edge_id,
            "path_length": len(path),
            "edges": report["edges"],
        }
        output_path = (
            Path(failure_output_path)
            if failure_output_path is not None
            else _default_failure_path(graph_path)
        )
        _append_failure_record(output_path, failure_record)
        report["failure_output_path"] = str(output_path)
    return report

"""Helpers for diagnosing real playback failures against a stored graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_agent.graph.io import load_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.playback.engine import run_playback
from graph_agent.run_e2e_acceptance import _get_start_url_from_graph


async def collect_playback_diagnostics(
    graph_path: str | Path,
    intent_query: str,
    start_url: str | None = None,
    test_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a query to a path and capture the paired playback result."""
    graph = load_graph(graph_path)
    path = get_path_from_query(intent_query, graph)
    playback_result = await run_playback(
        path,
        test_data=test_data or {},
        start_url=start_url or _get_start_url_from_graph(graph) or "",
    )
    return {
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
    }

from typing import Any

from graph_agent.models import GraphEdge
from graph_agent.web.app.config import DEFAULT_START_URL


def _resolve_start_url(edge_list: list[GraphEdge]) -> str:
    if edge_list:
        first = edge_list[0]
        if (
            first.source_url
            and isinstance(first.source_url, str)
            and first.source_url.startswith("http")
        ):
            return first.source_url
    return DEFAULT_START_URL


def _resolve_end_url(edge_list: list[GraphEdge]) -> str | None:
    if edge_list:
        last = edge_list[-1]
        if (
            last.target_url
            and isinstance(last.target_url, str)
            and last.target_url.startswith("http")
        ):
            return last.target_url
    return None


def _build_path_preview(edge_list: list[GraphEdge]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for edge in edge_list:
        intent_summary = ""
        if edge.intent:
            intent_summary = edge.intent.summary or edge.intent.key or ""
        steps.append(
            {
                "step_index": edge.step_index,
                "action": str(edge.action),
                "selector": edge.selector,
                "intent": intent_summary,
            }
        )
    return {"step_count": len(edge_list), "steps": steps}

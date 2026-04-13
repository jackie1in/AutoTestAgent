"""FastAPI app: GET /api/graph, GET /api/intents, POST /api/playback (SSE), GET /api/graph/stream (SSE)."""

import json
import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph_agent.graph.io import load_graph
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.models import Intent, ElementConstraints, GraphEdge
from graph_agent.playback.engine import run_playback

load_dotenv()

# graph_agent/web/app.py -> graph_agent -> data/graph.json
GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# MVP: fixed start/end URL for the-internet login flow
DEFAULT_START_URL = os.getenv("MAPPING_URL", "https://the-internet.herokuapp.com/login")
DEFAULT_EXPECTED_END_URL = (
    "https://the-internet.herokuapp.com/secure"  # or None to skip assertion
)

app = FastAPI(title="Graph Agent API")


class PlaybackRequest(BaseModel):
    """Request body for POST /api/playback."""

    intent: str
    test_data: dict = {}
    wait_for_network: bool = True


def _business_templates_to_json(G: Any) -> list[dict[str, Any]]:
    """Return frontend-friendly business template summaries with dependencies."""
    raw_templates = G.graph.get("business_templates", []) if hasattr(G, "graph") else []
    if not isinstance(raw_templates, list):
        return []

    templates: list[dict[str, Any]] = []
    for raw in raw_templates:
        if not isinstance(raw, dict):
            continue
        depends_on = raw.get("depends_on")
        templates.append(
            {
                "template_id": str(raw.get("template_id") or ""),
                "business_key": str(raw.get("business_key") or ""),
                "summary": str(raw.get("summary") or ""),
                "entry_node": str(raw.get("entry_node") or ""),
                "exit_node": str(raw.get("exit_node") or ""),
                "path_length": int(raw.get("path_length") or 0),
                "confidence": float(raw.get("confidence") or 0.0),
                "depends_on": [str(item) for item in depends_on]
                if isinstance(depends_on, list)
                else [],
            }
        )
    return templates


def _graph_to_json_dict(G):
    """Convert nx.DiGraph to API shape. Includes intent_failure_reason and diagnostic stats."""
    nodes = []
    for nid, data in G.nodes(data=True):
        node = {"id": str(nid)}
        if "label" in data:
            node["label"] = data["label"]
        if "url" in data:
            node["url"] = data["url"]
        if "title" in data:
            node["title"] = data["title"]
        nodes.append(node)

    edges = []
    missing_count = 0
    failure_reasons: set[str] = set()
    if hasattr(G, "is_multigraph") and G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
        tuples = [(u, v, k, data) for u, v, k, data in edge_iter]
    else:
        tuples = [(u, v, None, data) for u, v, data in G.edges(data=True)]

    for u, v, key, data in tuples:
        intent = data.get("intent")
        intent_dict = intent.model_dump() if isinstance(intent, Intent) else None
        if intent is None:
            missing_count += 1
            reason = data.get("intent_failure_reason")
            if reason:
                failure_reasons.add(reason)

        constraints = data.get("constraints")
        constraints_dict = (
            constraints.model_dump()
            if isinstance(constraints, ElementConstraints)
            else None
        )
        element = data.get("element")
        element_dict = (
            element.model_dump()
            if hasattr(element, "model_dump")
            else (element if isinstance(element, dict) else None)
        )

        edge = {
            "edge_id": data.get("edge_id") or (str(key) if key is not None else None),
            "step_index": data.get("step_index"),
            "source": str(u),
            "target": str(v),
            "source_url": data.get("source_url"),
            "target_url": data.get("target_url"),
            "selector": data.get("selector", ""),
            "action": data.get("action", ""),
            "intent": intent_dict,
            "context_level_used": data.get("context_level_used"),
            "intent_failure_reason": data.get("intent_failure_reason"),
            "param_name": data.get("param_name"),
            "action_value": data.get("action_value"),
            "element": element_dict,
            "constraints": constraints_dict,
        }
        edges.append(edge)
    metadata = {
        "filtered_non_ui_edges": G.graph.get("filtered_non_ui_edges", 0),
        "intent_missing_count": G.graph.get("intent_missing_count", missing_count),
        "intent_success_rate": G.graph.get("intent_success_rate"),
        "business_template_count": G.graph.get("business_template_count", 0),
        "business_template_generation_failures": G.graph.get(
            "business_template_generation_failures", 0
        ),
        "semantic_consistency_rate": G.graph.get("semantic_consistency_rate"),
        "inventory_non_empty_rate": G.graph.get("inventory_non_empty_rate"),
        "re_infer_success_rate": G.graph.get("re_infer_success_rate"),
        "runtime_non_ui_action_count": G.graph.get("runtime_non_ui_action_count"),
        "state_like_node_ratio": G.graph.get("state_like_node_ratio"),
        "business_intent_edge_ratio": G.graph.get("business_intent_edge_ratio"),
        "multi_edge_preserved_count": G.graph.get("multi_edge_preserved_count"),
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "missing_count": missing_count,
        "failure_reasons": sorted(failure_reasons),
        "business_templates": _business_templates_to_json(G),
        "metadata": metadata,
    }


@app.get("/api/graph")
def get_graph():
    """Load graph from graph_agent/data/graph.json. Return 200 with nodes/edges; if file missing return empty graph."""
    if not GRAPH_PATH.exists():
        return {
            "nodes": [],
            "edges": [],
            "missing_count": 0,
            "failure_reasons": [],
            "business_templates": [],
            "metadata": {
                "filtered_non_ui_edges": 0,
                "intent_missing_count": 0,
                "intent_success_rate": 1.0,
                "business_template_count": 0,
                "business_template_generation_failures": 0,
                "semantic_consistency_rate": 1.0,
                "inventory_non_empty_rate": 0.0,
                "re_infer_success_rate": None,
                "runtime_non_ui_action_count": 0,
                "state_like_node_ratio": 0.0,
                "business_intent_edge_ratio": 0.0,
                "multi_edge_preserved_count": 0,
            },
        }
    G = load_graph(GRAPH_PATH)
    return _graph_to_json_dict(G)


@app.get("/api/intents")
def get_intents():
    """Return template-first query options with backward-compatible fields."""
    if not GRAPH_PATH.exists():
        return []
    G = load_graph(GRAPH_PATH)
    options: list[dict[str, Any]] = []

    templates = G.graph.get("business_templates", [])
    if isinstance(templates, list):
        for raw in templates:
            if not isinstance(raw, dict):
                continue
            business_key = str(raw.get("business_key") or "")
            summary = str(raw.get("summary") or "")
            if not business_key and not summary:
                continue
            confidence = float(raw.get("confidence") or 0.0)
            value = business_key or summary
            label = (
                f"{business_key} - {summary}"
                if business_key and summary and business_key != summary
                else (summary or business_key)
            )
            options.append(
                {
                    "type": "template",
                    "value": value,
                    "key": business_key or None,
                    "summary": summary,
                    "confidence": confidence,
                    "label": label,
                    "path_length": int(raw.get("path_length") or 0),
                }
            )

    intents: dict[str, dict[str, Any]] = {}
    for _u, _v, data in G.edges(data=True):
        intent = data.get("intent")
        if intent is None:
            continue
        if isinstance(intent, Intent):
            key = intent.key or ""
            summary = intent.summary or intent.raw or ""
            confidence = intent.confidence if intent.confidence is not None else 0.0
        elif isinstance(intent, dict):
            key = str(intent.get("key") or "")
            summary = str(intent.get("summary") or intent.get("raw") or "")
            confidence = float(intent.get("confidence") or 0.0)
        else:
            key = ""
            summary = str(data.get("semantic_label") or "")
            confidence = 0.0

        value = key or summary
        if not value:
            continue
        label = (
            f"{key} - {summary}"
            if key and summary and key != summary
            else (summary or key)
        )
        existing = intents.get(value)
        if not existing or confidence > existing.get("confidence", 0.0):
            intents[value] = {
                "type": "intent",
                "value": value,
                "key": key or None,
                "summary": summary,
                "confidence": confidence,
                "label": label,
                "path_length": 1,
            }
    options.extend(
        sorted(
            intents.values(),
            key=lambda x: (
                x.get("key") is None,
                -(x.get("confidence") or 0.0),
                x["value"],
            ),
        )
    )
    templates_sorted = [item for item in options if item.get("type") == "template"]
    intents_sorted = [item for item in options if item.get("type") == "intent"]
    templates_sorted.sort(
        key=lambda x: (
            -(x.get("confidence") or 0.0),
            -(x.get("path_length") or 0),
            x["value"],
        )
    )
    if templates_sorted:
        return templates_sorted
    return intents_sorted


def _log_entry_to_sse(entry: dict) -> dict:
    """Convert playback engine log entry to SSE event payload (step_index, message, level, ...)."""
    step_index = entry.get("step_index", 0)
    selector = entry.get("selector", "")
    action = entry.get("action", "")
    success = entry.get("success", False)
    error = entry.get("error")
    intent_summary = entry.get("intent", "")

    message = f"Step {step_index}: {action} {selector}"
    if intent_summary:
        message += f" ({intent_summary})"

    if error:
        message += f" — {error}"
    level = "error" if not success else "info"
    out = {
        "step_index": step_index,
        "message": message,
        "level": level,
        "selector": selector,
        "action": action,
        "success": success,
        "intent": intent_summary,
    }
    if error is not None:
        out["error"] = error
    return out


def _resolve_start_url_for_path(edge_list: list[GraphEdge], graph: Any) -> str:
    """Resolve playback start URL from first edge source, fallback to default."""
    graph_start_url = graph.graph.get("start_url") if hasattr(graph, "graph") else None
    if edge_list:
        first = edge_list[0]
        source_id = first.source
        if source_id in graph:
            node_data = graph.nodes[source_id]
            url = node_data.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
    if isinstance(graph_start_url, str) and graph_start_url.startswith("http"):
        return graph_start_url
    return DEFAULT_START_URL


async def _sse_generator(
    edge_list: list[GraphEdge],
    test_data: dict,
    start_url: str,
    expected_end_url: str | None,
    wait_for_network: bool,
):
    """Yield SSE lines: data: {json}\n\n. Runs run_playback as a background task and streams logs."""
    q: asyncio.Queue = asyncio.Queue()

    async def log_callback(entry: dict) -> None:
        await q.put(_log_entry_to_sse(entry))

    async def worker():
        try:
            result = await run_playback(
                edge_list,
                test_data,
                start_url,
                expected_end_url,
                log_callback=log_callback,
                wait_for_network=wait_for_network,
            )
            if result["success"]:
                await q.put({"level": "success", "actual_url": result["actual_url"]})
            else:
                await q.put(
                    {
                        "level": "error",
                        "error": result.get("error") or "Playback failed",
                        "actual_url": result.get("actual_url", ""),
                    }
                )
        except Exception as e:
            await q.put({"level": "error", "error": str(e), "actual_url": ""})
        finally:
            # Signal end of stream
            await q.put(None)

    # Start worker task
    task = asyncio.create_task(worker())

    while True:
        item = await q.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"
        if item.get("level") in ("success", "error"):
            # Wait for worker to finish (it should be done or close to done)
            break

    await task


@app.get("/api/dashboard")
def get_dashboard():
    """
    Return dashboard statistics: edge_count, intent_missing_count, intent_success_rate,
    filtered_non_ui_edges, node_count, mapping_stopped, stop_reason, quality metrics.
    """
    if not GRAPH_PATH.exists():
        return {
            "node_count": 0,
            "edge_count": 0,
            "intent_missing_count": 0,
            "intent_success_rate": 1.0,
            "filtered_non_ui_edges": 0,
            "mapping_stopped": None,
            "stop_reason": None,
            "semantic_consistency_rate": 1.0,
            "inventory_non_empty_rate": 0.0,
            "re_infer_success_rate": None,
            "business_template_count": 0,
            "business_template_generation_failures": 0,
            "runtime_non_ui_action_count": 0,
            "state_like_node_ratio": 0.0,
            "business_intent_edge_ratio": 0.0,
            "multi_edge_preserved_count": 0,
        }
    G = load_graph(GRAPH_PATH)
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    missing_count = sum(
        1 for _u, _v, data in G.edges(data=True) if data.get("intent") is None
    )
    intent_success_rate = 1.0 - (missing_count / edge_count) if edge_count > 0 else 1.0
    metadata = dict(G.graph)
    filtered_non_ui_edges = metadata.get("filtered_non_ui_edges", 0)
    mapping_stopped = metadata.get("mapping_stopped")
    stop_reason = metadata.get("stop_reason")
    semantic_consistency_rate = metadata.get("semantic_consistency_rate")
    inventory_non_empty_rate = metadata.get("inventory_non_empty_rate")
    re_infer_success_rate = metadata.get("re_infer_success_rate")
    business_template_count = metadata.get("business_template_count", 0)
    business_template_generation_failures = metadata.get(
        "business_template_generation_failures", 0
    )
    runtime_non_ui_action_count = metadata.get("runtime_non_ui_action_count")
    state_like_node_ratio = metadata.get("state_like_node_ratio")
    business_intent_edge_ratio = metadata.get("business_intent_edge_ratio")
    multi_edge_preserved_count = metadata.get("multi_edge_preserved_count")
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "intent_missing_count": missing_count,
        "intent_success_rate": round(intent_success_rate, 4),
        "filtered_non_ui_edges": filtered_non_ui_edges,
        "mapping_stopped": mapping_stopped,
        "stop_reason": stop_reason,
        "semantic_consistency_rate": semantic_consistency_rate,
        "inventory_non_empty_rate": inventory_non_empty_rate,
        "re_infer_success_rate": re_infer_success_rate,
        "business_template_count": business_template_count,
        "business_template_generation_failures": business_template_generation_failures,
        "runtime_non_ui_action_count": runtime_non_ui_action_count,
        "state_like_node_ratio": state_like_node_ratio,
        "business_intent_edge_ratio": business_intent_edge_ratio,
        "multi_edge_preserved_count": multi_edge_preserved_count,
    }


@app.post("/api/playback")
def post_playback(body: PlaybackRequest):
    """
    Run playback for the given intent and test_data.
    Loads graph, resolves path from intent, runs playback in a background thread, streams SSE.
    Body: { "intent": str, "test_data": dict } (e.g. {"username": "tomsmith", "password": "SuperSecretPassword!"}).
    Multiple concurrent playbacks are allowed (no global lock).
    """
    if not GRAPH_PATH.exists():
        return JSONResponse(
            status_code=400,
            content={"error": "Graph file not found"},
        )
    G = load_graph(GRAPH_PATH)
    edge_list = get_path_from_query(body.intent, G)
    if not edge_list:
        # Stream one SSE event then close
        def no_path_stream():
            yield f"data: {json.dumps({'level': 'error', 'error': 'no matching path'})}\n\n"

        return StreamingResponse(
            no_path_stream(),
            media_type="text/event-stream",
        )
    # Determine expected end URL dynamically from the last edge
    expected_end_url = None
    if edge_list:
        last_edge = edge_list[-1]
        target_id = last_edge.target
        # Try to find URL from node in graph
        if target_id in G:
            node_data = G.nodes[target_id]
            url = node_data.get("url")
            if url and url.startswith("http"):
                expected_end_url = url

    start_url = _resolve_start_url_for_path(edge_list, G)
    return StreamingResponse(
        _sse_generator(
            edge_list,
            body.test_data,
            start_url=start_url,
            expected_end_url=expected_end_url,
            wait_for_network=body.wait_for_network,
        ),
        media_type="text/event-stream",
    )


_graph_stream_subscribers: list[asyncio.Queue] = []


def notify_intent_update(edge_id: str, intent_dict: dict | None, status: str) -> None:
    """Push an intent update event to all SSE subscribers (called from IntentWorker)."""
    event = {"edge_id": edge_id, "intent": intent_dict, "status": status}
    for q in list(_graph_stream_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


@app.get("/api/graph/stream")
async def graph_stream():
    """SSE endpoint that pushes edge intent updates as they resolve."""
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _graph_stream_subscribers.append(q)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in _graph_stream_subscribers:
                _graph_stream_subscribers.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve single-page UI: "/" -> index.html, other static files from graph_agent/web/static
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

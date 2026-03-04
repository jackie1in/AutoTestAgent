"""FastAPI app: GET /api/graph, GET /api/intents, POST /api/playback (SSE)."""

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

load_dotenv()

from graph_agent.graph.io import load_graph
from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.playback.engine import run_playback
from graph_agent.models import GraphEdge, Intent, ActionType, ElementConstraints

# graph_agent/web/app.py -> graph_agent -> data/graph.json
GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# MVP: fixed start/end URL for the-internet login flow
DEFAULT_START_URL = os.getenv("MAPPING_URL", "https://the-internet.herokuapp.com/login")
DEFAULT_EXPECTED_END_URL = "https://the-internet.herokuapp.com/secure"  # or None to skip assertion

app = FastAPI(title="Graph Agent API")


class PlaybackRequest(BaseModel):
    """Request body for POST /api/playback."""

    intent: str
    test_data: dict = {}


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
    for u, v, data in G.edges(data=True):
        intent = data.get("intent")
        intent_dict = intent.model_dump() if isinstance(intent, Intent) else None
        if intent is None:
            missing_count += 1
            reason = data.get("intent_failure_reason")
            if reason:
                failure_reasons.add(reason)
        
        constraints = data.get("constraints")
        constraints_dict = constraints.model_dump() if isinstance(constraints, ElementConstraints) else None

        edge = {
            "source": str(u),
            "target": str(v),
            "selector": data.get("selector", ""),
            "action": data.get("action", ""),
            "intent": intent_dict,
            "intent_failure_reason": data.get("intent_failure_reason"),
            "data_key": data.get("data_key"),
            "constraints": constraints_dict,
        }
        edges.append(edge)
    return {
        "nodes": nodes,
        "edges": edges,
        "missing_count": missing_count,
        "failure_reasons": sorted(failure_reasons),
    }


@app.get("/api/graph")
def get_graph():
    """Load graph from graph_agent/data/graph.json. Return 200 with nodes/edges; if file missing return empty graph."""
    if not GRAPH_PATH.exists():
        return {"nodes": [], "edges": [], "missing_count": 0, "failure_reasons": []}
    G = load_graph(GRAPH_PATH)
    return _graph_to_json_dict(G)


@app.get("/api/intents")
def get_intents():
    """Return unique intents with key/summary/confidence for UI selection."""
    if not GRAPH_PATH.exists():
        return []
    G = load_graph(GRAPH_PATH)
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
        label = f"{key} - {summary}" if key and summary and key != summary else (summary or key)
        existing = intents.get(value)
        if not existing or confidence > existing.get("confidence", 0.0):
            intents[value] = {
                "value": value,
                "key": key or None,
                "summary": summary,
                "confidence": confidence,
                "label": label,
            }

    return sorted(intents.values(), key=lambda x: (x.get("key") is None, -(x.get("confidence") or 0.0), x["value"]))


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


async def _sse_generator(edge_list: list[GraphEdge], test_data: dict, start_url: str, expected_end_url: str | None):
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
            )
            if result["success"]:
                await q.put({"level": "success", "actual_url": result["actual_url"]})
            else:
                await q.put({
                    "level": "error",
                    "error": result.get("error") or "Playback failed",
                    "actual_url": result.get("actual_url", ""),
                })
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
    edge_list = get_path_from_intent(body.intent, G)
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
        # Fallback: if target_id itself looks like a URL
        elif target_id.startswith("http"):
            expected_end_url = target_id

    return StreamingResponse(
        _sse_generator(
            edge_list,
            body.test_data,
            start_url=DEFAULT_START_URL,
            expected_end_url=expected_end_url,
        ),
        media_type="text/event-stream",
    )


# Serve single-page UI: "/" -> index.html, other static files from graph_agent/web/static
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

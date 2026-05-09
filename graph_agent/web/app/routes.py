import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from graph_agent.cartography.config import resolve_knowledge_trigger_profile
from graph_agent.graph.pathfinding import get_path_from_nl_query, get_path_from_query
from graph_agent.models import GraphEdge
from graph_agent.web.app.config import (
    DEFAULT_START_URL,
    _normalize_knowledge_profile,
    _resolve_app_name,
)
from graph_agent.web.app.data_queries import _get_edges_from_neo4j, _get_graph_from_neo4j
from graph_agent.web.app.graphrag_setup import (
    _app_lifespan,
    _ensure_graphrag,
    _get_driver,
)
from graph_agent.web.app.models import (
    KnowledgeSettingsRequest,
    NLPlaybackRequest,
    NLResolveRequest,
    PlaybackRequest,
)
from graph_agent.web.app.playback_utils import (
    _build_path_preview,
    _resolve_end_url,
    _resolve_start_url,
)
from graph_agent.web.app.sse_utils import (
    _graph_stream_subscribers,
    _sse_generator,
    notify_intent_update,
)

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Graph Agent API", lifespan=_app_lifespan)

logger = logging.getLogger(__name__)


@app.get("/api/graph")
async def get_graph():
    driver = await _get_driver()
    neo4j_data = await _get_graph_from_neo4j(driver)
    if neo4j_data is not None:
        return neo4j_data
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


@app.get("/api/intents")
async def get_intents():
    app_name = _resolve_app_name()
    try:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (a)-[:HAS_STATE]->(:State)<-[:FROM]-(t:Transition)-[:REALIZES]->(i:Intent)
                RETURN DISTINCT i.key AS key,
                       i.summary AS summary,
                       i.confidence AS confidence
                ORDER BY confidence DESC
                """,
                app_name=app_name,
            )
            records = [dict(r) async for r in result]
    except Exception as e:
        logger.warning("Neo4j intents query failed: %s", e)
        return []

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = rec.get("key") or ""
        summary = rec.get("summary") or ""
        confidence = rec.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence = float(confidence)
        else:
            confidence = 0.5
        value = key or summary
        if not value:
            continue
        label = (
            f"{key} - {summary}"
            if key and summary and key != summary
            else (summary or key)
        )
        if value not in seen or confidence > seen[value].get("confidence", 0.0):
            seen[value] = {
                "type": "intent",
                "value": value,
                "key": key or None,
                "summary": summary,
                "confidence": confidence,
                "label": label,
                "path_length": 1,
            }

    return sorted(
        seen.values(),
        key=lambda x: (x.get("key") is None, -(x.get("confidence") or 0.0), x["value"]),
    )


@app.get("/api/dashboard")
async def get_dashboard():
    app_name = _resolve_app_name()
    try:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
                OPTIONAL MATCH (a)-[:HAS_STATE]->(:State)<-[:FROM]-(t:Transition)
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN count(DISTINCT s) AS node_count,
                       count(DISTINCT t) AS edge_count,
                       count(DISTINCT CASE WHEN i IS NULL THEN t END) AS missing_count
                """,
                app_name=app_name,
            )
            record = await result.single()
            if record:
                node_count = record.get("node_count", 0)
                edge_count = record.get("edge_count", 0)
                missing_count = record.get("missing_count", 0)
                intent_success_rate = (
                    1.0 - (missing_count / edge_count) if edge_count > 0 else 1.0
                )
                session_result = await session.run(
                    """
                    MATCH (a:App)
                    WHERE $app_name = '' OR a.name = $app_name
                    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                    OPTIONAL MATCH (a)-[:HAS_SESSION]->(sess:Session)
                    RETURN sess
                    ORDER BY coalesce(sess.timestamp, sess.created_at) DESC
                    LIMIT 1
                    """,
                    app_name=app_name,
                )
                session_record = await session_result.single()
                sess = session_record.get("sess") if session_record else None
                knowledge_query_count = 0
                knowledge_hit_count = 0
                knowledge_cache_hit_count = 0
                knowledge_timeout_count = 0
                knowledge_circuit_open_count = 0
                knowledge_error_count = 0
                knowledge_avg_latency_ms = 0.0
                session_trigger_profile = "balanced"
                if sess:
                    knowledge_query_count = int(
                        sess.get("knowledge_query_count", 0) or 0
                    )
                    knowledge_hit_count = int(sess.get("knowledge_hit_count", 0) or 0)
                    knowledge_cache_hit_count = int(
                        sess.get("knowledge_cache_hit_count", 0) or 0
                    )
                    knowledge_timeout_count = int(
                        sess.get("knowledge_timeout_count", 0) or 0
                    )
                    knowledge_circuit_open_count = int(
                        sess.get("knowledge_circuit_open_count", 0) or 0
                    )
                    knowledge_error_count = int(
                        sess.get("knowledge_error_count", 0) or 0
                    )
                    knowledge_avg_latency_ms = float(
                        sess.get("knowledge_avg_latency_ms", 0.0) or 0.0
                    )
                    session_trigger_profile = str(
                        sess.get("knowledge_trigger_profile", "balanced") or "balanced"
                    )
                knowledge_trigger_profile = _normalize_knowledge_profile(
                    resolve_knowledge_trigger_profile()
                )
                return {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "intent_missing_count": missing_count,
                    "intent_success_rate": round(intent_success_rate, 4),
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
                    "knowledge_query_count": knowledge_query_count,
                    "knowledge_hit_count": knowledge_hit_count,
                    "knowledge_cache_hit_count": knowledge_cache_hit_count,
                    "knowledge_timeout_count": knowledge_timeout_count,
                    "knowledge_circuit_open_count": knowledge_circuit_open_count,
                    "knowledge_error_count": knowledge_error_count,
                    "knowledge_avg_latency_ms": round(knowledge_avg_latency_ms, 3),
                    "knowledge_trigger_profile": knowledge_trigger_profile,
                    "knowledge_last_session_trigger_profile": _normalize_knowledge_profile(
                        session_trigger_profile
                    ),
                }
    except Exception as e:
        logger.warning("Neo4j dashboard query failed: %s", e)

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
        "knowledge_query_count": 0,
        "knowledge_hit_count": 0,
        "knowledge_cache_hit_count": 0,
        "knowledge_timeout_count": 0,
        "knowledge_circuit_open_count": 0,
        "knowledge_error_count": 0,
        "knowledge_avg_latency_ms": 0.0,
        "knowledge_trigger_profile": _normalize_knowledge_profile(
            resolve_knowledge_trigger_profile()
        ),
        "knowledge_last_session_trigger_profile": "balanced",
    }


@app.get("/api/settings/knowledge")
async def get_knowledge_settings():
    from graph_agent.web.app.config import _ALLOWED_KNOWLEDGE_PROFILES
    return {
        "trigger_profile": _normalize_knowledge_profile(
            resolve_knowledge_trigger_profile()
        ),
        "allowed_profiles": sorted(_ALLOWED_KNOWLEDGE_PROFILES),
    }


@app.post("/api/settings/knowledge")
async def post_knowledge_settings(body: KnowledgeSettingsRequest):
    from graph_agent.web.app.config import _ALLOWED_KNOWLEDGE_PROFILES
    profile = _normalize_knowledge_profile(body.trigger_profile)
    if profile != (body.trigger_profile or "").strip().lower():
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid trigger_profile",
                "allowed_profiles": sorted(_ALLOWED_KNOWLEDGE_PROFILES),
            },
        )
    os.environ["CARTOGRAPHY_KNOWLEDGE_TRIGGER_PROFILE"] = profile
    return {
        "ok": True,
        "trigger_profile": profile,
    }


@app.post("/api/playback")
async def post_playback(body: PlaybackRequest):
    try:
        driver = await _get_driver()
        edges = await _get_edges_from_neo4j(driver)
    except Exception as e:
        logger.warning("Failed to load edges from Neo4j: %s", e)

        def error_stream():
            yield f"data: {json.dumps({'level': 'error', 'error': 'Failed to load graph data'})}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    edge_list = get_path_from_query(body.intent, edges)
    if not edge_list:

        def no_path_stream():
            yield f"data: {json.dumps({'level': 'error', 'error': 'no matching path'})}\n\n"

        return StreamingResponse(no_path_stream(), media_type="text/event-stream")

    start_url = _resolve_start_url(edge_list)
    expected_end_url = _resolve_end_url(edge_list)
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


@app.post("/api/nl-resolve")
async def post_nl_resolve(body: NLResolveRequest):
    try:
        driver = await _get_driver()
        edges = await _get_edges_from_neo4j(driver)
    except Exception as e:
        logger.warning("Failed to load edges from Neo4j: %s", e)
        return JSONResponse(
            status_code=400, content={"error": "Failed to load graph data"}
        )

    graphrag = await _ensure_graphrag()
    resolved_intents: list[Any] = []
    source = "keyword_fallback"

    if graphrag is not None:
        driver_wrapper, embedder = graphrag
        try:
            from graph_agent.cartography.nl_resolver import NLResolver

            resolver = NLResolver(driver_wrapper.driver, embedder)
            resolved_intents = await resolver.resolve(body.query, top_k=body.top_k)
            if resolved_intents:
                source = "graphrag"
        except Exception as e:
            logger.warning("NL resolve failed: %s", e)

    intent_key = ""
    if resolved_intents:
        intent_key = resolved_intents[0].get("key", "")
    if not intent_key:
        intent_key = body.query

    path = get_path_from_query(intent_key, edges)

    if not path and not resolved_intents:
        return {
            "resolved": False,
            "query": body.query,
            "error": "No matching intent found",
            "intents": [],
            "template": None,
            "path_preview": None,
            "source": None,
        }

    return {
        "resolved": True,
        "query": body.query,
        "intents": resolved_intents,
        "template": None,
        "path_preview": _build_path_preview(path) if path else None,
        "source": source,
    }


@app.post("/api/nl-playback")
async def post_nl_playback(body: NLPlaybackRequest):
    try:
        driver = await _get_driver()
        edges = await _get_edges_from_neo4j(driver)
    except Exception as e:
        logger.warning("Failed to load edges from Neo4j: %s", e)

        def error_stream():
            yield f"data: {json.dumps({'level': 'error', 'error': 'Failed to load graph data'})}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    graphrag = await _ensure_graphrag()
    driver_instance = None
    embedder_instance = None
    if graphrag is not None:
        driver_instance = graphrag[0].driver
        embedder_instance = graphrag[1]

    edge_list = await get_path_from_nl_query(
        body.query,
        edges,
        driver=driver_instance,
        embedder=embedder_instance,
    )

    if not edge_list:

        def no_path_stream():
            yield f"data: {json.dumps({'level': 'error', 'error': 'no matching path'})}\n\n"

        return StreamingResponse(no_path_stream(), media_type="text/event-stream")

    start_url = _resolve_start_url(edge_list)
    expected_end_url = _resolve_end_url(edge_list)
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


@app.get("/api/graph/stream")
async def graph_stream():
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


# Serve single-page UI
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

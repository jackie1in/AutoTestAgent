"""FastAPI app: GET /api/graph, GET /api/intents, POST /api/playback (SSE), GET /api/graph/stream (SSE)."""

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
from pydantic import BaseModel

from graph_agent.cartography.config import resolve_knowledge_trigger_profile
from graph_agent.graph.pathfinding import (
    _edge_to_model,
    get_path_from_nl_query,
    get_path_from_query,
    neo4j_transition_to_edge_data,
)
from graph_agent.models import GraphEdge
from graph_agent.playback.engine import run_playback

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"

# MVP: fixed start/end URL for the-internet login flow
DEFAULT_START_URL = os.getenv("MAPPING_URL", "https://the-internet.herokuapp.com/login")
DEFAULT_EXPECTED_END_URL = (
    "https://the-internet.herokuapp.com/secure"  # or None to skip assertion
)

app = FastAPI(title="Graph Agent API")

logger = logging.getLogger(__name__)

# -- GraphRAG lazy-init state (for NL playback) --
_neo4j_driver: Any | None = None
_embedder: Any | None = None
_graphrag_available: bool | None = None
_ALLOWED_KNOWLEDGE_PROFILES = {"conservative", "balanced", "aggressive"}


def _normalize_knowledge_profile(value: str) -> str:
    profile = (value or "").strip().lower()
    if profile in _ALLOWED_KNOWLEDGE_PROFILES:
        return profile
    return "balanced"


async def _ensure_graphrag() -> tuple[Any, Any] | None:
    """Lazy-init Neo4j driver + embedder. Returns (driver, embedder) or None."""
    global _neo4j_driver, _embedder, _graphrag_available

    if _graphrag_available is not None:
        return (_neo4j_driver, _embedder) if _graphrag_available else None

    try:
        from graph_agent.cartography.nl_resolver import _get_embedder
        from graph_agent.neo4j_client.driver import Neo4jDriver

        _neo4j_driver = Neo4jDriver()
        await _neo4j_driver.connect()
        _embedder = _get_embedder()
        _graphrag_available = True
        logger.info("GraphRAG initialized for NL playback")
        return (_neo4j_driver, _embedder)
    except Exception as e:
        logger.warning("GraphRAG initialization failed: %s", e)
        _graphrag_available = False
        return None


def _resolve_app_name() -> str:
    """Read MAPPING_APP_NAME from env; empty string means fallback to latest app."""
    return (os.getenv("MAPPING_APP_NAME") or "").strip()


def _resolve_release_id() -> str:
    """Optional release selector for playback reads."""
    return (os.getenv("MAPPING_RELEASE_ID") or "").strip()


async def _get_driver() -> Any:
    """Return the raw Neo4j async driver, initializing if needed."""
    graphrag = await _ensure_graphrag()
    if graphrag is not None:
        return graphrag[0].driver
    raise RuntimeError("Neo4j driver not available")


async def _get_edges_from_neo4j(driver) -> list[GraphEdge]:
    """Query transitions from Neo4j and build GraphEdge list."""
    app_name = _resolve_app_name()
    release_id = _resolve_release_id()
    async with driver.session() as session:
        if release_id:
            result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (r:GraphRelease {id: $release_id, app_id: a.id, status: 'active'})
                      <-[:IN_RELEASE]-(rev:TransitionRevision {is_active: true})
                      <-[:HAS_REVISION]-(:TransitionEntity)
                MATCH (t:Transition {id: rev.transition_id})
                OPTIONAL MATCH (s:State {id: rev.from_state_id})
                OPTIONAL MATCH (target:State {id: rev.to_state_id})
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN t.id AS id, t.step_index AS step_index,
                       s.id AS from_state_id, target.id AS to_state_id,
                       t.source_url AS source_url, t.target_url AS target_url,
                       t.selector AS selector, t.action AS action,
                       t.tab_id AS tab_id, t.target_tab_id AS target_tab_id,
                       t.tab_action AS tab_action,
                       t.intent_failure_reason AS intent_failure_reason,
                       t.param_name AS param_name, t.action_value AS action_value,
                       t.thought AS thought, t.element_snapshot AS element_snapshot,
                       t.frame_path AS frame_path,
                       i{.*} AS intent
                ORDER BY t.step_index
                """,
                app_name=app_name,
                release_id=release_id,
            )
            first_record = await result.peek()
            if first_record is None:
                result = await session.run(
                    """
                    MATCH (a:App)
                    WHERE $app_name = '' OR a.name = $app_name
                    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                    RETURN t.id AS id, t.step_index AS step_index,
                           s.id AS from_state_id, target.id AS to_state_id,
                           t.source_url AS source_url, t.target_url AS target_url,
                           t.selector AS selector, t.action AS action,
                           t.tab_id AS tab_id, t.target_tab_id AS target_tab_id,
                           t.tab_action AS tab_action,
                           t.intent_failure_reason AS intent_failure_reason,
                           t.param_name AS param_name, t.action_value AS action_value,
                           t.thought AS thought, t.element_snapshot AS element_snapshot,
                           t.frame_path AS frame_path,
                           i{.*} AS intent
                    ORDER BY t.step_index
                    """,
                    app_name=app_name,
                )
        else:
            result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN t.id AS id, t.step_index AS step_index,
                       s.id AS from_state_id, target.id AS to_state_id,
                       t.source_url AS source_url, t.target_url AS target_url,
                       t.selector AS selector, t.action AS action,
                       t.tab_id AS tab_id, t.target_tab_id AS target_tab_id,
                       t.tab_action AS tab_action,
                       t.intent_failure_reason AS intent_failure_reason,
                       t.param_name AS param_name, t.action_value AS action_value,
                       t.thought AS thought, t.element_snapshot AS element_snapshot,
                       t.frame_path AS frame_path,
                       i{.*} AS intent
                ORDER BY t.step_index
                """,
                app_name=app_name,
            )
        edges: list[GraphEdge] = []
        async for record in result:
            t = dict(record)
            data = neo4j_transition_to_edge_data(t)
            u = str(t.get("from_state_id", ""))
            v = str(t.get("to_state_id", ""))
            if u and v:
                edges.append(_edge_to_model(u, v, data))
        return edges


class PlaybackRequest(BaseModel):
    """Request body for POST /api/playback."""

    intent: str
    test_data: dict = {}
    wait_for_network: bool = True


class NLResolveRequest(BaseModel):
    """Request body for POST /api/nl-resolve."""

    query: str
    top_k: int = 3


class NLPlaybackRequest(BaseModel):
    """Request body for POST /api/nl-playback."""

    query: str
    test_data: dict = {}
    wait_for_network: bool = True


class KnowledgeSettingsRequest(BaseModel):
    """Request body for POST /api/settings/knowledge."""

    trigger_profile: str


async def _get_graph_from_neo4j(driver) -> dict[str, Any] | None:
    """Query nodes and edges from Neo4j, returning API-compatible graph dict."""
    app_name = _resolve_app_name()
    try:
        async with driver.session() as session:
            # Query nodes from the target app (or most recently active)
            node_result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
                WITH a, collect(DISTINCT {
                    id: s.id, type: 'State',
                    label: coalesce(s.name, s.title, s.url, s.id),
                    url: s.url, title: s.title
                }) AS states
                OPTIONAL MATCH (a)-[:HAS_ZONE]->(z:Zone)
                WITH a, states, collect(DISTINCT {
                    id: z.id, type: 'Zone',
                    label: coalesce(z.name, z.summary, z.zone_type, z.id),
                    summary: z.summary, zone_type: z.zone_type
                }) AS zones
                OPTIONAL MATCH (a)-[:HAS_MENU]->(m:Menu)
                WITH a, states, zones, collect(DISTINCT {
                    id: m.id, type: 'Menu',
                    label: coalesce(m.name, m.label, m.menu_key, m.id),
                    label_text: m.label, menu_key: m.menu_key
                }) AS menus
                OPTIONAL MATCH (a)-[:HAS_ENTITY]->(e:Entity)
                RETURN states AS state_nodes, zones AS zone_nodes,
                       menus AS menu_nodes,
                       collect(DISTINCT {
                           id: e.id, type: 'Entity',
                           label: coalesce(e.name, e.description, e.id),
                           name: e.name
                       }) AS entity_nodes
                """,
                app_name=app_name,
            )
            node_record = await node_result.single()
            nodes: list[dict[str, Any]] = []
            if node_record:
                for key in ("state_nodes", "zone_nodes", "menu_nodes", "entity_nodes"):
                    nodes.extend(node_record.get(key, []))
            nodes = [n for n in nodes if n and n.get("id")]

            # Query transitions (edges)
            edge_result = await session.run(
                """
                MATCH (a:App)
                WHERE $app_name = '' OR a.name = $app_name
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN t.id AS edge_id, t.step_index AS step_index,
                       s.id AS source, target.id AS target,
                       t.source_url AS source_url, t.target_url AS target_url,
                       t.selector AS selector, t.action AS action,
                       i{.*} AS intent,
                       t.intent_failure_reason AS intent_failure_reason,
                       t.param_name AS param_name, t.action_value AS action_value
                ORDER BY t.step_index
                """,
                app_name=app_name,
            )
            edges = [dict(record) async for record in edge_result]

            missing_count = sum(1 for e in edges if e.get("intent") is None)
            failure_reasons = sorted(
                set(
                    e["intent_failure_reason"]
                    for e in edges
                    if e.get("intent_failure_reason")
                )
            )

            return {
                "nodes": nodes,
                "edges": edges,
                "missing_count": missing_count,
                "failure_reasons": failure_reasons,
                "business_templates": [],
                "metadata": {
                    "filtered_non_ui_edges": 0,
                    "intent_missing_count": missing_count,
                    "intent_success_rate": 1.0 - (missing_count / len(edges))
                    if edges
                    else 1.0,
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
    except Exception as e:
        logger.warning("Neo4j graph query failed: %s", e)
        return None


@app.get("/api/graph")
async def get_graph():
    """Return graph data from Neo4j."""
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
    """Return intent query options from Neo4j."""
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


def _log_entry_to_sse(entry: dict) -> dict:
    """Convert playback engine log entry to SSE event payload."""
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


def _resolve_start_url(edge_list: list[GraphEdge]) -> str:
    """Resolve playback start URL from first edge source_url, fallback to default."""
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
    """Resolve expected end URL from last edge target_url."""
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
    """Build a preview of the playback path for frontend display."""
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


async def _apply_playback_feedback(
    edge_list: list[GraphEdge],
    playback_result: dict[str, Any],
) -> None:
    """Feed playback outcome back into transition confidence and coverage stats."""
    try:
        driver = await _get_driver()
    except Exception:
        return
    success = bool(playback_result.get("success"))
    failed_step_index = playback_result.get("failed_step_index")
    failed_edge_id = playback_result.get("failed_edge_id")

    transition_ids = [edge.edge_id for edge in edge_list if edge.edge_id]
    if not transition_ids:
        return
    async with driver.session() as session:
        for idx, edge in enumerate(edge_list):
            if not edge.edge_id:
                continue
            passed = success or (
                failed_step_index is not None and idx < int(failed_step_index)
            )
            if failed_edge_id and edge.edge_id == failed_edge_id:
                passed = False
            delta = 0.1 if passed else -0.2
            await session.run(
                """
                MATCH (t:Transition {id: $transition_id})
                SET t.confidence = CASE
                    WHEN coalesce(t.confidence, 0.5) + $delta > 1.0 THEN 1.0
                    WHEN coalesce(t.confidence, 0.5) + $delta < 0.0 THEN 0.0
                    ELSE coalesce(t.confidence, 0.5) + $delta
                END,
                t.last_validated = datetime(),
                t.validation_count = coalesce(t.validation_count, 0) + 1
                """,
                transition_id=edge.edge_id,
                delta=delta,
            )

        app_name = _resolve_app_name()
        stats_result = await session.run(
            """
            MATCH (a:App)
            WHERE $app_name = '' OR a.name = $app_name
            WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
            OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
            WITH a, count(s) AS total_states
            OPTIONAL MATCH (a)-[:HAS_STATE]->(:State)<-[:FROM]-(t:Transition)
            WITH a, total_states, count(t) AS total_transitions,
                 count(CASE WHEN coalesce(t.validation_count, 0) > 0 THEN 1 END) AS validated_transitions
            SET a.last_playback_success = $success,
                a.last_playback_at = datetime(),
                a.interaction_coverage = CASE
                    WHEN total_transitions = 0 THEN 0.0
                    ELSE toFloat(validated_transitions) / toFloat(total_transitions)
                END
            RETURN a.id AS app_id
            """,
            app_name=app_name,
            success=success,
        )
        await stats_result.single()


async def _sse_generator(
    edge_list: list[GraphEdge],
    test_data: dict,
    start_url: str,
    expected_end_url: str | None,
    wait_for_network: bool,
):
    """Yield SSE lines: data: {json}\n\n."""
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
            await _apply_playback_feedback(edge_list, result)
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
            await q.put(None)

    task = asyncio.create_task(worker())
    while True:
        item = await q.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"
        if item.get("level") in ("success", "error"):
            break
    await task


@app.get("/api/dashboard")
async def get_dashboard():
    """Return dashboard statistics from Neo4j."""
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
    """Return active knowledge trigger settings."""
    return {
        "trigger_profile": _normalize_knowledge_profile(
            resolve_knowledge_trigger_profile()
        ),
        "allowed_profiles": sorted(_ALLOWED_KNOWLEDGE_PROFILES),
    }


@app.post("/api/settings/knowledge")
async def post_knowledge_settings(body: KnowledgeSettingsRequest):
    """Update active knowledge trigger profile for current process."""
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
    """Run playback for the given intent and test_data."""
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
    """Resolve natural language query to intent + path preview via GraphRAG."""
    try:
        driver = await _get_driver()
        edges = await _get_edges_from_neo4j(driver)
    except Exception as e:
        logger.warning("Failed to load edges from Neo4j: %s", e)
        return JSONResponse(
            status_code=400, content={"error": "Failed to load graph data"}
        )

    # Try GraphRAG first
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
    """Natural language playback: resolve intent via GraphRAG, then stream playback via SSE."""
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


_graph_stream_subscribers: list[asyncio.Queue] = []


def notify_intent_update(edge_id: str, intent_dict: dict | None, status: str) -> None:
    """Push an intent update event to all SSE subscribers."""
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


@app.on_event("shutdown")
async def shutdown_event():
    """Close Neo4j driver on app shutdown."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        try:
            await _neo4j_driver.close()
            logger.info("Neo4j driver closed on shutdown")
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", e)


# Serve single-page UI
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

import logging
from typing import Any

from graph_agent.graph.pathfinding import (
    _edge_to_model,
    neo4j_transition_to_edge_data,
)
from graph_agent.models import GraphEdge
from graph_agent.web.app.config import _resolve_app_name, _resolve_release_id

logger = logging.getLogger(__name__)


async def _get_edges_from_neo4j(driver) -> list[GraphEdge]:
    app_name = _resolve_app_name()
    release_id = _resolve_release_id()

    query: str
    params: dict[str, str]

    if release_id:
        query = """
            MATCH (a:App)
            WHERE $app_name = '' OR a.name = $app_name
            WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
            MATCH (r:GraphRelease {id: $release_id, app_id: a.id, status: 'active'})
                  <-[:IN_RELEASE]-(rev:TransitionRevision {is_active: true})
                  <-[:HAS_REVISION]-(:TransitionEntity)
            MATCH (s:State {id: rev.from_state_id})
            MATCH (target:State {id: rev.to_state_id})
            RETURN rev.revision_id AS id, rev.action AS action,
                   rev.selector AS selector, rev.intent_key AS intent_key,
                   s.id AS from_state_id, target.id AS to_state_id,
                   s.url AS source_url, target.url AS target_url,
                   rev.action_value AS action_value, rev.frame_path AS frame_path,
                   rev.element_snapshot AS element_snapshot,
                   rev.tab_id AS tab_id, rev.target_tab_id AS target_tab_id,
                   rev.tab_action AS tab_action, rev.thought AS thought,
                   rev.step_index AS step_index, rev.param_name AS param_name,
                   rev.intent_failure_reason AS intent_failure_reason
            ORDER BY rev.created_at
        """
        params = {"app_name": app_name, "release_id": release_id}
    else:
        query = """
            MATCH (a:App)
            WHERE $app_name = '' OR a.name = $app_name
            WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
            MATCH (a)-[:HAS_STATE]->(s:State)
            MATCH (rev:TransitionRevision {from_state_id: s.id, is_active: true})
            MATCH (target:State {id: rev.to_state_id})
            RETURN rev.revision_id AS id, rev.action AS action,
                   rev.selector AS selector, rev.intent_key AS intent_key,
                   s.id AS from_state_id, target.id AS to_state_id,
                   s.url AS source_url, target.url AS target_url,
                   rev.action_value AS action_value, rev.frame_path AS frame_path,
                   rev.element_snapshot AS element_snapshot,
                   rev.tab_id AS tab_id, rev.target_tab_id AS target_tab_id,
                   rev.tab_action AS tab_action, rev.thought AS thought,
                   rev.step_index AS step_index, rev.param_name AS param_name,
                   rev.intent_failure_reason AS intent_failure_reason
            ORDER BY rev.created_at
        """
        params = {"app_name": app_name}

    async with driver.session() as session:
        result = await session.run(query, **params)
        edges: list[GraphEdge] = []
        async for record in result:
            t = dict(record)
            data = neo4j_transition_to_edge_data(t)
            u = str(t.get("from_state_id", ""))
            v = str(t.get("to_state_id", ""))
            if u and v:
                edges.append(_edge_to_model(u, v, data))
        return edges


async def _get_graph_from_neo4j(driver) -> dict[str, Any] | None:
    app_name = _resolve_app_name()
    try:
        async with driver.session() as session:
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

"""Save/load NetworkX graph as JSON using Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from pydantic import ValidationError

from graph_agent.models import (
    ActionType,
    ElementConstraints,
    ElementSnapshot,
    GraphData,
    GraphEdge,
    GraphNode,
    Intent,
)


def save_graph(G: nx.Graph, path: str | Path) -> None:
    """Serialize graph to JSON file using GraphData model."""
    path = Path(path)
    
    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append(GraphNode(
            id=str(nid),
            url=data.get("url", ""),
            title=data.get("title")
        ))
        
    edges = []
    edge_iter: Iterable[tuple[Any, Any, Any | None, dict[str, Any]]]
    if isinstance(G, nx.MultiDiGraph):
        edge_iter = ((u, v, key, data) for u, v, key, data in G.edges(keys=True, data=True))
    else:
        edge_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    for u, v, key, data in edge_iter:
        # Handle intent: might be an Intent object or a dict
        intent_data = data.get("intent")
        intent: Intent | None
        if isinstance(intent_data, dict):
            summary = str(intent_data.get("summary") or intent_data.get("raw") or data.get("semantic_label", "Unknown action"))
            intent = Intent(
                raw=str(intent_data.get("raw") or summary),
                verb=str(intent_data.get("verb") or "Unknown"),
                object=str(intent_data.get("object") or "Unknown"),
                summary=summary,
                key=intent_data.get("key"),
                confidence=intent_data.get("confidence"),
            )
        elif isinstance(intent_data, Intent):
            intent = intent_data
        else:
            intent = None

        # Handle constraints
        constraints_data = data.get("constraints")
        constraints = None
        if isinstance(constraints_data, dict):
            constraints = ElementConstraints(**constraints_data)
        elif isinstance(constraints_data, ElementConstraints):
            constraints = constraints_data

        element_data = data.get("element")
        element = None
        if isinstance(element_data, dict):
            element = ElementSnapshot(**element_data)
        elif isinstance(element_data, ElementSnapshot):
            element = element_data

        edge = GraphEdge(
            edge_id=str(data.get("edge_id") or key) if (data.get("edge_id") or key) is not None else None,
            step_index=data.get("step_index"),
            source=str(u),
            target=str(v),
            selector=data.get("selector", ""),
            action=data.get("action", ActionType.UNKNOWN),
            tab_id=data.get("tab_id", "tab-0"),
            target_tab_id=data.get("target_tab_id"),
            tab_action=data.get("tab_action"),
            tab=data.get("tab"),
            frame_path=data.get("frame_path", []),
            intent=intent,
            context_level_used=data.get("context_level_used"),
            intent_failure_reason=data.get("intent_failure_reason"),
            param_name=data.get("param_name"),
            action_value=data.get("action_value"),
            element=element,
            constraints=constraints
        )
        edges.append(edge)
        
    metadata = dict(G.graph)
    
    graph_data = GraphData(nodes=nodes, edges=edges, metadata=metadata)
    
    path.write_text(graph_data.model_dump_json(indent=2), encoding="utf-8")


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load graph from JSON file using GraphData model."""
    path = Path(path)
    if not path.exists():
        return nx.MultiDiGraph()
        
    text = path.read_text(encoding="utf-8")
    try:
        graph_data = GraphData.model_validate_json(text)
    except ValidationError:
        # Fallback for old format? Or just fail as per instructions "backward compatibility is NOT required"
        print(f"Warning: Failed to validate graph data from {path}. Returning empty graph.")
        return nx.MultiDiGraph()

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.graph.update(graph_data.metadata)
    
    for node in graph_data.nodes:
        G.add_node(node.id, url=node.url, title=node.title)
        
    for edge in graph_data.edges:
        # Store complex objects directly in the graph
        G.add_edge(
            edge.source,
            edge.target,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            step_index=edge.step_index,
            selector=edge.selector,
            action=edge.action,
            tab_id=edge.tab_id,
            target_tab_id=edge.target_tab_id,
            tab_action=edge.tab_action,
            tab=edge.tab,
            frame_path=edge.frame_path,
            intent=edge.intent,
            context_level_used=edge.context_level_used,
            intent_failure_reason=edge.intent_failure_reason,
            param_name=edge.param_name,
            action_value=edge.action_value,
            element=edge.element,
            constraints=edge.constraints
        )
        
    return G

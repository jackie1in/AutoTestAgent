"""Save/load NetworkX DiGraph as JSON using Pydantic models."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from pydantic import ValidationError

from graph_agent.models import GraphData, GraphEdge, GraphNode, Intent, ActionType, ElementConstraints


def save_graph(G: nx.DiGraph, path: str | Path) -> None:
    """Serialize DiGraph to JSON file using GraphData model."""
    path = Path(path)
    
    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append(GraphNode(
            id=str(nid),
            url=data.get("url", ""),
            title=data.get("title")
        ))
        
    edges = []
    for u, v, data in G.edges(data=True):
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

        edge = GraphEdge(
            source=str(u),
            target=str(v),
            selector=data.get("selector", ""),
            action=data.get("action", ActionType.UNKNOWN),
            intent=intent,
            intent_failure_reason=data.get("intent_failure_reason"),
            data_key=data.get("data_key"),
            constraints=constraints
        )
        edges.append(edge)
        
    metadata = dict(G.graph)
    
    graph_data = GraphData(nodes=nodes, edges=edges, metadata=metadata)
    
    path.write_text(graph_data.model_dump_json(indent=2), encoding="utf-8")


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load DiGraph from JSON file using GraphData model."""
    path = Path(path)
    if not path.exists():
        return nx.DiGraph()
        
    text = path.read_text(encoding="utf-8")
    try:
        graph_data = GraphData.model_validate_json(text)
    except ValidationError:
        # Fallback for old format? Or just fail as per instructions "backward compatibility is NOT required"
        print(f"Warning: Failed to validate graph data from {path}. Returning empty graph.")
        return nx.DiGraph()

    G: nx.DiGraph = nx.DiGraph()
    G.graph.update(graph_data.metadata)
    
    for node in graph_data.nodes:
        G.add_node(node.id, url=node.url, title=node.title)
        
    for edge in graph_data.edges:
        # Store complex objects directly in the graph
        G.add_edge(
            edge.source, 
            edge.target, 
            selector=edge.selector,
            action=edge.action,
            intent=edge.intent,
            intent_failure_reason=edge.intent_failure_reason,
            data_key=edge.data_key,
            constraints=edge.constraints
        )
        
    return G

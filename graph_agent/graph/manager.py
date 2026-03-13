import networkx as nx
from typing import Dict, Any, Optional


class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_state_node(
        self, fingerprint: str, url: str, screenshot: Optional[str] = None, **kwargs
    ) -> str:
        """Adds a state node to the graph."""
        self.graph.add_node(fingerprint, url=url, screenshot=screenshot, **kwargs)
        return fingerprint

    def add_transition(
        self, source: str, target: str, action: Dict[str, Any], **kwargs
    ) -> None:
        """Adds a transition edge between two states."""
        self.graph.add_edge(source, target, action=action, **kwargs)

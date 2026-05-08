"""Cartography: LLM-first web application exploration and mapping.

Canonical entry point: :func:`run_mapping` orchestrates multi-page
exploration using :class:`ReActExplorer` for in-page deep dives.
:class:`NLResolver` is exposed for web/pathfinding consumers that need
to resolve natural-language intents against the discovered graph.
"""

from graph_agent.cartography.react_explorer import ReActExplorer
from graph_agent.cartography.nl_resolver import NLResolver


def run_mapping(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy wrapper to avoid eager import of runner at package init."""
    from graph_agent.cartography.runner import run_mapping as _run_mapping
    return _run_mapping(*args, **kwargs)


__all__ = [
    "run_mapping",
    "ReActExplorer",
    "NLResolver",
]

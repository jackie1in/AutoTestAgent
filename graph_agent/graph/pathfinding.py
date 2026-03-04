"""Path finding by user intent: get_path_from_intent."""

from __future__ import annotations

import networkx as nx

from graph_agent.models import GraphEdge, Intent, ActionType, ElementConstraints

LOW_CONFIDENCE_THRESHOLD = 0.4
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "fill_username": ("fill_username", "username", "填写用户名", "输入用户名"),
    "fill_password": ("fill_password", "password", "填写密码", "输入密码"),
    "submit_login": ("submit_login", "login", "登录", "提交登录"),
    "logout": ("logout", "log out", "登出", "退出"),
    "search": ("search", "搜索", "查询"),
    "click": ("click", "点击"),
    "fill": ("fill", "输入", "填写"),
    "navigate": ("navigate", "跳转", "访问"),
}


def _key_matches(intent_key: str | None, user_query: str) -> bool:
    """True if user query semantically maps to intent key aliases."""
    if not intent_key:
        return False
    key = intent_key.lower()
    query = (user_query or "").lower()
    aliases = _KEY_ALIASES.get(key, (key,))
    return any(alias in query or query in alias for alias in aliases if alias)


def _summary_matches(intent: Intent | None, user_query: str) -> bool:
    """Fallback summary match using substring, case-insensitive."""
    if not intent:
        return False
    summary = (intent.summary or "").lower()
    query = (user_query or "").lower()
    return query in summary or summary in query


def _match_score(intent: Intent | None, user_query: str) -> float:
    """Score intent-query match; key match preferred, low confidence down-weighted."""
    if not intent:
        return 0.0
    confidence = intent.confidence if intent.confidence is not None else 0.5
    confidence_factor = 1.0 if confidence >= LOW_CONFIDENCE_THRESHOLD else 0.35

    if _key_matches(intent.key, user_query):
        return 1.0 * confidence_factor
    if _summary_matches(intent, user_query):
        return 0.7 * confidence_factor
    return 0.0


def _edge_to_model(u: str, v: str, data: dict) -> GraphEdge:
    """Build GraphEdge model from graph data."""
    intent = data.get("intent")
    if not isinstance(intent, Intent):
        intent = None
        
    constraints = data.get("constraints")
    if not isinstance(constraints, ElementConstraints) and constraints is not None:
         # Should not happen if loaded correctly, but safe fallback
         if isinstance(constraints, dict):
             constraints = ElementConstraints(**constraints)
         else:
             constraints = None

    return GraphEdge(
        source=u,
        target=v,
        selector=data.get("selector", ""),
        action=data.get("action", ActionType.UNKNOWN),
        intent=intent,
        intent_failure_reason=data.get("intent_failure_reason"),
        data_key=data.get("data_key"),
        constraints=constraints
    )


def get_path_from_intent(user_query: str, graph: nx.DiGraph) -> list[GraphEdge]:
    """Return list of GraphEdges from entry to exit matching intent, or [].

    Entry = node with in_degree 0.
    Match priority: intent.key > intent.summary.
    Skips intent=None edges for matching but traverses through them to reach matching edges.
    """
    if graph.number_of_edges() == 0:
        return []

    # Find all nodes with in_degree 0 as potential start nodes
    entries = [n for n in graph if graph.in_degree(n) == 0]
    if not entries:
        entries = list(graph.nodes())[:1]

    def dfs(
        node: str,
        path: list[GraphEdge],
        visited: set[str],
        has_matched: bool,
    ) -> list[GraphEdge] | None:
        matching: list[tuple[float, str, str, dict]] = []
        traversal: list[tuple[str, str, dict]] = []
        for u, v, data in graph.out_edges(node, data=True):
            intent = data.get("intent")
            if intent is None or not isinstance(intent, Intent):
                # Skip for matching; collect for traversal (T6: empty intent tolerance)
                traversal.append((u, v, data))
                continue
            score = _match_score(intent, user_query)
            if score > 0:
                matching.append((score, u, v, data))

        # Try matching edges first
        matching.sort(key=lambda x: x[0], reverse=True)
        for _score, u, v, data in matching:
            if v in visited:
                continue
            visited.add(v)
            path.append(_edge_to_model(u, v, data))
            result = dfs(v, path, visited, True)
            if result is not None:
                return result
            path.pop()
            visited.discard(v)

        # If we have a path with at least one match, return it (greedy)
        if has_matched and path:
            return list(path)

        # Traverse through null-intent edges to reach nodes with matching edges
        for u, v, data in traversal:
            if v in visited:
                continue
            visited.add(v)
            path.append(_edge_to_model(u, v, data))
            result = dfs(v, path, visited, has_matched)
            if result is not None:
                return result
            path.pop()
            visited.discard(v)

        return None

    for start in entries:
        visited = {start}
        result = dfs(start, [], visited, False)
        if result is not None:
            return result

    for node in graph.nodes():
        if node in entries:
            continue
        visited = {node}
        result = dfs(node, [], visited, False)
        if result is not None:
            return result

    return []

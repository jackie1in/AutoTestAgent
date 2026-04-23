"""Path finding by user intent: get_path_from_intent.

Operates directly on a list of GraphEdge objects (from Neo4j query results)
without requiring an intermediate SimpleGraph.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from graph_agent.models import (
    ActionType,
    Intent,
    ElementConstraints,
    ElementSnapshot,
    FrameLocatorSnapshot,
    GraphEdge,
    TabActionType,
)

LOW_CONFIDENCE_THRESHOLD = 0.4
MATCHABLE_CONFIDENCE_THRESHOLD = 0.5
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "fill_username": ("fill_username", "username", "填写用户名", "输入用户名"),
    "fill_password": ("fill_password", "password", "填写密码", "输入密码"),
    "submit_login": ("submit_login", "login", "登录", "提交登录"),
    "logout": ("logout", "log out", "登出", "退出"),
    "search": ("search", "搜索", "查询"),
    "click": ("click", "点击"),
    "fill": ("fill", "输入", "填写"),
    "navigate": ("navigate", "跳转", "访问"),
    "auth.login": ("auth.login", "submit_login", "login", "登录", "提交登录"),
    "auth.submit.login": (
        "auth.submit.login",
        "submit_login",
        "login",
        "登录",
        "提交登录",
    ),
    "auth.fill.username": (
        "auth.fill.username",
        "fill_username",
        "username",
        "用户名",
        "账号",
    ),
    "auth.fill.password": ("auth.fill.password", "fill_password", "password", "密码"),
    "auth.logout": ("auth.logout", "logout", "登出", "退出"),
    "elements.iframe.type": (
        "elements.iframe.type",
        "iframe.type",
        "iframe",
        "iframe 输入",
    ),
    "navigation.module.select": (
        "navigation.module.select",
        "登录后进入目标模块",
        "进入目标模块",
        "进入一级业务模块",
        "项目管理",
        "项目模块",
        "project.navigation.report_access",
        "report_access",
        "elements.navigation.select",
        "module.select",
    ),
    "elements.management.add": (
        "elements.management.add",
        "项目列表进入子项目并打开概览",
        "项目列表进入子项目",
        "项目进入子项目打开概览",
        "elements.navigation.select",
        "elements.add",
    ),
}


def _key_matches(intent_key: str | None, user_query: str) -> bool:
    """True if user query semantically maps to intent key aliases."""
    if not intent_key:
        return False
    key = intent_key.lower()
    query = (user_query or "").lower()
    module_flow_terms = ("进入", "目标模块", "一级业务模块")
    if any(term in query for term in module_flow_terms):
        if "login" in key or "auth" in key:
            return False
        if "module" in key:
            return True
    aliases = _KEY_ALIASES.get(key, (key,))
    if any(alias in query or query in alias for alias in aliases if alias):
        return True

    submit_terms = ("submit", "提交")
    if any(term in query for term in submit_terms) and not any(
        term in key for term in submit_terms
    ):
        return False

    login_terms = ("登录", "login", "signin", "sign in", "auth")
    if any(term in query for term in login_terms) and any(
        term in key for term in ("login", "auth")
    ):
        return True
    password_terms = ("密码", "password", "passwd")
    if any(term in query for term in password_terms) and "password" in key:
        return True
    username_terms = ("用户名", "账号", "username", "user")
    if any(term in query for term in username_terms) and "username" in key:
        return True
    logout_terms = ("登出", "退出", "logout", "log out", "signout", "sign out")
    if any(term in query for term in logout_terms) and "logout" in key:
        return True
    return False


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
    if confidence < MATCHABLE_CONFIDENCE_THRESHOLD:
        return 0.0
    confidence_factor = 1.0 if confidence >= LOW_CONFIDENCE_THRESHOLD else 0.35

    if _key_matches(intent.key, user_query):
        return 1.0 * confidence_factor
    if _summary_matches(intent, user_query):
        return 0.7 * confidence_factor
    return 0.0


def _parse_frame_path(raw: object) -> list[FrameLocatorSnapshot]:
    """Parse frame_path from graph data for playback context."""
    if not raw:
        return []
    result: list[FrameLocatorSnapshot] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, FrameLocatorSnapshot):
            result.append(item)
        elif isinstance(item, dict):
            result.append(FrameLocatorSnapshot(**item))
    return result


def neo4j_transition_to_edge_data(t: dict[str, Any]) -> dict[str, Any]:
    """Convert a Neo4j transition record into graph edge data for playback."""
    import json

    intent = t.get("intent")
    if isinstance(intent, dict):
        try:
            intent = Intent(**intent)
        except Exception:
            intent = None

    element = None
    element_snapshot_raw = t.get("element_snapshot")
    if element_snapshot_raw:
        try:
            parsed = json.loads(element_snapshot_raw)
            if isinstance(parsed, dict):
                element = ElementSnapshot(**parsed)
        except Exception:
            pass

    frame_path: list[FrameLocatorSnapshot] = []
    frame_path_raw = t.get("frame_path")
    if frame_path_raw:
        try:
            parsed = json.loads(frame_path_raw)
            if isinstance(parsed, list):
                frame_path = [
                    FrameLocatorSnapshot(**item) if isinstance(item, dict) else item
                    for item in parsed
                ]
        except Exception:
            pass

    tab_action = None
    tab_action_raw = t.get("tab_action")
    if isinstance(tab_action_raw, str) and tab_action_raw:
        try:
            tab_action = TabActionType(tab_action_raw)
        except ValueError:
            tab_action = None

    action_raw = t.get("action", ActionType.UNKNOWN)
    action = action_raw if isinstance(action_raw, ActionType) else ActionType(str(action_raw))

    return {
        "edge_id": t.get("id"),
        "step_index": t.get("step_index"),
        "source_url": t.get("source_url"),
        "target_url": t.get("target_url"),
        "selector": t.get("selector", ""),
        "action": action,
        "tab_id": t.get("tab_id", "tab-0"),
        "target_tab_id": t.get("target_tab_id"),
        "tab_action": tab_action,
        "frame_path": frame_path,
        "intent": intent,
        "intent_failure_reason": t.get("intent_failure_reason"),
        "param_name": t.get("param_name"),
        "action_value": t.get("action_value"),
        "thought": t.get("thought"),
        "element": element,
    }


def _edge_to_model(u: str, v: str, data: dict) -> GraphEdge:
    """Build GraphEdge model from graph data (preserves tab/frame context for playback)."""
    intent = data.get("intent")
    if not isinstance(intent, Intent):
        intent = None

    constraints = data.get("constraints")
    if not isinstance(constraints, ElementConstraints) and constraints is not None:
        if isinstance(constraints, dict):
            constraints = ElementConstraints(**constraints)
        else:
            constraints = None

    element = data.get("element")
    if not isinstance(element, ElementSnapshot) and element is not None:
        if isinstance(element, dict):
            element = ElementSnapshot(**element)
        else:
            element = None

    tab_action_raw = data.get("tab_action")
    tab_action = (
        TabActionType(tab_action_raw)
        if isinstance(tab_action_raw, str) and tab_action_raw
        else None
    )

    return GraphEdge(
        edge_id=data.get("edge_id"),
        step_index=data.get("step_index"),
        source=u,
        target=v,
        source_url=data.get("source_url"),
        target_url=data.get("target_url"),
        selector=data.get("selector", ""),
        action=data.get("action", ActionType.UNKNOWN),
        tab_id=data.get("tab_id", "tab-0"),
        target_tab_id=data.get("target_tab_id"),
        tab_action=tab_action,
        frame_path=_parse_frame_path(data.get("frame_path")),
        intent=intent,
        intent_failure_reason=data.get("intent_failure_reason"),
        param_name=data.get("param_name"),
        action_value=data.get("action_value"),
        element=element,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Adjacency helpers (replacing SimpleGraph)
# ---------------------------------------------------------------------------

Adjacency = dict[str, list[GraphEdge]]


def _build_adjacency(edges: list[GraphEdge]) -> Adjacency:
    """Build adjacency list from edges."""
    adj: Adjacency = {}
    for edge in edges:
        adj.setdefault(edge.source, []).append(edge)
    for out_edges in adj.values():
        out_edges.sort(key=lambda e: (e.step_index if e.step_index is not None else 10**9))
    return adj


def _find_entry_nodes(edges: list[GraphEdge]) -> list[str]:
    """Find nodes with in-degree == 0."""
    in_degree: Counter[str] = Counter()
    nodes: set[str] = set()
    for edge in edges:
        in_degree[edge.target] += 1
        nodes.add(edge.source)
        nodes.add(edge.target)
    entries = [n for n in nodes if in_degree[n] == 0]
    return entries if entries else list(nodes)[:1]


def _shortest_path(
    adjacency: Adjacency,
    from_node: str,
    to_node: str,
) -> list[str] | None:
    """BFS shortest node path."""
    if from_node == to_node:
        return [from_node]
    queue = deque([(from_node, [from_node])])
    visited = {from_node}
    while queue:
        node, path = queue.popleft()
        for edge in adjacency.get(node, []):
            if edge.target in visited:
                continue
            new_path = path + [edge.target]
            if edge.target == to_node:
                return new_path
            visited.add(edge.target)
            queue.append((edge.target, new_path))
    return None


def _get_connecting_path(
    adjacency: Adjacency,
    from_node: str,
    to_node: str,
) -> list[GraphEdge]:
    """Return GraphEdges along shortest path from from_node to to_node."""
    if from_node == to_node:
        return []
    node_path = _shortest_path(adjacency, from_node, to_node)
    if node_path is None:
        return []
    edges: list[GraphEdge] = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        for edge in adjacency.get(u, []):
            if edge.target == v:
                edges.append(edge)
                break
    return edges


def _iter_out_edges(adjacency: Adjacency, node: str) -> list[GraphEdge]:
    """Return sorted outgoing edges for a node."""
    return list(adjacency.get(node, []))


def _to_step_index(raw: object) -> int:
    """Normalize step index for ordering/comparison."""
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Core DFS pathfinding (previously depended on SimpleGraph)
# ---------------------------------------------------------------------------

def _get_path_from_atomic_intent(
    user_query: str, edges: list[GraphEdge]
) -> list[GraphEdge]:
    """Return list of GraphEdges from entry to exit matching intent, or [].

    Entry = node with in_degree 0.
    Match priority: intent.key > intent.summary.
    Skips intent=None edges for matching but traverses through them to reach matching edges.
    """
    if not edges:
        return []

    adjacency = _build_adjacency(edges)
    entries = _find_entry_nodes(edges)

    def dfs(
        node: str,
        path: list[GraphEdge],
        visited: set[str],
        has_matched: bool,
        used_edges: set[tuple[str, str, int, str]],
    ) -> list[GraphEdge] | None:
        matching: list[tuple[float, GraphEdge]] = []
        traversal: list[GraphEdge] = []
        for edge in _iter_out_edges(adjacency, node):
            step_index = _to_step_index(edge.step_index)
            edge_key = (edge.source, edge.target, step_index, edge.selector)
            if edge_key in used_edges:
                continue
            intent = edge.intent
            if intent is None or not isinstance(intent, Intent):
                traversal.append(edge)
                continue
            score = _match_score(intent, user_query)
            if score > 0:
                matching.append((score, edge))
            else:
                traversal.append(edge)

        # Try matching edges first
        matching.sort(key=lambda x: x[0], reverse=True)
        for _score, edge in matching:
            step_index = _to_step_index(edge.step_index)
            edge_key = (edge.source, edge.target, step_index, edge.selector)
            v = edge.target
            u = edge.source
            if v in visited and u == v:
                candidate = list(path)
                candidate.append(edge)
                return candidate
            if v in visited:
                continue

            # Inject earlier same-node self-loops as prerequisites before a matched edge.
            prereq_keys: list[tuple[str, str, int, str]] = []
            prereq_edges: list[GraphEdge] = []
            for pre_edge in _iter_out_edges(adjacency, node):
                p_step = _to_step_index(pre_edge.step_index)
                if pre_edge.source != node or pre_edge.target != node:
                    continue
                if p_step < 0 or p_step >= step_index:
                    continue
                p_key = (pre_edge.source, pre_edge.target, p_step, pre_edge.selector)
                if p_key in used_edges:
                    continue
                prereq_keys.append(p_key)
                prereq_edges.append(pre_edge)

            for idx, p_model in enumerate(prereq_edges):
                used_edges.add(prereq_keys[idx])
                path.append(p_model)
            visited.add(v)
            used_edges.add(edge_key)
            path.append(edge)
            result = dfs(v, path, visited, True, used_edges)
            if result is not None:
                return result
            path.pop()
            used_edges.discard(edge_key)
            visited.discard(v)
            for idx in range(len(prereq_edges) - 1, -1, -1):
                path.pop()
                used_edges.discard(prereq_keys[idx])

        # If we have a path with at least one match, return it (greedy)
        if has_matched and path:
            return list(path)

        # Traverse through null-intent edges to reach nodes with matching edges
        for edge in traversal:
            step_index = _to_step_index(edge.step_index)
            edge_key = (edge.source, edge.target, step_index, edge.selector)
            if edge_key in used_edges:
                continue
            v = edge.target
            u = edge.source
            if v in visited and u == v:
                used_edges.add(edge_key)
                path.append(edge)
                result = dfs(v, path, visited, has_matched, used_edges)
                if result is not None:
                    return result
                path.pop()
                used_edges.discard(edge_key)
                continue
            if v in visited:
                continue
            visited.add(v)
            used_edges.add(edge_key)
            path.append(edge)
            result = dfs(v, path, visited, has_matched, used_edges)
            if result is not None:
                return result
            path.pop()
            used_edges.discard(edge_key)
            visited.discard(v)

        return None

    for start in entries:
        visited = {start}
        result = dfs(start, [], visited, False, set())
        if result is not None:
            return result

    # Fallback: try all nodes as start
    all_nodes = {e.source for e in edges} | {e.target for e in edges}
    for node in all_nodes:
        if node in entries:
            continue
        visited = {node}
        result = dfs(node, [], visited, False, set())
        if result is not None:
            return result

    return []


def _prepend_entry_path(
    edges: list[GraphEdge], expanded: list[GraphEdge]
) -> list[GraphEdge]:
    """If *expanded* doesn't start at an entry node, prepend a connecting path."""
    if not expanded:
        return expanded
    first_source = expanded[0].source
    adjacency = _build_adjacency(edges)
    entries = _find_entry_nodes(edges)
    # Check if first_source is already an entry
    if first_source in entries:
        return expanded
    seen_edge_ids = {e.edge_id for e in expanded if e.edge_id}
    for entry in entries:
        prefix = _get_connecting_path(adjacency, entry, first_source)
        if prefix:
            unique_prefix = [e for e in prefix if not e.edge_id or e.edge_id not in seen_edge_ids]
            if unique_prefix:
                return unique_prefix + expanded
    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_path_from_query(user_query: str, edges: list[GraphEdge]) -> list[GraphEdge]:
    """Resolve query to a replayable path from the given edges."""
    path = _get_path_from_atomic_intent(user_query, edges)
    if path:
        return _prepend_entry_path(edges, path)
    return []


def get_path_from_intent(user_query: str, edges: list[GraphEdge]) -> list[GraphEdge]:
    """Backward-compatible wrapper."""
    return get_path_from_query(user_query, edges)


async def get_path_from_nl_query(
    user_query: str,
    edges: list[GraphEdge],
    *,
    driver: Any | None = None,
    embedder: Any | None = None,
    top_k: int = 1,
) -> list[GraphEdge]:
    """Resolve natural language query via GraphRAG first, fallback to keyword matching.

    Args:
        user_query: Natural language description, e.g. "建任务", "登录并创建项目".
        edges: List of GraphEdge from Neo4j query.
        driver: Neo4j async driver (optional). If provided with embedder,
            GraphRAG semantic search is attempted first.
        embedder: Embedding provider (optional). Must be provided alongside driver.
        top_k: Number of top GraphRAG candidates to consider.

    Returns:
        List of GraphEdge representing the resolved playback path.
        Empty list if no match found.
    """
    if driver is not None and embedder is not None:
        try:
            from graph_agent.cartography.nl_resolver import NLResolver

            resolver = NLResolver(driver, embedder)
            results = await resolver.resolve(user_query, top_k=top_k)
            if results:
                best = results[0]
                intent_key = best.get("key", "")
                if intent_key:
                    path = get_path_from_query(intent_key, edges)
                    if path:
                        return path
        except Exception:
            pass

    return get_path_from_query(user_query, edges)

"""Path finding by user intent: get_path_from_intent."""

from __future__ import annotations

import networkx as nx

from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    ElementConstraints,
    ElementSnapshot,
    FrameLocatorSnapshot,
    GraphEdge,
    Intent,
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
    # 意图 B: 登录后进入目标模块 (PRD 8.3)
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
    # 意图 C: 项目列表进入子项目并打开概览 (PRD 8.3)
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
    # 意图 B: "登录后进入目标模块" 应匹配 navigation.module.select，不匹配仅 auth.login
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

    # Concept-level fallback for modern dot-keys.
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


def _template_summary_matches(summary: str, user_query: str) -> bool:
    """Fallback summary match for business templates."""
    text = (summary or "").lower()
    query = (user_query or "").lower()
    return bool(text) and (query in text or text in query)


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


def _template_match_score(template: BusinessTemplate, user_query: str) -> float:
    """Score a business template against the user query."""
    confidence = template.confidence if template.confidence is not None else 0.5
    if confidence < MATCHABLE_CONFIDENCE_THRESHOLD:
        return 0.0
    confidence_factor = 1.0 if confidence >= LOW_CONFIDENCE_THRESHOLD else 0.35
    if _key_matches(template.business_key, user_query):
        return 1.2 * confidence_factor
    if _template_summary_matches(template.summary, user_query):
        return 0.8 * confidence_factor
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


def _coerce_template(raw: object) -> BusinessTemplate | None:
    """Normalize stored business template metadata into a model."""
    if isinstance(raw, BusinessTemplate):
        return raw
    if isinstance(raw, dict):
        try:
            return BusinessTemplate(**raw)
        except Exception:
            return None
    return None


def _iter_business_templates(graph: nx.Graph) -> list[BusinessTemplate]:
    """Load stored business templates from graph metadata."""
    raw_templates = graph.graph.get("business_templates", [])
    if not isinstance(raw_templates, list):
        return []
    templates: list[BusinessTemplate] = []
    for raw in raw_templates:
        template = _coerce_template(raw)
        if template is not None:
            templates.append(template)
    return templates


def _prefer_template_candidate(
    current: BusinessTemplate | None,
    candidate: BusinessTemplate,
) -> BusinessTemplate:
    """Choose the preferred template for the same business key."""
    if current is None:
        return candidate
    current_conf = current.confidence if current.confidence is not None else 0.0
    candidate_conf = candidate.confidence if candidate.confidence is not None else 0.0
    if candidate_conf > current_conf:
        return candidate
    if candidate_conf < current_conf:
        return current
    if candidate.path_length > current.path_length:
        return candidate
    return current


def _template_lookup(templates: list[BusinessTemplate]) -> dict[str, BusinessTemplate]:
    """Index templates by business key using the strongest candidate."""
    lookup: dict[str, BusinessTemplate] = {}
    for template in templates:
        lookup[template.business_key] = _prefer_template_candidate(
            lookup.get(template.business_key),
            template,
        )
    return lookup


def _expand_template_steps(
    template: BusinessTemplate, graph: nx.Graph
) -> list[GraphEdge]:
    """Expand stored template steps back into GraphEdge models.

    Uses (edge_id, source, target) as lookup key to handle graphs with duplicate
    edge_ids across different node pairs (e.g. merged mapping runs).
    """
    edge_lookup: dict[tuple[str, str, str], GraphEdge] = {}
    if isinstance(graph, nx.MultiDiGraph):
        rows = [(u, v, data) for u, v, _k, data in graph.edges(keys=True, data=True)]
    else:
        rows = [(u, v, data) for u, v, data in graph.edges(data=True)]
    for u, v, data in rows:
        edge = _edge_to_model(str(u), str(v), data)
        if edge.edge_id:
            key = (edge.edge_id, str(u), str(v))
            edge_lookup[key] = edge
    expanded: list[GraphEdge] = []
    for step in template.steps:
        if not step.edge_id:
            return []
        key = (step.edge_id, step.source, step.target)
        if key not in edge_lookup:
            return []
        expanded.append(edge_lookup[key])
    return expanded


def _expand_template_to_path(
    template: BusinessTemplate,
    graph: nx.Graph,
    templates_by_key: dict[str, BusinessTemplate],
    *,
    active_keys: set[str] | None = None,
    seen_edge_ids: set[str] | None = None,
) -> list[GraphEdge]:
    """Expand template plus explicit prerequisites into a replayable edge path.

    When a dependency's exit_node differs from the next template's entry_node,
    inserts the connecting path from the graph so playback can reach the correct state.
    """
    active = active_keys if active_keys is not None else set()
    seen = seen_edge_ids if seen_edge_ids is not None else set()
    if template.business_key in active:
        return []

    active.add(template.business_key)
    expanded: list[GraphEdge] = []
    try:
        last_exit_node: str | None = None
        for dependency_key in template.depends_on:
            dependency = templates_by_key.get(dependency_key)
            if dependency is None:
                return []
            if last_exit_node is not None and last_exit_node != dependency.entry_node:
                connecting = _get_connecting_path(
                    graph, last_exit_node, dependency.entry_node
                )
                for edge in connecting:
                    edge_id = edge.edge_id
                    if edge_id and edge_id in seen:
                        continue
                    if edge_id:
                        seen.add(edge_id)
                    expanded.append(edge)
            dependency_path = _expand_template_to_path(
                dependency,
                graph,
                templates_by_key,
                active_keys=active,
            )
            if dependency.steps and not dependency_path:
                return []
            for edge in dependency_path:
                edge_id = edge.edge_id
                if edge_id and edge_id in seen:
                    continue
                if edge_id:
                    seen.add(edge_id)
                expanded.append(edge)
            last_exit_node = dependency.exit_node

        if last_exit_node is not None and last_exit_node != template.entry_node:
            connecting = _get_connecting_path(
                graph, last_exit_node, template.entry_node
            )
            for edge in connecting:
                edge_id = edge.edge_id
                if edge_id and edge_id in seen:
                    continue
                if edge_id:
                    seen.add(edge_id)
                expanded.append(edge)

        for edge in _expand_template_steps(template, graph):
            edge_id = edge.edge_id
            if edge_id and edge_id in seen:
                continue
            if edge_id:
                seen.add(edge_id)
            expanded.append(edge)
    finally:
        active.discard(template.business_key)
    return expanded


def _get_connecting_path(
    graph: nx.Graph, from_node: str, to_node: str
) -> list[GraphEdge]:
    """Return GraphEdges along shortest path from from_node to to_node, or [] if no path or same node."""
    if from_node == to_node:
        return []
    if from_node not in graph or to_node not in graph:
        return []
    try:
        node_path = nx.shortest_path(graph, from_node, to_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    edges: list[GraphEdge] = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        for _u, _v, data in _iter_out_edges(graph, u):
            if str(_v) == str(v):
                edges.append(_edge_to_model(str(_u), str(_v), data))
                break
    return edges


def _iter_out_edges(graph: nx.Graph, node: str) -> list[tuple[str, str, dict]]:
    """Unified out-edge iterator for DiGraph and MultiDiGraph."""
    if isinstance(graph, nx.MultiDiGraph):
        rows = [
            (u, v, data)
            for u, v, _k, data in graph.out_edges(node, keys=True, data=True)
        ]
    else:
        rows = [(u, v, data) for u, v, data in graph.out_edges(node, data=True)]

    def _step_order(item: tuple[str, str, dict]) -> int:
        raw = item[2].get("step_index")
        if raw is None:
            return 10**9
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 10**9

    return sorted(rows, key=_step_order)


def _to_step_index(raw: object) -> int:
    """Normalize step index for ordering/comparison."""
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _get_path_from_atomic_intent(user_query: str, graph: nx.DiGraph) -> list[GraphEdge]:
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
        used_edges: set[tuple[str, str, int, str]],
    ) -> list[GraphEdge] | None:
        matching: list[tuple[float, str, str, dict]] = []
        traversal: list[tuple[str, str, dict]] = []
        for u, v, data in _iter_out_edges(graph, node):
            step_index_raw = data.get("step_index")
            try:
                step_index = int(step_index_raw) if step_index_raw is not None else -1
            except (TypeError, ValueError):
                step_index = -1
            edge_key = (str(u), str(v), step_index, str(data.get("selector", "")))
            if edge_key in used_edges:
                continue
            intent = data.get("intent")
            if intent is None or not isinstance(intent, Intent):
                # Skip for matching; collect for traversal (T6: empty intent tolerance)
                traversal.append((u, v, data))
                continue
            score = _match_score(intent, user_query)
            if score > 0:
                matching.append((score, u, v, data))
            else:
                # Non-matching intent: still traversable to reach matching edges
                traversal.append((u, v, data))

        # Try matching edges first
        matching.sort(key=lambda x: x[0], reverse=True)
        for _score, u, v, data in matching:
            step_index = _to_step_index(data.get("step_index"))
            edge_key = (str(u), str(v), step_index, str(data.get("selector", "")))
            # Allow returning a self-loop match at current node.
            # This is common for same-page actions (e.g., modal accept/filter click).
            if v in visited and u == v:
                candidate = list(path)
                candidate.append(_edge_to_model(u, v, data))
                return candidate
            if v in visited:
                continue

            # Inject earlier same-node self-loops as prerequisites before a matched edge.
            prereq_keys: list[tuple[str, str, int, str]] = []
            prereq_edges: list[GraphEdge] = []
            for pu, pv, pdata in _iter_out_edges(graph, node):
                p_step = _to_step_index(pdata.get("step_index"))
                if pu != node or pv != node:
                    continue
                if p_step < 0 or p_step >= step_index:
                    continue
                p_key = (str(pu), str(pv), p_step, str(pdata.get("selector", "")))
                if p_key in used_edges:
                    continue
                prereq_keys.append(p_key)
                prereq_edges.append(_edge_to_model(pu, pv, pdata))

            for idx, p_model in enumerate(prereq_edges):
                used_edges.add(prereq_keys[idx])
                path.append(p_model)
            visited.add(v)
            used_edges.add(edge_key)
            path.append(_edge_to_model(u, v, data))
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
        for u, v, data in traversal:
            step_index = _to_step_index(data.get("step_index"))
            edge_key = (str(u), str(v), step_index, str(data.get("selector", "")))
            if edge_key in used_edges:
                continue
            if v in visited and u == v:
                # Keep self-loop prerequisites (e.g., checkbox accept) in path once,
                # then continue searching from the same node.
                used_edges.add(edge_key)
                path.append(_edge_to_model(u, v, data))
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
            path.append(_edge_to_model(u, v, data))
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

    for node in graph.nodes():
        if node in entries:
            continue
        visited = {node}
        result = dfs(node, [], visited, False, set())
        if result is not None:
            return result

    return []


def _prepend_entry_path(
    graph: nx.Graph, expanded: list[GraphEdge]
) -> list[GraphEdge]:
    """If *expanded* doesn't start at a graph entry node, prepend a connecting path."""
    if not expanded:
        return expanded
    first_source = expanded[0].source
    if graph.in_degree(first_source) == 0:
        return expanded
    entries = [n for n in graph if graph.in_degree(n) == 0]
    seen_edge_ids = {e.edge_id for e in expanded if e.edge_id}
    for entry in entries:
        prefix = _get_connecting_path(graph, entry, first_source)
        if prefix:
            unique_prefix = [e for e in prefix if not e.edge_id or e.edge_id not in seen_edge_ids]
            if unique_prefix:
                return unique_prefix + expanded
    return expanded


def get_path_from_query(
    user_query: str,
    graph: nx.DiGraph,
    *,
    prefer_templates: bool = True,
) -> list[GraphEdge]:
    """Resolve query to path, preferring business templates over atomic intents."""
    if prefer_templates:
        templates = _iter_business_templates(graph)
        templates_by_key = _template_lookup(templates)
        ranked: list[tuple[float, int, BusinessTemplate]] = []
        for template in templates:
            score = _template_match_score(template, user_query)
            if score > 0:
                ranked.append((score, template.path_length, template))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _score, _path_length, template in ranked:
            expanded = _expand_template_to_path(
                template,
                graph,
                templates_by_key,
            )
            if expanded:
                return _prepend_entry_path(graph, expanded)
    return _get_path_from_atomic_intent(user_query, graph)


def get_path_from_intent(user_query: str, graph: nx.DiGraph) -> list[GraphEdge]:
    """Backward-compatible wrapper for template-first query resolution."""
    return get_path_from_query(user_query, graph)

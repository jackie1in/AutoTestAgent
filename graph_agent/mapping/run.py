"""Run browser-use Agent for mapping: explore flow, build DiGraph from history."""

from __future__ import annotations

import json
import os
import asyncio
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from networkx import MultiDiGraph

from graph_agent.graph.templates import generate_business_templates, store_business_templates
from graph_agent.graph.io import save_graph, load_graph
from graph_agent.llm import get_llm
from graph_agent.mapping.parser import parse_browser_use_step, infer_intent_for_context
from graph_agent.mapping.scout import extract_derived_urls, run_scout, run_scout_multi
from graph_agent.models import ActionType

# Generic task template for site-agnostic mapping.
DEFAULT_TASK_TEMPLATE = (
    "从起始URL开始探索核心业务流程：{start_url}。"
    "探索阶段优先使用UI交互动作：click/fill/navigate/select。"
    "在点击菜单、列表项、详情入口、子项目入口、概览入口后，继续探索进入的派生页面，不要停留在入口页。"
    "记录每一步的 selector、业务意图、动作类型及目标状态。"
    "严禁在探索过程中使用 read_file/write_file/replace_file 等文件工具；仅允许在最终 done 时输出结论。"
    "遇到无法完成的表单（缺少必填数据）或潜在破坏性操作（删除、清空、提交不可逆变更）时立即停止，"
    "并在最终回复中写明原因（例如：Stopped: unfillable form / Stopped: would delete data）。"
)

FILTERED_ACTION_KEYS = {"read_file", "write_file", "done", "unknown"}
DISALLOWED_RUNTIME_ACTION_KEYS = {"read_file", "write_file", "replace_file", "done"}


def _clean_url(url: str) -> str:
    """Strip query parameters and hash fragments from URL to ensure stable Node IDs."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        # Keep scheme, netloc, path. Drop params, query, fragment.
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return url


def _extract_next_goal(thought: dict | object) -> str:
    """Extract next_goal text from thought dict/object."""
    if isinstance(thought, dict):
        value = thought.get("next_goal", "") or ""
    else:
        value = getattr(thought, "next_goal", "") or ""
    return str(value).strip()


def _extract_action_key(action: dict | object) -> str:
    """Extract coarse action key for state naming."""
    data: dict[str, Any]
    if isinstance(action, dict):
        data = action
    elif hasattr(action, "model_dump"):
        converted = action.model_dump()
        if not isinstance(converted, dict):
            return "unknown"
        data = converted
    elif hasattr(action, "dict"):
        converted = action.dict()
        if not isinstance(converted, dict):
            return "unknown"
        data = converted
    elif hasattr(action, "__dict__"):
        converted = action.__dict__
        if not isinstance(converted, dict):
            return "unknown"
        data = converted
    else:
        return "unknown"
    preferred = (
        "click",
        "click_element",
        "input",
        "input_text",
        "navigate",
        "navigate_browser",
        "go_back",
        "select_dropdown",
        "send_keys",
    )
    for key in preferred:
        if key in data:
            return key
    return next(iter(data.keys()), "unknown")


def _collect_history_snapshots(history: Any) -> tuple[list[dict[str, Any]], list[dict | object], list[str]]:
    """Collect action/thought/url snapshots from history as plain lists."""
    raw_actions = list(history.model_actions()) if history else []
    actions = [_action_to_dict(a) for a in raw_actions]
    thoughts = list(history.model_thoughts()) if history else []
    try:
        urls = list(history.urls()) if history else []
    except Exception:
        urls = []
    return actions, thoughts, urls


def _runtime_filter_snapshots(
    actions: list[dict[str, Any]],
    thoughts: list[dict | object],
    urls: list[str],
    disallowed_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict | object], list[str], int]:
    """Filter non-UI actions before parser/build stage to reduce noise."""
    blocked = disallowed_keys or DISALLOWED_RUNTIME_ACTION_KEYS
    keep_indexes: list[int] = []
    filtered_count = 0
    for idx, action in enumerate(actions):
        key = _extract_action_key(action)
        if key in blocked:
            filtered_count += 1
            continue
        keep_indexes.append(idx)

    filtered_actions = [actions[i] for i in keep_indexes]
    filtered_thoughts = [thoughts[i] if i < len(thoughts) else {} for i in keep_indexes]
    filtered_urls = [urls[i] if i < len(urls) else "" for i in keep_indexes]
    # Keep one trailing URL snapshot for i+1 lookups if available.
    if keep_indexes and urls:
        last = keep_indexes[-1] + 1
        if last < len(urls):
            filtered_urls.append(urls[last])
    filtered_urls = _stabilize_url_snapshots(filtered_urls)
    return filtered_actions, filtered_thoughts, filtered_urls, filtered_count


def _stabilize_url_snapshots(urls: list[str]) -> list[str]:
    """Fill missing URL snapshots using nearest valid http(s) neighbors."""
    if not urls:
        return []
    stabilized = [_clean_url(u or "") for u in urls]

    # Forward fill: use latest known concrete URL.
    last_http = ""
    for i, value in enumerate(stabilized):
        if _is_http_url(value):
            last_http = value
            continue
        if last_http:
            stabilized[i] = last_http

    # Backward fill: handle leading missing entries.
    next_http = ""
    for i in range(len(stabilized) - 1, -1, -1):
        value = stabilized[i]
        if _is_http_url(value):
            next_http = value
            continue
        if next_http:
            stabilized[i] = next_http
    return stabilized


def _resolve_mapping_url(url: str | None) -> str:
    """Resolve mapping URL: function arg first, then MAPPING_URL env, else raise."""
    value = (url or "").strip()
    if value:
        return value
    env_value = (os.getenv("MAPPING_URL") or "").strip()
    if env_value:
        return env_value
    raise ValueError("url is required. Provide --url or set MAPPING_URL.")


def _build_mapping_task(task: str | None, start_url: str) -> str:
    """Use custom task if provided; otherwise render generic task template."""
    custom_task = (task or "").strip()
    if custom_task:
        return custom_task
    return DEFAULT_TASK_TEMPLATE.format(start_url=start_url)


def _build_login_hint_from_env() -> str:
    """Build optional login hint from env; empty when no credentials configured."""
    username = (os.getenv("MAPPING_USERNAME") or "").strip()
    password = (os.getenv("MAPPING_PASSWORD") or "").strip()
    if not username and not password:
        return ""

    parts: list[str] = []
    if username:
        parts.append(f"username={username}")
    if password:
        parts.append(f"password={password}")
    credentials = ", ".join(parts)
    return (
        "若页面包含登录表单，优先使用以下测试账号完成登录："
        f"{credentials}。"
        "如字段名不同，请根据语义匹配对应输入框。"
    )


def _build_mapping_task_with_env_hints(task: str | None, start_url: str) -> str:
    """Build mapping task and append login hint only when env credentials exist."""
    base = _build_mapping_task(task, start_url)
    hint = _build_login_hint_from_env()
    if not hint:
        return base
    return f"{base}\n{hint}"


def _state_from_snapshot(step: int, raw_url: str, thought: dict | object, action: dict | object) -> tuple[str, str]:
    """Build opaque state id and normalized page URL from a snapshot."""
    cleaned = _clean_url(raw_url or "")

    goal = _extract_next_goal(thought)
    action_key = _extract_action_key(action).replace("_", " ")
    goal_brief = re.sub(r"\s+", " ", goal).strip()[:40]
    descriptor = f"{action_key} {goal_brief}".strip() or "unknown"
    fingerprint_src = f"{cleaned}|{step}|{action_key}|{goal}"
    fingerprint = hashlib.md5(fingerprint_src.encode("utf-8")).hexdigest()[:6]
    state_id = f"state-{step}-{fingerprint}"
    return state_id, cleaned


def _action_to_dict(action: dict | object) -> dict[str, Any]:
    """Convert action object to dict for robust downstream parsing."""
    if isinstance(action, dict):
        return action
    if hasattr(action, "model_dump"):
        converted = action.model_dump()
        if isinstance(converted, dict):
            return converted
    if hasattr(action, "dict"):
        converted = action.dict()
        if isinstance(converted, dict):
            return converted
    if hasattr(action, "__dict__") and isinstance(action.__dict__, dict):
        return action.__dict__
    return {}


def _build_neighbor_steps(
    idx: int,
    actions: list[dict[str, Any]],
    thoughts: list[dict | object],
    urls: list[str],
    window: int = 1,
) -> list[dict[str, str]]:
    """Build local step window around index for progressive context."""
    neighbors: list[dict[str, str]] = []
    left = max(0, idx - window)
    right = min(len(actions) - 1, idx + window)
    for j in range(left, right + 1):
        if j == idx:
            continue
        raw_action = actions[j] if j < len(actions) else {}
        action_dict = _action_to_dict(raw_action)
        action_key = _extract_action_key(action_dict)
        thought = thoughts[j] if j < len(thoughts) else {}
        src = urls[j] if j < len(urls) else ""
        tgt = urls[j + 1] if j + 1 < len(urls) else src
        neighbors.append(
            {
                "action": action_key,
                "selector": "",
                "source_url": _clean_url(src or ""),
                "target_url": _clean_url(tgt or ""),
                "thought": _extract_next_goal(thought),
            }
        )
    return neighbors


def _build_page_signals(source_url: str, target_url: str, action_key: str) -> dict[str, str]:
    """Build compact page-level signals for L2 inference."""
    from urllib.parse import urlparse

    src = urlparse(source_url) if source_url else None
    tgt = urlparse(target_url) if target_url else None
    return {
        "source_host": src.netloc if src else "",
        "source_path": src.path if src else "",
        "target_host": tgt.netloc if tgt else "",
        "target_path": tgt.path if tgt else "",
        "transition": f"{(src.path if src else '')} -> {(tgt.path if tgt else '')}",
        "action_key": action_key,
    }


def _is_http_url(value: str) -> bool:
    """Return True if value looks like a stable http(s) URL."""
    v = (value or "").strip()
    return v.startswith("http://") or v.startswith("https://")


def _resolve_target_state(
    i: int,
    urls: list[str],
    thoughts: list[dict | object],
    actions: list[dict[str, Any]],
) -> tuple[str, str]:
    """Resolve target node with best-effort real URL lookahead.

    Browser history sometimes misses immediate post-action URL and yields empty
    snapshots, which creates pseudo states and noisy edges. We try i+1 first;
    if empty/non-http, look ahead a few steps for the next concrete URL.
    """
    next_thought = thoughts[i + 1] if i + 1 < len(thoughts) else {}
    next_action = actions[i + 1] if i + 1 < len(actions) else {}
    raw_to = urls[i + 1] if i + 1 < len(urls) else ""
    if _is_http_url(raw_to):
        return _state_from_snapshot(i + 1, raw_to, next_thought, next_action)

    # Look ahead up to 3 steps to find a concrete URL.
    upper = min(len(urls), i + 4)
    for j in range(i + 2, upper):
        candidate = urls[j] if j < len(urls) else ""
        if _is_http_url(candidate):
            thought_j = thoughts[j] if j < len(thoughts) else {}
            action_j = actions[j] if j < len(actions) else {}
            return _state_from_snapshot(j, candidate, thought_j, action_j)

    # Fallback to original behavior.
    return _state_from_snapshot(i + 1, raw_to, next_thought, next_action)


def _semantic_consistency(action: ActionType, intent: Any, selector: str = "") -> bool:
    """Action-intent consistency check for quality metric."""
    if intent is None:
        return False
    key = str(getattr(intent, "key", "") or "").lower()
    summary = str(getattr(intent, "summary", "") or "").lower()
    verb = str(getattr(intent, "verb", "") or "").lower()
    obj = str(getattr(intent, "object", "") or "").lower()
    sel = (selector or "").lower()
    text = f"{key} {summary} {verb} {obj}"

    if action == ActionType.FILL:
        if any(token in sel for token in ("input", "textarea", "select", "password", "username", "email", "search")):
            return True
        return any(k in text for k in ("fill", "input", "enter", "type", "select", "choose", "set", "credentials"))

    if action == ActionType.CLICK:
        if any(token in key for token in (".click", "click.", ".navigate", "navigate.", ".submit", "submit.", ".logout", "logout.")):
            return True
        return any(k in text for k in ("click", "submit", "press", "tap", "toggle", "check", "open", "navigate", "visit", "go", "logout", "login"))

    if action == ActionType.NAVIGATE:
        if any(token in key for token in ("navigation.", ".navigate", "navigate.")):
            return True
        return any(k in text for k in ("navigate", "open", "visit", "go", "redirect", "return", "route", "page"))

    return False


async def _build_graph_from_history(
    history: Any,
    inventory: list[dict] | None = None,
    actions: list[dict[str, Any]] | None = None,
    thoughts: list[dict | object] | None = None,
    urls: list[str] | None = None,
    runtime_non_ui_action_count: int = 0,
) -> MultiDiGraph:
    """Build DiGraph from agent history.
    
    - Node IDs are cleaned URLs (no query params).
    - Edges are added for all valid click/fill actions.
    - Inventory is optional (deprecated constraint).
    """
    G: MultiDiGraph = MultiDiGraph()
    actions = actions if actions is not None else (list(history.model_actions()) if history else [])
    actions = [_action_to_dict(a) for a in actions]
    thoughts = thoughts if thoughts is not None else (list(history.model_thoughts()) if history else [])
    if urls is None:
        try:
            urls = list(history.urls()) if history else []
        except Exception:
            urls = []
    urls = _stabilize_url_snapshots(urls)

    # Ensure start node exists
    first_url = urls[0] if urls else ""
    first_thought = thoughts[0] if thoughts else {}
    first_action = actions[0] if actions else {}
    start_node, start_url = _state_from_snapshot(0, first_url, first_thought, first_action)
    
    # Try to find title for start node
    start_label = start_url or start_node
    # We can't easily get the title for the very first state from history actions/thoughts
    # unless we look at the first thought's context, but let's keep it simple.
    G.add_node(start_node, label=start_label, url=start_url)

    edges_added = 0
    filtered_non_ui_edges = 0
    for i, action in enumerate(actions):
        # Ensure action is a dict
        action = _action_to_dict(action)

        thought = thoughts[i] if i < len(thoughts) else {}
        
        # Determine From/To Nodes
        # from_node is the URL before action i
        raw_from = urls[i] if i < len(urls) else ""
        from_node, from_url = _state_from_snapshot(i, raw_from, thought, action)
        
        # to_node: best-effort concrete URL resolution to reduce pseudo-state noise.
        to_node, to_url = _resolve_target_state(i, urls, thoughts, actions)

        action_key = _extract_action_key(action)
        neighbor_steps = _build_neighbor_steps(i, actions, thoughts, urls, window=1)
        page_signals = _build_page_signals(from_url, to_url, action_key)
        edge_model = await parse_browser_use_step(
            action,
            thought,
            from_url,
            to_url,
            neighbor_steps=neighbor_steps,
            page_signals=page_signals,
        )
        
        if from_node not in G:
            G.add_node(from_node, label=from_url or from_node, url=from_url)
        if to_node not in G:
            G.add_node(to_node, label=to_url or to_node, url=to_url)

        if action_key in FILTERED_ACTION_KEYS:
            filtered_non_ui_edges += 1
            continue

        # Skip adding edge for navigate; only add for click/fill with selector
        if edge_model.action == ActionType.UNKNOWN:
            filtered_non_ui_edges += 1
            continue
        if edge_model.action == ActionType.NAVIGATE:
            continue
        if edge_model.action in (ActionType.CLICK, ActionType.FILL) and edge_model.selector:
            # Add edge regardless of inventory (dynamic discovery)
            edge_id = f"step-{i}"
            G.add_edge(
                from_node,
                to_node,
                key=edge_id,
                edge_id=edge_id,
                step_index=i,
                selector=edge_model.selector,
                action=edge_model.action,
                tab_id=getattr(edge_model, "tab_id", "tab-0"),
                target_tab_id=getattr(edge_model, "target_tab_id", None),
                tab_action=getattr(edge_model, "tab_action", None),
                tab=getattr(edge_model, "tab", None),
                frame_path=getattr(edge_model, "frame_path", []),
                intent=edge_model.intent,
                context_level_used=getattr(edge_model, "context_level_used", None),
                intent_failure_reason=edge_model.intent_failure_reason,
                param_name=edge_model.param_name,
                action_value=edge_model.action_value,
                element=edge_model.element,
                constraints=edge_model.constraints,
            )
            edges_added += 1

    G.graph["filtered_non_ui_edges"] = filtered_non_ui_edges
    G.graph["runtime_non_ui_action_count"] = runtime_non_ui_action_count
    if edges_added == 0:
        G.graph["filtered_all_edges"] = True
    return G


def _load_inventory(inventory_path: str | Path) -> list[dict]:
    """Load elements list from inventory JSON; raise if file missing or invalid."""
    path = Path(inventory_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {path}. Run scout first: uv run python -m graph_agent.mapping.run --url <url> (with --inventory and --output)."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    elements = raw.get("elements") if isinstance(raw, dict) else []
    return elements if isinstance(elements, list) else []


def _resolve_snapshot_path(graph_output_path: str | Path) -> Path:
    """Resolve acceptance snapshot path next to graph output."""
    graph_path = Path(graph_output_path)
    return graph_path.parent / "acceptance_snapshot.json"


def _extract_inventory_snapshot(inventory_path: str | Path | None) -> dict[str, Any]:
    """Read inventory metadata for acceptance snapshot."""
    if inventory_path is None:
        return {"exists": False}
    path = Path(inventory_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "path": str(path), "parse_error": True}

    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    return {
        "exists": True,
        "path": str(path),
        "mode": payload.get("mode", "single_page") if isinstance(payload, dict) else "single_page",
        "page_count": int(metadata.get("page_count", 1)) if isinstance(metadata, dict) else 1,
        "aggregated_element_count": int(metadata.get("aggregated_element_count", payload.get("elements", []) and len(payload.get("elements", [])) or 0)) if isinstance(payload, dict) else 0,
        "type_counts": metadata.get("type_counts", {}) if isinstance(metadata, dict) else {},
    }


def _build_snapshot_metric_map(graph: nx.Graph) -> dict[str, float]:
    """Build normalized metric map for snapshot comparison."""
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "intent_missing_count": float(graph.graph.get("intent_missing_count", 0)),
        "intent_success_rate": float(graph.graph.get("intent_success_rate", 0.0)),
        "business_template_count": float(graph.graph.get("business_template_count", 0)),
        "semantic_consistency_rate": float(graph.graph.get("semantic_consistency_rate", 0.0)),
        "business_intent_edge_ratio": float(graph.graph.get("business_intent_edge_ratio", 0.0)),
        "runtime_non_ui_action_count": float(graph.graph.get("runtime_non_ui_action_count", 0)),
        "state_like_node_ratio": float(graph.graph.get("state_like_node_ratio", 0.0)),
        "re_infer_success_rate": float(graph.graph.get("re_infer_success_rate", 0.0)),
    }


def _compute_snapshot_delta(previous: dict[str, Any] | None, current_metrics: dict[str, float]) -> dict[str, float]:
    """Compute metric deltas (current - previous)."""
    if not previous:
        return {}
    prev_metrics_raw = previous.get("metrics", {})
    if not isinstance(prev_metrics_raw, dict):
        return {}
    delta: dict[str, float] = {}
    for key, value in current_metrics.items():
        try:
            prev_value = float(prev_metrics_raw.get(key, value))
        except (TypeError, ValueError):
            prev_value = value
        delta[key] = round(value - prev_value, 6)
    return delta


def _write_acceptance_snapshot(
    graph: nx.Graph,
    graph_output_path: str | Path,
    inventory_path: str | Path | None,
    target_url: str | None,
) -> Path:
    """Persist acceptance snapshot and include delta vs previous run."""
    snapshot_path = _resolve_snapshot_path(graph_output_path)
    previous: dict[str, Any] | None = None
    if snapshot_path.exists():
        try:
            loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else None
        except Exception:
            previous = None

    metrics = _build_snapshot_metric_map(graph)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_url": target_url or "",
        "graph_path": str(Path(graph_output_path)),
        "inventory": _extract_inventory_snapshot(inventory_path),
        "metrics": metrics,
        "comparison": {
            "has_previous": previous is not None,
            "delta": _compute_snapshot_delta(previous, metrics),
        },
    }
    snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot_path


async def _refresh_business_templates(graph: nx.Graph) -> None:
    """Regenerate business templates and store them in graph metadata."""
    try:
        templates = await generate_business_templates(graph)
    except Exception:
        graph.graph["business_templates"] = []
        graph.graph["business_template_count"] = 0
        graph.graph["business_template_generation_failures"] = 1
        graph.graph["business_templates_generated_at"] = datetime.now(timezone.utc).isoformat()
        graph.graph["business_template_stats"] = {"count": 0}
        return
    store_business_templates(graph, templates)


async def re_infer_missing_intents(graph_path: str | Path, inventory_path: str | Path | None = None) -> dict[str, int]:
    """Re-infer missing intents for edges in an existing graph file."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    graph = load_graph(path)
    total = 0
    succeeded = 0
    failed = 0
    edge_iter: list[tuple[str, str, Any, dict[str, Any]]]
    if isinstance(graph, nx.MultiDiGraph):
        edge_iter = [(u, v, k, data) for u, v, k, data in graph.edges(keys=True, data=True)]
    else:
        edge_iter = [(u, v, None, data) for u, v, data in graph.edges(data=True)]

    for u, v, k, data in edge_iter:
        if data.get("intent") is not None:
            continue
        total += 1
        action_raw = data.get("action", ActionType.UNKNOWN)
        if isinstance(action_raw, ActionType):
            action = action_raw
        else:
            try:
                action = ActionType(str(action_raw))
            except ValueError:
                action = ActionType.UNKNOWN
        selector = str(data.get("selector", ""))
        param_name = data.get("param_name")
        thought_text = ""
        source_meta = graph.nodes[u] if u in graph else {}
        target_meta = graph.nodes[v] if v in graph else {}
        source_url = str(source_meta.get("url") or u)
        target_url = str(target_meta.get("url") or v)
        intent, reason = await infer_intent_for_context(
            action=action,
            selector=selector,
            source_url=source_url,
            target_url=target_url,
            data_key=str(param_name) if param_name is not None else None,
            thought_text=thought_text,
        )
        if intent is not None:
            if k is None:
                graph.edges[u, v]["intent"] = intent
                graph.edges[u, v]["intent_failure_reason"] = None
            else:
                graph.edges[u, v, k]["intent"] = intent
                graph.edges[u, v, k]["intent_failure_reason"] = None
            succeeded += 1
        else:
            if k is None:
                graph.edges[u, v]["intent"] = None
                graph.edges[u, v]["intent_failure_reason"] = reason
            else:
                graph.edges[u, v, k]["intent"] = None
                graph.edges[u, v, k]["intent_failure_reason"] = reason
            failed += 1
    edge_count = graph.number_of_edges()
    missing_after = sum(1 for _u, _v, d in graph.edges(data=True) if d.get("intent") is None)
    graph.graph["intent_missing_count"] = missing_after
    graph.graph["intent_success_rate"] = (1.0 - (missing_after / edge_count)) if edge_count else 1.0
    graph.graph["re_infer_success_rate"] = (succeeded / total) if total else 1.0
    await _refresh_business_templates(graph)
    save_graph(graph, path)
    _write_acceptance_snapshot(
        graph=graph,
        graph_output_path=path,
        inventory_path=inventory_path,
        target_url="",
    )
    return {"total": total, "succeeded": succeeded, "failed": failed}


async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,
    task: str | None = None,
    max_steps: int = 30,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,
) -> MultiDiGraph:
    """Run browser-use Agent to explore UI flow and build DiGraph.

    - url: Start URL (required via arg or MAPPING_URL env).
    - output_path: Where to save graph JSON (default: graph_agent/data/graph.json).
    - task: Task description for the agent (default: rendered from DEFAULT_TASK_TEMPLATE).
    - max_steps: Maximum agent steps.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the built DiGraph and saves it to output_path.
    """
    if inventory_path is None or not str(inventory_path).strip():
        raise ValueError(
            "inventory_path is required. Run scout first, then pass --inventory to mapping."
        )
    inventory = _load_inventory(inventory_path)

    from browser_use import Agent, Browser

    resolved_url = _resolve_mapping_url(url)
    resolved_task = _build_mapping_task_with_env_hints(task, resolved_url)
    pkg_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        # Resolve default relative to package: graph_agent/data/graph.json
        output_path = str(pkg_root / "data" / "graph.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    browser = Browser(headless=True)
    llm = get_llm()
    initial_actions = [{"navigate": {"url": resolved_url, "new_tab": False}}]
    full_task = resolved_task

    # Use graph_agent/data as the base for file operations
    # Note: browser-use will append 'browseruse_agent_data' to this path
    data_dir = pkg_root / "data"

    agent: Agent = Agent(
        task=full_task,
        llm=llm,
        browser=browser,
        initial_actions=initial_actions,
        file_system_path=str(data_dir),
    )

    try:
        history = await agent.run(max_steps=max_steps)
    finally:
        if hasattr(browser, "stop"):
            await browser.stop()
        elif hasattr(browser, "close"):
            await browser.close()

    all_actions, all_thoughts, all_urls = _collect_history_snapshots(history)
    actions, thoughts, urls, runtime_filtered = _runtime_filter_snapshots(all_actions, all_thoughts, all_urls)

    # Console: print each step (step number, URL, selector, action, semantic_label)
    for i, action in enumerate(actions):
        thought: dict | object = thoughts[i] if i < len(thoughts) else {}
        step_url = urls[i] if i < len(urls) else ""
        if step_url is None:
            step_url = ""
        next_url = urls[i+1] if i+1 < len(urls) else step_url
        if next_url is None:
            next_url = ""
        
        edge_model = await parse_browser_use_step(action, thought, step_url, next_url)
        intent_text = edge_model.intent.summary if edge_model.intent else f"<missing-intent:{edge_model.intent_failure_reason}>"
        
        print(
            f"  [{i + 1}] url={step_url!r} selector={edge_model.selector!r} "
            f"action={edge_model.action!r} intent={intent_text!r}"
        )

    G: MultiDiGraph = await _build_graph_from_history(
        history,
        inventory=inventory,
        actions=actions,
        thoughts=thoughts,
        urls=urls,
        runtime_non_ui_action_count=runtime_filtered,
    )
    G.graph["start_url"] = resolved_url

    # Task 2: Record visited URLs for derived URL scout (unique http(s) from history).
    visited_urls = list(dict.fromkeys(u for u in urls if _is_http_url(u or "")))
    G.graph["visited_urls"] = visited_urls

    # Optionally merge with existing graph on disk.
    path_obj = Path(output_path)
    if merge_existing and path_obj.exists():
        try:
            existing_G = load_graph(output_path)
            # Merge new G into existing G
            G = nx.compose(existing_G, G)
            print(f"Merged with existing graph from {output_path}")
        except Exception as e:
            print(f"Warning: Could not load existing graph to merge: {e}")

    # Parse stop reason from final_result and write to graph metadata
    final_text = ""
    try:
        if history and hasattr(history, "final_result"):
            final_text = (history.final_result() or "") or ""
        else:
            final_text = str(history) if history else ""
    except Exception:
        final_text = ""
    for marker in ("Stopped:", "停止："):
        if marker in final_text:
            idx = final_text.find(marker)
            reason = final_text[idx + len(marker) :].strip()
            if len(reason) > 200:
                reason = reason[:200] + "..."
            G.graph["mapping_stopped"] = True
            G.graph["stop_reason"] = reason
            break
    else:
        G.graph["mapping_stopped"] = False
        G.graph["stop_reason"] = None

    # T4: intent_missing_count for quality assessment
    intent_missing_count = sum(1 for _u, _v, d in G.edges(data=True) if d.get("intent") is None)
    G.graph["intent_missing_count"] = intent_missing_count
    total_edges = G.number_of_edges()
    intent_success_rate = (1.0 - (intent_missing_count / total_edges)) if total_edges else 1.0
    G.graph["intent_success_rate"] = round(intent_success_rate, 4)
    consistent = 0
    non_null_intent_edges = 0
    for _u, _v, data in G.edges(data=True):
        intent = data.get("intent")
        if intent is None:
            continue
        non_null_intent_edges += 1
        if _semantic_consistency(data.get("action", ActionType.UNKNOWN), intent, selector=str(data.get("selector", ""))):
            consistent += 1
    consistency_rate = (consistent / non_null_intent_edges) if non_null_intent_edges else 1.0
    G.graph["semantic_consistency_rate"] = round(consistency_rate, 4)
    G.graph["inventory_non_empty_rate"] = 1.0 if inventory else 0.0
    state_like_nodes = sum(1 for n, data in G.nodes(data=True) if str(n) != str(data.get("url", "")))
    G.graph["state_like_node_ratio"] = (state_like_nodes / G.number_of_nodes()) if G.number_of_nodes() else 0.0
    business_edges = 0
    for _u, _v, d in G.edges(data=True):
        key = getattr(d.get("intent"), "key", None)
        if isinstance(key, str) and key and not key.startswith("navigation."):
            business_edges += 1
    G.graph["business_intent_edge_ratio"] = (business_edges / G.number_of_edges()) if G.number_of_edges() else 0.0
    uv_counts: dict[tuple[str, str], int] = {}
    for u, v in G.edges():
        uv_counts[(str(u), str(v))] = uv_counts.get((str(u), str(v)), 0) + 1
    G.graph["multi_edge_preserved_count"] = sum(max(0, c - 1) for c in uv_counts.values())

    await _refresh_business_templates(G)
    save_graph(G, output_path)
    snapshot_path = _write_acceptance_snapshot(
        graph=G,
        graph_output_path=output_path,
        inventory_path=inventory_path,
        target_url=resolved_url,
    )
    edge_count = G.number_of_edges()
    print(
        f"Graph saved: {output_path} "
        f"(nodes={G.number_of_nodes()}, edges={edge_count}, intent_missing={intent_missing_count})"
    )
    print(f"Acceptance snapshot saved: {snapshot_path}")
    return G


def main() -> None:
    """CLI entry: run scout then mapping. Scout writes inventory, mapping uses it to build graph."""
    import argparse
    from dotenv import load_dotenv

    # Try loading from current directory first, then fallback to project root
    if not load_dotenv():
        # Fallback: try to find .env in project root
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded .env from {env_path}")
        else:
            print(f"Warning: .env not found at {env_path} or current directory")

    pkg_root = Path(__file__).resolve().parent.parent
    default_output = str(pkg_root / "data" / "graph.json")
    default_inventory = str(pkg_root / "data" / "element_inventory.json")

    parser = argparse.ArgumentParser(
        description="Run scout (list page elements) then mapping (explore flow, build graph)."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MAPPING_URL", ""),
        help="Start URL for scout and mapping. If omitted, uses MAPPING_URL.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MAPPING_OUTPUT", default_output),
        help="Output graph JSON path (default: graph_agent/data/graph.json)",
    )
    parser.add_argument(
        "--inventory",
        default=os.getenv("MAPPING_INVENTORY", default_inventory),
        help="Scout inventory JSON path (default: graph_agent/data/element_inventory.json)",
    )
    parser.add_argument(
        "--scout-pages",
        default=os.getenv("SCOUT_PAGES", ""),
        help="Comma-separated extra pages for multi-page scout aggregation. Supports relative paths or absolute URLs.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly mapped graph with existing output file.",
    )
    parser.add_argument(
        "--re-infer-missing",
        action="store_true",
        help="Re-infer missing intents (intent=null) in existing graph and overwrite.",
    )
    args = parser.parse_args()
    output = (args.output or "").strip() or default_output
    inventory = (args.inventory or "").strip() or default_inventory
    scout_pages_arg = (args.scout_pages or "").strip()
    scout_pages = [item.strip() for item in scout_pages_arg.split(",") if item.strip()]
    merge_existing = bool(args.merge_existing)
    re_infer_missing = bool(args.re_infer_missing)
    url = _resolve_mapping_url(args.url) if not re_infer_missing else ""

    async def _run() -> None:
        if re_infer_missing:
            stats = await re_infer_missing_intents(output, inventory_path=inventory)
            print(
                "Re-infer completed:",
                f"total={stats['total']}, succeeded={stats['succeeded']}, failed={stats['failed']}",
            )
            return
        print("Step 1: Scout (list interactive elements)...")
        if scout_pages:
            await run_scout_multi(start_url=url, page_hints=scout_pages, output_path=inventory)
        else:
            await run_scout(url, output_path=inventory)
        print(f"Inventory saved: {inventory}")
        print("Step 2: Mapping (explore flow, build graph)...")
        G = await run_mapping(
            url=url,
            output_path=output,
            inventory_path=inventory,
            merge_existing=merge_existing,
        )

        # Task 2: Scout derived URLs discovered during mapping to enrich inventory.
        visited_urls = G.graph.get("visited_urls", [])
        derived = extract_derived_urls(visited_urls, url, exclude_start=True)
        if derived:
            print(f"Step 3: Scout derived URLs ({len(derived)} pages)...")
            await run_scout_multi(start_url=url, page_hints=derived, output_path=inventory)
            print(f"Inventory enriched with derived pages: {inventory}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()

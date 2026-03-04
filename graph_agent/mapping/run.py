"""Run browser-use Agent for mapping: explore flow, build DiGraph from history."""

from __future__ import annotations

import json
import os
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import networkx as nx
from networkx import DiGraph

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.llm import get_llm
from graph_agent.mapping.parser import parse_browser_use_step, infer_intent_for_context
from graph_agent.mapping.scout import run_scout
from graph_agent.models import ActionType

# Generic task template for site-agnostic mapping.
DEFAULT_TASK_TEMPLATE = (
    "从起始URL开始探索核心业务流程：{start_url}。"
    "记录每一步的 selector、业务意图、动作类型及目标状态。"
    "遇到无法完成的表单（缺少必填数据）或潜在破坏性操作（删除、清空、提交不可逆变更）时立即停止，"
    "并在最终回复中写明原因（例如：Stopped: unfillable form / Stopped: would delete data）。"
)

FILTERED_ACTION_KEYS = {"read_file", "write_file", "done"}


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
    )
    for key in preferred:
        if key in data:
            return key
    return next(iter(data.keys()), "unknown")


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


def _state_from_snapshot(step: int, raw_url: str, thought: dict | object, action: dict | object) -> str:
    """Build state node ID from URL; fallback to explainable pseudo-state when URL is missing."""
    cleaned = _clean_url(raw_url or "")
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned

    goal = _extract_next_goal(thought)
    action_key = _extract_action_key(action).replace("_", " ")
    goal_brief = re.sub(r"\s+", " ", goal).strip()[:40]
    descriptor = f"{action_key} {goal_brief}".strip() or "unknown"
    fingerprint_src = f"{step}|{action_key}|{goal}|{cleaned}"
    fingerprint = hashlib.md5(fingerprint_src.encode("utf-8")).hexdigest()[:6]
    return f"State {step} [{descriptor}]#{fingerprint}"


async def _build_graph_from_history(history, inventory: list[dict] | None = None) -> DiGraph:
    """Build DiGraph from agent history.
    
    - Node IDs are cleaned URLs (no query params).
    - Edges are added for all valid click/fill actions.
    - Inventory is optional (deprecated constraint).
    """
    G: DiGraph = DiGraph()
    actions = list(history.model_actions()) if history else []
    thoughts = list(history.model_thoughts()) if history else []
    try:
        urls = list(history.urls()) if history else []
    except Exception:
        urls = []

    # Ensure start node exists
    first_url = urls[0] if urls else ""
    first_thought = thoughts[0] if thoughts else {}
    first_action = actions[0] if actions else {}
    start_node = _state_from_snapshot(0, first_url, first_thought, first_action)
    
    # Try to find title for start node
    start_label = start_node
    # We can't easily get the title for the very first state from history actions/thoughts
    # unless we look at the first thought's context, but let's keep it simple.
    G.add_node(start_node, label=start_label, url=start_node)

    edges_added = 0
    filtered_non_ui_edges = 0
    for i, action in enumerate(actions):
        # Ensure action is a dict
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        elif hasattr(action, "dict"):
            action = action.dict()

        thought = thoughts[i] if i < len(thoughts) else {}
        
        # Determine From/To Nodes
        # from_node is the URL before action i
        raw_from = urls[i] if i < len(urls) else ""
        from_node = _state_from_snapshot(i, raw_from, thought, action)
        
        # to_node is the URL after action i (which is usually captured at i+1)
        next_thought = thoughts[i + 1] if i + 1 < len(thoughts) else {}
        next_action = actions[i + 1] if i + 1 < len(actions) else {}
        raw_to = urls[i + 1] if i + 1 < len(urls) else ""
        to_node = _state_from_snapshot(i + 1, raw_to, next_thought, next_action)

        edge_model = await parse_browser_use_step(action, thought, from_node, to_node)
        
        if from_node not in G:
            G.add_node(from_node, label=from_node, url=from_node)
        if to_node not in G:
            G.add_node(to_node, label=to_node, url=to_node)

        action_key = _extract_action_key(action)
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
            G.add_edge(
                from_node,
                to_node,
                selector=edge_model.selector,
                action=edge_model.action,
                intent=edge_model.intent,
                intent_failure_reason=edge_model.intent_failure_reason,
                data_key=edge_model.data_key,
                constraints=edge_model.constraints,
            )
            edges_added += 1

    G.graph["filtered_non_ui_edges"] = filtered_non_ui_edges
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


async def re_infer_missing_intents(graph_path: str | Path) -> dict[str, int]:
    """Re-infer missing intents for edges in an existing graph file."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    graph = load_graph(path)
    total = 0
    succeeded = 0
    failed = 0
    for u, v, data in graph.edges(data=True):
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
        data_key = data.get("data_key")
        thought_text = ""
        intent, reason = await infer_intent_for_context(
            action=action,
            selector=selector,
            source_url=str(u),
            target_url=str(v),
            data_key=str(data_key) if data_key is not None else None,
            thought_text=thought_text,
        )
        if intent is not None:
            graph.edges[u, v]["intent"] = intent
            graph.edges[u, v]["intent_failure_reason"] = None
            succeeded += 1
        else:
            graph.edges[u, v]["intent"] = None
            graph.edges[u, v]["intent_failure_reason"] = reason
            failed += 1
    save_graph(graph, path)
    return {"total": total, "succeeded": succeeded, "failed": failed}


async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,
    task: str | None = None,
    max_steps: int = 30,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,
) -> DiGraph:
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
    resolved_task = _build_mapping_task(task, resolved_url)
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

    actions = list(history.model_actions()) if history else []
    thoughts = list(history.model_thoughts()) if history else []
    try:
        urls = list(history.urls()) if history else []
    except Exception:
        urls = []

    # Console: print each step (step number, URL, selector, action, semantic_label)
    for i, action in enumerate(actions):
        # Ensure action is a dict
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        elif hasattr(action, "dict"):
            action = action.dict()
        
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

    G: DiGraph = await _build_graph_from_history(history, inventory=inventory)

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

    save_graph(G, output_path)
    print(f"Graph saved: {output_path} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")
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
    merge_existing = bool(args.merge_existing)
    re_infer_missing = bool(args.re_infer_missing)
    url = _resolve_mapping_url(args.url) if not re_infer_missing else ""

    async def _run() -> None:
        if re_infer_missing:
            stats = await re_infer_missing_intents(output)
            print(
                "Re-infer completed:",
                f"total={stats['total']}, succeeded={stats['succeeded']}, failed={stats['failed']}",
            )
            return
        print("Step 1: Scout (list interactive elements)...")
        await run_scout(url, output_path=inventory)
        print(f"Inventory saved: {inventory}")
        print("Step 2: Mapping (explore flow, build graph)...")
        await run_mapping(
            url=url,
            output_path=output,
            inventory_path=inventory,
            merge_existing=merge_existing,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()

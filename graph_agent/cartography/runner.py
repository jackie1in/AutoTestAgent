"""Run browser-use Agent for mapping: explore flow and save to Neo4j."""

from __future__ import annotations

import json
import os
import asyncio
import hashlib
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

from graph_agent.graph.templates import (
    generate_business_templates,
)
from graph_agent.graph.manager import Neo4jGraphManager as GraphManager
from graph_agent.llm import get_llm
from graph_agent.intent.parser import (
    parse_browser_use_step,
    parse_browser_use_step_lite,
)
from graph_agent.cartography.scout import extract_derived_urls, run_scout, run_scout_multi
from graph_agent.models import ActionType

# Global registry for active browser sessions (for cleanup on Ctrl+C)
_active_browsers: list[Any] = []
_shutdown_requested = False


def _register_browser(browser: Any) -> None:
    """Register a browser instance for cleanup on shutdown."""
    if browser not in _active_browsers:
        _active_browsers.append(browser)


def _unregister_browser(browser: Any) -> None:
    """Unregister a browser instance after cleanup."""
    if browser in _active_browsers:
        _active_browsers.remove(browser)


async def _cleanup_all_browsers() -> None:
    """Clean up all registered browser sessions using kill() API."""
    global _active_browsers
    if not _active_browsers:
        return
    
    print(f"\n[INFO] Cleaning up {len(_active_browsers)} browser session(s)...")
    for browser in list(_active_browsers):
        try:
            # Use kill() API for forceful cleanup (browser-use recommended)
            if hasattr(browser, "kill"):
                await browser.kill()
                print("  [OK] Browser killed")
            elif hasattr(browser, "stop"):
                await browser.stop()
                print("  [OK] Browser stopped")
            elif hasattr(browser, "close"):
                await browser.close()
                print("  [OK] Browser closed")
        except Exception as e:
            print(f"  [WARN] Error during browser cleanup: {e}")
        finally:
            _unregister_browser(browser)
    print("[INFO] Browser cleanup complete")


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C (SIGINT) and SIGTERM signals."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[FORCE] Force exit requested")
        sys.exit(1)
    
    _shutdown_requested = True
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[INFO] Received {signal_name}, shutting down gracefully...")
    print("[INFO] Press Ctrl+C again to force exit")
    
    # Note: We can't do async cleanup here, so we set a flag
    # The main loop should check _shutdown_requested


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@asynccontextmanager
async def managed_browser(browser: Any):
    """Context manager for browser lifecycle with cleanup on exit."""
    _register_browser(browser)
    try:
        yield browser
    finally:
        try:
            # Use kill() API for forceful cleanup (browser-use recommended)
            if hasattr(browser, "kill"):
                await browser.kill()
            elif hasattr(browser, "stop"):
                await browser.stop()
            elif hasattr(browser, "close"):
                await browser.close()
        except Exception as e:
            print(f"[WARN] Browser cleanup error: {e}")
        finally:
            _unregister_browser(browser)

# Generic task template for site-agnostic mapping.
DEFAULT_TASK_TEMPLATE = (
    "【目标URL】{start_url} - 必须首先导航到这个地址！ "
    "【第一步】使用 navigate 动作访问 {start_url} "
    "【第二步】从该页面开始探索核心业务流程: "
    "探索阶段优先使用UI交互动作：click/fill/navigate/select。"
    "遇到表单时尽量填写所有字段，包括富文本编辑器（contenteditable/TinyMCE/CKEditor/Quill等）——"
    "使用 input_text 动作向富文本区域输入示例文本即可。"
    "在点击菜单、列表项、详情入口、子项目入口、概览入口后，继续探索进入的派生页面，不要停留在入口页。"
    "记录每一步的 selector、业务意图、动作类型及目标状态。"
    "严禁在探索过程中使用 read_file/write_file/replace_file 等文件工具；仅允许在最终 done 时输出结论。"
    "遇到无法完成的表单（缺少必填数据）或潜在破坏性操作（删除、清空、提交不可逆变更）时立即停止，"
    "并在最终回复中写明原因（例如：Stopped: unfillable form / Stopped: would delete data）。"
)

FILTERED_ACTION_KEYS = {"read_file", "write_file", "done", "unknown"}
# Filter wait actions before state construction so they do not create
# disconnected pseudo-states between two real UI interactions.
DISALLOWED_RUNTIME_ACTION_KEYS = {
    "read_file",
    "write_file",
    "replace_file",
    "done",
    "wait",
}


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


def _collect_history_snapshots(
    history: Any,
) -> tuple[list[dict[str, Any]], list[dict | object], list[str]]:
    """Collect action/thought/url snapshots from history as plain lists.

    ``model_actions()`` *flattens* multi-action steps (one history item can
    contain several actions), while ``model_thoughts()`` and ``urls()``
    return one entry per history item.  We must expand thoughts/urls to
    match the flattened action list so that downstream code can index them
    with the same ``i``.
    """
    if not history:
        return [], [], []

    # When the history object exposes the internal step list (AgentHistoryList),
    # iterate step-by-step so that thoughts/urls are duplicated for
    # multi-action steps, keeping indices aligned with the flat action list.
    history_items = getattr(history, "history", None)
    if history_items is not None:
        actions: list[dict[str, Any]] = []
        thoughts: list[dict | object] = []
        urls: list[str] = []

        try:
            raw_urls = list(history.urls())
        except Exception:
            raw_urls = []

        for step_idx, h in enumerate(history_items):
            model_output = getattr(h, "model_output", None)
            if not model_output:
                continue
            thought = model_output.current_state
            url = raw_urls[step_idx] if step_idx < len(raw_urls) else ""

            state = getattr(h, "state", None)
            ie_list = (
                getattr(state, "interacted_element", None)
                if state
                else None
            ) or [None] * len(model_output.action)
            for action_obj, ie in zip(model_output.action, ie_list):
                if hasattr(action_obj, "model_dump"):
                    output = action_obj.model_dump(
                        exclude_none=True, mode="json"
                    )
                else:
                    output = _action_to_dict(action_obj)
                output["interacted_element"] = ie
                actions.append(output)
                thoughts.append(thought)
                urls.append(url if url is not None else "")

        return actions, thoughts, urls

    # Fallback for legacy / mock history objects that only expose the
    # high-level methods.  This path has the known multi-action alignment
    # issue but keeps backward compat with test mocks.
    raw_actions = list(history.model_actions()) if history else []
    actions_fb = [_action_to_dict(a) for a in raw_actions]
    thoughts_fb = list(history.model_thoughts()) if history else []
    try:
        urls_fb = list(history.urls()) if history else []
    except Exception:
        urls_fb = []
    return actions_fb, thoughts_fb, urls_fb


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


def _resolve_mapping_headless() -> bool:
    """Resolve mapping headless mode from MAPPING_HEADLESS env."""
    raw = (os.getenv("MAPPING_HEADLESS") or "").strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_mapping_channel() -> str | None:
    """Resolve optional browser channel from MAPPING_CHANNEL env."""
    raw = (os.getenv("MAPPING_CHANNEL") or "").strip()
    return raw or None


def _setup_browser_use_timeouts():
    """Setup browser-use timeout environment variables from MAPPING_TIMEOUT.
    
    browser-use's _navigate_and_wait has hardcoded 8s timeout, but we can
    increase the overall event timeout to give more time for slow pages.
    """
    mapping_timeout = os.getenv("MAPPING_TIMEOUT", "").strip()
    if mapping_timeout:
        try:
            timeout_val = float(mapping_timeout)
            # Set browser-use timeout environment variables
            os.environ.setdefault("TIMEOUT_NavigateToUrlEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStartEvent", str(timeout_val))
        except ValueError:
            pass


def _resolve_intent_context_window(default: int = 1) -> int:
    """Resolve intent context window size from env.

    Uses MAPPING_INTENT_CONTEXT_WINDOW, clamps to [0, 5] to avoid oversized prompts.
    """
    raw = (os.getenv("MAPPING_INTENT_CONTEXT_WINDOW") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, min(5, value))


def _resolve_intent_mode() -> str:
    """Resolve intent inference mode: 'sync' (blocking) or 'async' (deferred)."""
    raw = (os.getenv("MAPPING_INTENT_MODE") or "").strip().lower()
    if raw in ("async", "deferred"):
        return "async"
    return "sync"


def _resolve_intent_concurrency(default: int = 3) -> int:
    """Resolve parallel intent inference concurrency from env."""
    raw = (os.getenv("MAPPING_INTENT_CONCURRENCY") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, min(10, value))



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


def _state_from_snapshot(
    step: int, raw_url: str, thought: dict | object, action: dict | object
) -> tuple[str, str]:
    """Build opaque state id and normalized page URL from a snapshot."""
    cleaned = _clean_url(raw_url or "")

    goal = _extract_next_goal(thought)
    action_key = _extract_action_key(action).replace("_", " ")
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


def _build_page_signals(
    source_url: str, target_url: str, action_key: str
) -> dict[str, str]:
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


def _action_key_from_edge_data(data: dict[str, Any]) -> str:
    """Extract action key string from edge data."""
    action_raw = data.get("action", ActionType.UNKNOWN)
    if isinstance(action_raw, ActionType):
        return action_raw.value
    text = str(action_raw or "").strip().lower()
    return text or ActionType.UNKNOWN.value




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

    if action == ActionType.SELECT:
        return any(
            k in text
            for k in ("select", "choose", "pick", "dropdown", "option")
        )

    if action == ActionType.RICH_TEXT:
        return any(
            k in text
            for k in ("type", "fill", "input", "enter", "edit", "write", "rich", "content")
        )

    if action == ActionType.FILL:
        if any(
            token in sel
            for token in (
                "input",
                "textarea",
                "select",
                "password",
                "username",
                "email",
                "search",
            )
        ):
            return True
        return any(
            k in text
            for k in (
                "fill",
                "input",
                "enter",
                "type",
                "select",
                "choose",
                "set",
                "credentials",
            )
        )

    if action == ActionType.CLICK:
        if any(
            token in key
            for token in (
                ".click",
                "click.",
                ".navigate",
                "navigate.",
                ".submit",
                "submit.",
                ".logout",
                "logout.",
            )
        ):
            return True
        return any(
            k in text
            for k in (
                "click",
                "submit",
                "press",
                "tap",
                "toggle",
                "check",
                "open",
                "navigate",
                "visit",
                "go",
                "logout",
                "login",
            )
        )

    if action == ActionType.NAVIGATE:
        if any(token in key for token in ("navigation.", ".navigate", "navigate.")):
            return True
        return any(
            k in text
            for k in (
                "navigate",
                "open",
                "visit",
                "go",
                "redirect",
                "return",
                "route",
                "page",
            )
        )

    return False


async def _build_graph_in_neo4j(
    manager: GraphManager,
    app_id: str,
    session_id: str,
    history: Any,
    inventory: list[dict] | None = None,
    actions: list[dict[str, Any]] | None = None,
    thoughts: list[dict | object] | None = None,
    urls: list[str] | None = None,
    runtime_non_ui_action_count: int = 0,
    step_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    intent_mode: str | None = None,
) -> dict[str, Any]:
    """Build graph in Neo4j from agent history.

    - States (nodes) are saved to Neo4j.
    - Transitions (edges) are saved to Neo4j.
    - intent_mode: 'sync' (default, blocking LLM) or 'async' (deferred via IntentWorker).
    """
    from graph_agent.models import State, Transition, Session
    
    if intent_mode is None:
        intent_mode = _resolve_intent_mode()
    use_async = intent_mode == "async"
    
    # Create or update session
    session = Session(
        id=session_id,
        app_id=app_id,
        timestamp=datetime.now(timezone.utc),
        focus="breadth",
    )
    await manager.add_session(session)

    # Track stats
    stats = {
        "states_added": 0,
        "transitions_added": 0,
        "filtered_non_ui_edges": 0,
        "semantic_mismatch_warnings": 0,
        "url_discontinuity_warnings": 0,
        "frame_context_transition_warnings": 0,
    }
    actions = (
        actions
        if actions is not None
        else (list(history.model_actions()) if history else [])
    )
    actions = [_action_to_dict(a) for a in actions]
    thoughts = (
        thoughts
        if thoughts is not None
        else (list(history.model_thoughts()) if history else [])
    )
    if urls is None:
        try:
            urls = list(history.urls()) if history else []
        except Exception:
            urls = []
    urls = _stabilize_url_snapshots(urls)

    # Track processed states and transitions
    processed_states: dict[str, State] = {}
    processed_steps: list[dict[str, str]] = []
    previous_target_url = ""
    previous_had_frame_path = False
    context_window = _resolve_intent_context_window(default=1)
    
    for i, action in enumerate(actions):
        action = _action_to_dict(action)
        thought = thoughts[i] if i < len(thoughts) else {}
        
        raw_from = urls[i] if i < len(urls) else ""
        from_node_id, from_url = _state_from_snapshot(i, raw_from, thought, action)
        to_node_id, to_url = _resolve_target_state(i, urls, thoughts, actions)
        
        # Create or get from_state
        if from_node_id not in processed_states:
            from_state = State(
                id=from_node_id,
                url=from_url or from_node_id,
                title=_extract_next_goal(thought) or from_node_id,
                app_id=app_id,
            )
            await manager.add_state(from_state)
            await manager.link_app_state(app_id, from_state.id)
            await manager.link_session_discovered(session_id, from_state.id)
            processed_states[from_node_id] = from_state
            stats["states_added"] += 1
        else:
            from_state = processed_states[from_node_id]
        
        # Create or get to_state
        if to_node_id not in processed_states:
            to_state = State(
                id=to_node_id,
                url=to_url or to_node_id,
                title="",
                app_id=app_id,
            )
            await manager.add_state(to_state)
            await manager.link_app_state(app_id, to_state.id)
            await manager.link_session_discovered(session_id, to_state.id)
            processed_states[to_node_id] = to_state
            stats["states_added"] += 1
        else:
            to_state = processed_states[to_node_id]

        action_key = _extract_action_key(action)
        if action_key in FILTERED_ACTION_KEYS:
            stats["filtered_non_ui_edges"] += 1
            continue
            
        neighbor_steps = (
            processed_steps[-context_window:]
            if processed_steps and context_window > 0
            else []
        )
        page_signals = _build_page_signals(from_url, to_url, action_key)

        if use_async:
            edge_model = parse_browser_use_step_lite(
                action, thought, from_url, to_url,
            )
        else:
            edge_model = await parse_browser_use_step(
                action, thought, from_url, to_url,
                neighbor_steps=neighbor_steps,
                page_signals=page_signals,
            )

        intent_text = (
            edge_model.intent.summary
            if edge_model.intent
            else f"<missing-intent:{edge_model.intent_failure_reason}>"
        )
        callback_payload = {
            "index": i,
            "source_url": from_url,
            "target_url": to_url,
            "action_key": action_key,
            "edge_model": edge_model,
            "intent_text": intent_text,
        }
        if step_callback and not use_async:
            maybe_result = step_callback(callback_payload)
            if maybe_result and hasattr(maybe_result, "__await__"):
                await maybe_result

        if not use_async and edge_model.intent and not _semantic_consistency(
            edge_model.action, edge_model.intent, selector=edge_model.selector
        ):
            stats["semantic_mismatch_warnings"] += 1
        if previous_target_url and from_url and previous_target_url != from_url:
            stats["url_discontinuity_warnings"] += 1
        current_had_frame_path = bool(getattr(edge_model, "frame_path", []))
        if i > 0 and current_had_frame_path != previous_had_frame_path:
            stats["frame_context_transition_warnings"] += 1

        if edge_model.action == ActionType.UNKNOWN:
            stats["filtered_non_ui_edges"] += 1
            continue
        if edge_model.action == ActionType.NAVIGATE:
            continue
        if (
            edge_model.action in (ActionType.CLICK, ActionType.FILL, ActionType.SELECT, ActionType.RICH_TEXT)
            and edge_model.selector
        ):
            # Create transition
            transition = Transition(
                id=f"{session_id}:step-{i}",
                selector=edge_model.selector,
                action=edge_model.action,
                from_state_id=from_state.id,
                to_state_id=to_state.id,
                intent=edge_model.intent,
                confidence=0.8 if edge_model.intent else 0.5,
            )
            await manager.add_transition(transition)
            await manager.link_session_transition(session_id, transition.id)
            stats["transitions_added"] += 1

        processed_steps.append(
            {
                "action": action_key,
                "selector": edge_model.selector or "",
                "source_url": _clean_url(from_url or ""),
                "target_url": _clean_url(to_url or ""),
                "thought": _extract_next_goal(thought),
            }
        )
        previous_target_url = to_url or from_url or previous_target_url
        previous_had_frame_path = current_had_frame_path

    return stats


# Legacy function for backward compatibility (tests)

def _load_inventory(inventory_path: str | Path) -> list[dict]:
    """Load elements list from inventory JSON; raise if file missing or invalid."""
    path = Path(inventory_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {path}. Run scout first: uv run python -m graph_agent.cartography.runner --url <url> (with --inventory and --output)."
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
        "mode": payload.get("mode", "single_page")
        if isinstance(payload, dict)
        else "single_page",
        "page_count": int(metadata.get("page_count", 1))
        if isinstance(metadata, dict) and metadata is not None
        else 1,
        "aggregated_element_count": int(
            metadata.get(
                "aggregated_element_count",
                payload.get("elements", []) and len(payload.get("elements", [])) or 0,
            )
        )
        if isinstance(payload, dict) and metadata is not None
        else 0,
        "type_counts": metadata.get("type_counts", {})
        if isinstance(metadata, dict) and metadata is not None
        else {},
    }




async def _refresh_business_templates_neo4j(
    manager: "GraphManager", 
    app_id: str
) -> None:
    """Regenerate business templates and store them in Neo4j."""
    print("Generating business templates …")
    try:
        # Get all transitions for the app
        transitions = await manager.get_all_transitions(app_id)
        
        # Get states with intents
        states = await manager.get_states_with_intents(app_id)
        
        # Build a simple in-memory graph for template generation
        import networkx as nx
        G = nx.MultiDiGraph()
        
        # Add nodes (states)
        for state_data in states:
            state_id = state_data.get("state_id", "")
            url = state_data.get("url", "")
            title = state_data.get("title", "")
            G.add_node(state_id, label=title or url, url=url)
        
        # Add edges (transitions)
        for t in transitions:
            from_id = t.get("from_state_id", "")
            to_id = t.get("to_state_id", "")
            if from_id and to_id:
                edge_data = {
                    "selector": t.get("selector", ""),
                    "action": t.get("action", "UNKNOWN"),
                    "intent": t.get("intent"),
                }
                G.add_edge(from_id, to_id, **edge_data)
        
        # Generate templates using existing logic
        templates = await generate_business_templates(G)
        
        # Store templates in Neo4j (in App node)
        from datetime import datetime, timezone
        query = """
        MATCH (a:App {id: $app_id})
        SET a.business_templates = $templates,
            a.business_template_count = $count,
            a.business_template_generation_failures = 0,
            a.business_templates_generated_at = $generated_at
        """
        template_dicts = [t.model_dump(mode="json") for t in templates]
        await manager._run_write(
            query,
            app_id=app_id,
            templates=template_dicts,
            count=len(templates),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"  Generated {len(templates)} business templates")
    except Exception as e:
        print(f"  Error generating business templates: {e}")
        # Store empty templates on error
        from datetime import datetime, timezone
        query = """
        MATCH (a:App {id: $app_id})
        SET a.business_templates = [],
            a.business_template_count = 0,
            a.business_template_generation_failures = 1,
            a.business_templates_generated_at = $generated_at,
            a.business_template_error = $error
        """
        await manager._run_write(
            query,
            app_id=app_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )




def _load_playback_failures(
    feedback: str | Path | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load normalized playback failure records."""
    if isinstance(feedback, list):
        return [item for item in feedback if isinstance(item, dict)]
    path = Path(feedback)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        raw = payload.get("failures", [])
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _edge_locator_keys(
    source: str | None,
    target: str | None,
    edge_id: str | None,
) -> tuple[str, str, str]:
    return (str(source or ""), str(target or ""), str(edge_id or ""))



async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,  # Kept for API compatibility, ignored
    task: str | None = None,
    max_steps: int = 30,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,  # Kept for API compatibility, ignored
) -> str:
    """Run browser-use Agent to explore UI flow and store in Neo4j.

    - url: Start URL (required via arg or MAPPING_URL env).
    - output_path: Kept for API compatibility, data now stored in Neo4j.
    - task: Task description for the agent (default: rendered from DEFAULT_TASK_TEMPLATE).
    - max_steps: Maximum agent steps.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the app_id of the newly created app in Neo4j.
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

    browser_kwargs: dict[str, Any] = {"headless": _resolve_mapping_headless()}
    channel = _resolve_mapping_channel()
    if channel:
        browser_kwargs["channel"] = channel
    try:
        browser = Browser(**browser_kwargs)
    except TypeError:
        # Older browser-use versions may not support channel keyword.
        browser_kwargs.pop("channel", None)
        browser = Browser(**browser_kwargs)
    
    async with managed_browser(browser):
        llm = get_llm()
        initial_actions = [{"navigate": {"url": resolved_url, "new_tab": False}}]
        full_task = resolved_task
        print(f"[DEBUG] Mapping task (length={len(full_task)}): {full_task[:200]}...")

        # Use graph_agent/data as the base for file operations
        # Note: browser-use will append 'browseruse_agent_data' to this path
        data_dir = pkg_root / "data"

        # Note: We use initial_actions instead of directly_open_url=True because
        # browser-use's _extract_start_url has bugs with Chinese text and IP addresses
        agent: Agent = Agent(
            task=full_task,
            llm=llm,
            browser=browser,
            initial_actions=initial_actions,
            file_system_path=str(data_dir),
        )

        # Check for shutdown request before running
        if _shutdown_requested:
            print("[INFO] Shutdown requested before agent run, cleaning up...")
            return ""

        try:
            history = await agent.run(max_steps=max_steps)
        except asyncio.CancelledError:
            print("[INFO] Agent run cancelled, cleaning up...")
            raise

    all_actions, all_thoughts, all_urls = _collect_history_snapshots(history)
    actions, thoughts, urls, runtime_filtered = _runtime_filter_snapshots(
        all_actions, all_thoughts, all_urls
    )

    async def _log_step(info: dict[str, Any]) -> None:
        edge_model = info.get("edge_model")
        step_url = info["source_url"]
        intent_text = info["intent_text"]
        index = int(info["index"]) + 1
        if edge_model:
            print(
                f"  [{index}] url={step_url!r} selector={edge_model.selector!r} "
                f"action={edge_model.action!r} intent={intent_text!r}"
            )
        else:
            print(
                f"  [{index}] url={step_url!r} intent={intent_text!r}"
                f" (async resolved)"
            )

    # Build graph in Neo4j
    from graph_agent.neo4j.manager import GraphManager
    from graph_agent.models import App, Session
    from urllib.parse import urlparse
    
    # Generate app_id and session_id
    parsed = urlparse(resolved_url)
    domain = parsed.netloc or parsed.path.split('/')[0]
    app_name = domain or "unknown"
    app_id = f"app:{app_name}:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    session_id = f"session:{app_id}:{datetime.now(timezone.utc).isoformat()}"
    
    async with GraphManager() as manager:
        # Create app and session
        app = App(id=app_id, name=app_name, base_url=resolved_url)
        await manager.add_app(app)
        
        session = Session(
            id=session_id,
            app_id=app_id,
            start_url=resolved_url,
        )
        await manager.add_session(session)
        
        # Link inventory
        if inventory:
            await manager.set_session_inventory(session_id, inventory)
        
        # Build graph in Neo4j
        stats = await _build_graph_in_neo4j(
            history,
            manager=manager,
            app_id=app_id,
            session_id=session_id,
            inventory=inventory,
            actions=actions,
            thoughts=thoughts,
            urls=urls,
            runtime_non_ui_action_count=runtime_filtered,
            step_callback=_log_step,
        )
        
        # Record visited URLs
        visited_urls = list(dict.fromkeys(u for u in urls if _is_http_url(u or "")))
        
        # Parse stop reason
        final_text = ""
        try:
            if history and hasattr(history, "final_result"):
                final_text = (history.final_result() or "") or ""
            else:
                final_text = str(history) if history else ""
        except Exception:
            final_text = ""
        
        mapping_stopped = False
        stop_reason = None
        for marker in ("Stopped:", "停止："):
            if marker in final_text:
                idx = final_text.find(marker)
                stop_reason = final_text[idx + len(marker) :].strip()
                if len(stop_reason) > 200:
                    stop_reason = stop_reason[:200] + "..."
                mapping_stopped = True
                break
        
        # Update session stats
        await manager.update_session_stats(
            session_id=session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": resolved_url,
                **stats,
            }
        )
        
        # Refresh business templates
        await _refresh_business_templates_neo4j(manager, app_id)
        
        # Print summary
        print(
            f"Graph stored in Neo4j: app_id={app_id} "
            f"(states={stats.get('states_added', 0)}, transitions={stats.get('transitions_added', 0)})"
        )
        
        return app_id


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
    
    # Setup browser-use timeouts from MAPPING_TIMEOUT env
    _setup_browser_use_timeouts()

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
    args = parser.parse_args()
    output = (args.output or "").strip() or default_output
    inventory = (args.inventory or "").strip() or default_inventory
    scout_pages_arg = (args.scout_pages or "").strip()
    scout_pages = [item.strip() for item in scout_pages_arg.split(",") if item.strip()]
    merge_existing = bool(args.merge_existing)
    url = _resolve_mapping_url(args.url)

    async def _run() -> None:
        global _shutdown_requested
        
        if _shutdown_requested:
            print("[INFO] Shutdown requested, exiting before scout...")
            return
            
        print("Step 1: Scout (list interactive elements)...")
        try:
            if scout_pages:
                await run_scout_multi(
                    start_url=url, page_hints=scout_pages, output_path=inventory
                )
            else:
                await run_scout(url, output_path=inventory)
            print(f"Inventory saved: {inventory}")
        except asyncio.CancelledError:
            print("[INFO] Scout cancelled")
            raise
        
        if _shutdown_requested:
            print("[INFO] Shutdown requested, exiting before mapping...")
            return
        
        print("Step 2: Mapping (explore flow, build graph)...")
        try:
            app_id = await run_mapping(
                url=url,
                output_path=output,
                inventory_path=inventory,
                merge_existing=merge_existing,
            )
        except asyncio.CancelledError:
            print("[INFO] Mapping cancelled")
            raise

        # Task 2: Scout derived URLs discovered during mapping to enrich inventory.
        # Get visited_urls from the session in Neo4j
        from graph_agent.neo4j.manager import GraphManager
        visited_urls: list[str] = []
        try:
            async with GraphManager() as manager:
                # Query the latest session for this app
                query = """
                MATCH (a:App {id: $app_id})<-[:BELONGS_TO]-(s:Session)
                RETURN s.visited_urls as urls
                ORDER BY s.created_at DESC
                LIMIT 1
                """
                result = await manager._run_read(query, app_id=app_id)
                if result and result[0].get("urls"):
                    visited_urls = result[0]["urls"]
        except Exception as e:
            print(f"[WARN] Could not retrieve visited URLs from Neo4j: {e}")
        
        # Also try to load from old JSON output for backward compatibility
        if not visited_urls and Path(output).exists():
            try:
                from graph_agent.graph.serialization import load_graph
                G = load_graph(output)
                visited_urls = G.graph.get("visited_urls", [])
            except Exception:
                pass
        derived = extract_derived_urls(visited_urls, url, exclude_start=True)
        if derived and not _shutdown_requested:
            print(f"Step 3: Scout derived URLs ({len(derived)} pages)...")
            try:
                await run_scout_multi(
                    start_url=url, page_hints=derived, output_path=inventory
                )
                print(f"Inventory enriched with derived pages: {inventory}")
            except asyncio.CancelledError:
                print("[INFO] Derived URL scout cancelled")
                raise

    # Use a custom event loop to handle signals and cleanup properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Add signal handlers for the main thread
    def _sigint_handler():
        """Handle Ctrl+C in the main thread."""
        global _shutdown_requested
        _shutdown_requested = True
        print("\n[INFO] SIGINT received, shutting down gracefully...")
        # Cancel all tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
    
    # Use asyncio's signal handling (works on Unix and Windows with ProactorEventLoop)
    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
        loop.add_signal_handler(signal.SIGTERM, _sigint_handler)
    except NotImplementedError:
        # Fallback for Windows with SelectorEventLoop
        signal.signal(signal.SIGINT, lambda s, f: _sigint_handler())
        signal.signal(signal.SIGTERM, lambda s, f: _sigint_handler())
    
    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, initiating shutdown...")
    except asyncio.CancelledError:
        print("\n[INFO] Tasks cancelled, cleaning up...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise
    finally:
        # Clean up browsers
        try:
            loop.run_until_complete(_cleanup_all_browsers())
            print("[INFO] Browser cleanup complete")
        except Exception as cleanup_error:
            print(f"[WARN] Cleanup error: {cleanup_error}")
        
        # Remove signal handlers
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass
        
        loop.close()
        print("[INFO] Shutdown complete")
        sys.exit(130 if _shutdown_requested else 0)


if __name__ == "__main__":
    main()

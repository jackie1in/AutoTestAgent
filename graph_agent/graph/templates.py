"""Offline business template generation from atomic graph paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from graph_agent.llm import ainvoke_structured, get_llm
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    Intent,
)


class BusinessFlowClassification(BaseModel):
    """Classification of whether a path forms a meaningful business workflow."""
    is_business_flow: bool = Field(description="Whether this path forms a coherent business workflow")
    business_key: str = Field(default="", description="Dot-separated business key, e.g. auth.login")
    summary: str = Field(default="", description="Short summary of the business flow")
    slots: dict[str, Any] = Field(default_factory=dict, description="Slot mappings")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence: dict[str, Any] = Field(default_factory=dict, description="Evidence including intent_keys and selectors")

MIN_TEMPLATE_PATH_LENGTH = 2
MAX_TEMPLATE_PATH_LENGTH = 6
MATCHABLE_INTENT_CONFIDENCE = 0.5

EdgeRow = tuple[str, str, Any | None, dict[str, Any]]
TemplateClassifier = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]

_MODULE_NAVIGATION_KEYS = {
    "navigation.menu_selection",
    "project.navigation.menu_selection",
}


def _is_navigation_like_intent_key(intent_key: str | None) -> bool:
    """Heuristic to keep tab/menu wandering out of template classification."""
    if not intent_key:
        return False
    normalized = intent_key.lower()
    if ".navigation." in normalized:
        return True
    if ".select_tab" in normalized or ".select_" in normalized and "tab" in normalized:
        return True
    return normalized.endswith(".tab")


def _iter_out_edges(graph: nx.Graph, node: str) -> list[EdgeRow]:
    """Return deterministic outgoing edges ordered by step_index then selector."""
    rows: list[EdgeRow]
    if isinstance(graph, nx.MultiDiGraph):
        rows = [
            (str(u), str(v), key, data)
            for u, v, key, data in graph.out_edges(node, keys=True, data=True)
        ]
    else:
        rows = [
            (str(u), str(v), None, data)
            for u, v, data in graph.out_edges(node, data=True)
        ]

    def _sort_key(row: EdgeRow) -> tuple[int, str]:
        _u, _v, _k, data = row
        raw = data.get("step_index")
        try:
            step_index = int(raw) if raw is not None else 10**9
        except (TypeError, ValueError):
            step_index = 10**9
        return (step_index, str(data.get("selector", "")))

    return sorted(rows, key=_sort_key)


def _normalize_intent(intent: object) -> Intent | None:
    """Convert dict-like intent payloads into Intent objects when possible."""
    if isinstance(intent, Intent):
        return intent
    if isinstance(intent, dict):
        try:
            return Intent(**intent)
        except Exception:
            return None
    return None


def _candidate_terminal_intent(data: dict[str, Any]) -> Intent | None:
    """Return template-worthy terminal intent or None."""
    action_raw = data.get("action", ActionType.UNKNOWN)
    if isinstance(action_raw, ActionType):
        action = action_raw
    else:
        try:
            action = ActionType(str(action_raw))
        except ValueError:
            action = ActionType.UNKNOWN
    if action != ActionType.CLICK:
        return None
    intent = _normalize_intent(data.get("intent"))
    if intent is None or not intent.key:
        return None
    confidence = intent.confidence if intent.confidence is not None else 0.5
    if confidence < MATCHABLE_INTENT_CONFIDENCE:
        return None
    return intent


def _edge_id_for_row(key: Any | None, data: dict[str, Any]) -> str:
    """Resolve stable edge id for template steps."""
    edge_id = data.get("edge_id")
    if edge_id:
        return str(edge_id)
    if key is not None:
        return str(key)
    step_index = data.get("step_index")
    if step_index is not None:
        return f"step-{step_index}"
    payload = (
        f"{data.get('selector', '')}|{data.get('action', '')}|{data.get('intent')}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _node_url(graph: nx.Graph, node_id: str) -> str:
    """Resolve node url metadata for classifier context."""
    if node_id in graph:
        url = graph.nodes[node_id].get("url")
        if isinstance(url, str):
            return url
    return ""


def _path_to_payload(graph: nx.Graph, path: list[EdgeRow]) -> dict[str, Any]:
    """Build compact classifier payload from atomic edge path."""
    first_u, _, _, _first_data = path[0]
    _, last_v, _, _last_data = path[-1]
    steps: list[dict[str, Any]] = []
    selectors: list[str] = []
    intent_keys: list[str] = []
    navigation_like_count = 0
    for u, v, key, data in path:
        intent = _normalize_intent(data.get("intent"))
        intent_key = intent.key if intent else None
        selector = str(data.get("selector", ""))
        if selector:
            selectors.append(selector)
        if intent_key:
            intent_keys.append(intent_key)
            if _is_navigation_like_intent_key(intent_key):
                navigation_like_count += 1
        action_value = data.get("action")
        action = (
            action_value.value
            if isinstance(action_value, ActionType)
            else str(action_value or "")
        )
        steps.append(
            {
                "edge_id": _edge_id_for_row(key, data),
                "source": u,
                "target": v,
                "selector": selector,
                "action": action,
                "intent_key": intent_key,
                "param_name": data.get("param_name"),
            }
        )
    return {
        "entry_node": first_u,
        "exit_node": last_v,
        "path_length": len(path),
        "source_url": _node_url(graph, first_u) or first_u,
        "target_url": _node_url(graph, last_v) or last_v,
        "steps": steps,
        "signals": {
            "selectors": selectors,
            "intent_keys": intent_keys,
            "has_fill": any(step["action"] == ActionType.FILL.value for step in steps),
            "has_click": any(
                step["action"] == ActionType.CLICK.value for step in steps
            ),
            "has_non_navigation_intent": any(
                intent_key and not _is_navigation_like_intent_key(intent_key)
                for intent_key in intent_keys
            ),
            "navigation_like_ratio": (
                navigation_like_count / len(intent_keys) if intent_keys else 0.0
            ),
        },
    }


def _is_business_path_candidate(payload: dict[str, Any]) -> bool:
    """Cheap filter before asking the classifier to label a path."""
    signals = payload.get("signals", {})
    if not bool(signals.get("has_click")):
        return False
    if bool(signals.get("has_fill")):
        return True

    path_length = int(payload.get("path_length") or 0)
    if path_length < MIN_TEMPLATE_PATH_LENGTH:
        return False

    source_url = str(payload.get("source_url") or "")
    target_url = str(payload.get("target_url") or "")
    crossed_url = bool(source_url and target_url and source_url != target_url)
    has_non_navigation_intent = bool(signals.get("has_non_navigation_intent"))
    navigation_like_ratio = float(signals.get("navigation_like_ratio") or 0.0)
    if navigation_like_ratio >= 1.0:
        return False
    return crossed_url or has_non_navigation_intent


def _response_to_text(response: Any) -> str:
    """Normalize model response into text."""
    if hasattr(response, "completion"):
        return str(response.completion)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(getattr(item, "text", getattr(item, "content", item)))
            for item in content
        )
    return str(response)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse first JSON object from text."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def _classify_candidate_with_llm(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Use the configured LLM to decide whether a path is a business flow."""
    system_prompt = """You classify atomic browser interaction paths into higher-level business flows.

Given the path JSON, decide whether it forms one meaningful business workflow
such as auth.login, auth.register, search.execute, checkout.submit."""
    
    user_prompt = f"""Classify the following path JSON:

{json.dumps(payload, ensure_ascii=False)}

Rules:
- Prefer high-level business concepts over atomic actions.
- If the path is not a coherent business flow, return is_business_flow=false.
- If true, confidence must be between 0 and 1."""
    
    llm = get_llm()
    try:
        result = await ainvoke_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_format=BusinessFlowClassification,
        )
        return result.model_dump()
    except Exception:
        return None


def _make_template(
    payload: dict[str, Any], classification: dict[str, Any]
) -> BusinessTemplate:
    """Build the final BusinessTemplate model."""
    steps = [
        BusinessTemplateStep(
            edge_id=str(step.get("edge_id") or ""),
            source=str(step.get("source") or ""),
            target=str(step.get("target") or ""),
            selector=str(step.get("selector") or ""),
            action=ActionType(str(step.get("action") or ActionType.UNKNOWN.value)),
            intent_key=str(step.get("intent_key"))
            if step.get("intent_key") is not None
            else None,
            param_name=str(step.get("param_name"))
            if step.get("param_name") is not None
            else None,
        )
        for step in payload["steps"]
    ]
    business_key = str(classification.get("business_key") or "").strip()
    if business_key in _MODULE_NAVIGATION_KEYS:
        business_key = "navigation.module.select"
    edge_ids = ",".join(step.edge_id or "" for step in steps)
    template_id = hashlib.sha1(
        f"{business_key}|{edge_ids}".encode("utf-8")
    ).hexdigest()[:16]
    confidence_raw = classification.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    evidence_raw = classification.get("evidence")
    evidence = evidence_raw if isinstance(evidence_raw, dict) else {}
    slots_raw = classification.get("slots")
    slots = slots_raw if isinstance(slots_raw, dict) else {}
    return BusinessTemplate(
        template_id=template_id,
        business_key=business_key,
        summary=str(classification.get("summary") or business_key),
        entry_node=str(payload["entry_node"]),
        exit_node=str(payload["exit_node"]),
        path_length=int(payload["path_length"]),
        confidence=confidence,
        steps=steps,
        slots=slots,
        evidence=evidence,
    )


def _prefer_template(
    current: BusinessTemplate | None, candidate: BusinessTemplate
) -> BusinessTemplate:
    """Choose the better template for the same business/terminal flow."""
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


def _choose_dependency_candidate(
    template: BusinessTemplate,
    templates: list[BusinessTemplate],
    graph: nx.Graph,
) -> BusinessTemplate | None:
    """Select the most plausible immediate predecessor template.

    Prefers direct connection (other.exit_node == template.entry_node).
    When no direct match exists, considers templates whose exit_node has a path
    to template.entry_node in the graph, preferring the one with shortest path.
    """
    direct_candidate: BusinessTemplate | None = None
    path_candidates: list[tuple[int, BusinessTemplate]] = []

    for other in templates:
        if other.business_key == template.business_key:
            continue
        if other.exit_node == template.entry_node:
            direct_candidate = _prefer_template(direct_candidate, other)
            continue
        if other.exit_node not in graph or template.entry_node not in graph:
            continue
        try:
            if not nx.has_path(graph, other.exit_node, template.entry_node):
                continue
        except Exception:
            continue
        try:
            path_len = (
                len(nx.shortest_path(graph, other.exit_node, template.entry_node)) - 1
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        path_candidates.append((path_len, other))

    if direct_candidate is not None:
        return direct_candidate
    if not path_candidates:
        return None
    path_candidates.sort(key=lambda x: x[0])
    best: BusinessTemplate | None = None
    for _path_len, other in path_candidates:
        best = _prefer_template(best, other)
    return best


def _attach_direct_template_dependencies(
    graph: nx.Graph,
    templates: list[BusinessTemplate],
) -> list[BusinessTemplate]:
    """Attach one immediate business predecessor when templates touch at a state boundary or are reachable via graph path."""
    for template in templates:
        if template.depends_on:
            continue
        dependency = _choose_dependency_candidate(template, templates, graph)
        if dependency is not None:
            template.depends_on.append(dependency.business_key)
    return templates


def _is_login_like_url(url: str) -> bool:
    """Identify login-style entry pages that should not depend on auth.login."""
    normalized = url.lower()
    return "login" in normalized or "signin" in normalized or "sign-in" in normalized


def _attach_auth_login_dependency(
    graph: nx.Graph,
    templates: list[BusinessTemplate],
) -> list[BusinessTemplate]:
    """Attach auth.login as a root prerequisite for non-auth templates missing one."""
    auth_templates = [
        template for template in templates if template.business_key == "auth.login"
    ]
    if not auth_templates:
        return templates

    auth_template = auth_templates[0]
    for candidate in auth_templates[1:]:
        auth_template = _prefer_template(auth_template, candidate)

    if auth_template.exit_node not in graph:
        auth_exit_url = ""
    else:
        auth_exit_url = _node_url(graph, auth_template.exit_node)

    for template in templates:
        if template.business_key == "auth.login":
            continue
        if template.depends_on:
            continue
        if template.business_key.startswith("auth."):
            continue
        entry_url = (
            _node_url(graph, template.entry_node)
            if template.entry_node in graph
            else ""
        )
        if entry_url and _is_login_like_url(entry_url):
            continue
        reachable = False
        if auth_template.exit_node in graph and template.entry_node in graph:
            try:
                reachable = nx.has_path(
                    graph, auth_template.exit_node, template.entry_node
                )
            except Exception:
                reachable = False
        if (
            reachable
            or not auth_exit_url
            or not entry_url
            or not _is_login_like_url(entry_url)
        ):
            template.depends_on.append("auth.login")
    return templates


def _is_module_navigation_edge(data: dict[str, Any]) -> bool:
    """Heuristic: identify post-login module-entry menu clicks."""
    action_raw = data.get("action", ActionType.UNKNOWN)
    if isinstance(action_raw, ActionType):
        action = action_raw
    else:
        try:
            action = ActionType(str(action_raw))
        except ValueError:
            action = ActionType.UNKNOWN
    if action != ActionType.CLICK:
        return False
    intent = _normalize_intent(data.get("intent"))
    key = str(intent.key or "").lower() if intent else ""
    if not key:
        return False
    if "tab" in key:
        return False
    return (
        ".navigation." in key
        or ".menu" in key
        or ".select." in key
        or key.endswith(".select")
    )


def _build_module_navigation_template(
    auth_template: BusinessTemplate,
    graph: nx.Graph,
) -> BusinessTemplate | None:
    """Synthesize a module-entry template from short post-login menu click chains."""
    if auth_template.exit_node not in graph:
        return None

    best_path: list[EdgeRow] = []
    for first in _iter_out_edges(graph, auth_template.exit_node):
        _u1, v1, _k1, data1 = first
        if not _is_module_navigation_edge(data1):
            continue
        candidate = [first]
        for second in _iter_out_edges(graph, str(v1)):
            _u2, _v2, _k2, data2 = second
            if not _is_module_navigation_edge(data2):
                continue
            candidate = [first, second]
            break
        if len(candidate) > len(best_path):
            best_path = candidate

    if not best_path:
        return None

    steps: list[BusinessTemplateStep] = []
    evidence_keys: list[str] = []
    for u, v, key, data in best_path:
        edge_id = _edge_id_for_row(key, data)
        intent = _normalize_intent(data.get("intent"))
        intent_key = intent.key if intent else None
        if intent_key:
            evidence_keys.append(intent_key)
        action_raw = data.get("action", ActionType.UNKNOWN)
        action = (
            action_raw if isinstance(action_raw, ActionType) else ActionType(str(action_raw))
        )
        steps.append(
            BusinessTemplateStep(
                edge_id=edge_id,
                source=str(u),
                target=str(v),
                selector=str(data.get("selector") or ""),
                action=action,
                intent_key=intent_key,
                param_name=str(data.get("param_name"))
                if data.get("param_name") is not None
                else None,
            )
        )

    template_id = hashlib.sha1(
        (
            "navigation.module.select|"
            + ",".join(step.edge_id or "" for step in steps)
        ).encode("utf-8")
    ).hexdigest()[:16]
    return BusinessTemplate(
        template_id=template_id,
        business_key="navigation.module.select",
        summary="登录后进入目标模块",
        entry_node=str(best_path[0][0]),
        exit_node=str(best_path[-1][1]),
        path_length=len(steps),
        confidence=0.76,
        steps=steps,
        slots={},
        evidence={"intent_keys": evidence_keys, "source": "heuristic.module_navigation"},
    )


def _synthesize_navigation_module_template(
    graph: nx.Graph,
    templates: list[BusinessTemplate],
) -> list[BusinessTemplate]:
    """Add a deterministic post-login module-entry template when LLM classification is too generic."""
    if any(t.business_key == "navigation.module.select" for t in templates):
        return templates
    auth_templates = [t for t in templates if t.business_key == "auth.login"]
    if not auth_templates:
        return templates
    auth_template = auth_templates[0]
    for candidate in auth_templates[1:]:
        auth_template = _prefer_template(auth_template, candidate)
    synthesized = _build_module_navigation_template(auth_template, graph)
    if synthesized is None:
        return templates
    return [*templates, synthesized]


async def generate_business_templates(
    graph: nx.Graph,
    classifier: TemplateClassifier | None = None,
    min_path_length: int = MIN_TEMPLATE_PATH_LENGTH,
    max_path_length: int = MAX_TEMPLATE_PATH_LENGTH,
    concurrency: int = 5,
) -> list[BusinessTemplate]:
    """Generate deduplicated high-level business templates from graph paths.

    Classification LLM calls run concurrently (bounded by ``concurrency``).
    """
    if graph.number_of_edges() == 0:
        return []

    classify = classifier if classifier is not None else _classify_candidate_with_llm

    # Phase 1: collect candidate payloads via DFS (no LLM calls)
    candidates: list[dict[str, Any]] = []
    for start_node in graph.nodes():
        stack: list[tuple[str, list[EdgeRow], set[str]]] = [
            (str(start_node), [], set())
        ]
        while stack:
            node, path, used_edge_ids = stack.pop()
            if len(path) >= max_path_length:
                continue
            for row in reversed(_iter_out_edges(graph, node)):
                u, v, key, data = row
                edge_id = _edge_id_for_row(key, data)
                if edge_id in used_edge_ids:
                    continue
                next_path = [*path, row]
                next_used = set(used_edge_ids)
                next_used.add(edge_id)
                terminal_intent = _candidate_terminal_intent(data)
                if terminal_intent is not None and len(next_path) >= min_path_length:
                    payload = _path_to_payload(graph, next_path)
                    if _is_business_path_candidate(payload):
                        candidates.append(payload)
                stack.append((v, next_path, next_used))

    if not candidates:
        return []

    # Phase 2: classify candidates concurrently
    print(f"  Classifying {len(candidates)} business template candidates "
          f"(concurrency={concurrency}) …")
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def classify_one(
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        nonlocal completed
        async with sem:
            try:
                result = await classify(payload)
            except Exception:
                result = None
            completed += 1
            if completed % 10 == 0 or completed == len(candidates):
                print(f"    template classification: {completed}/{len(candidates)}")
            return payload, result

    results = await asyncio.gather(*[classify_one(p) for p in candidates])

    # Phase 3: build deduplicated templates from results
    deduped: dict[tuple[str, str], BusinessTemplate] = {}
    for payload, classified in results:
        if (
            classified
            and classified.get("is_business_flow") is True
            and classified.get("business_key")
        ):
            template = _make_template(payload, classified)
            dedupe_key = (
                template.business_key,
                template.steps[-1].edge_id or "",
            )
            deduped[dedupe_key] = _prefer_template(
                deduped.get(dedupe_key), template
            )

    templates = _synthesize_navigation_module_template(graph, list(deduped.values()))
    templates = _attach_direct_template_dependencies(graph, templates)
    return _attach_auth_login_dependency(graph, templates)


def store_business_templates(
    graph: nx.Graph, templates: list[BusinessTemplate]
) -> None:
    """Persist generated templates into graph metadata as plain JSON-ready dicts."""
    graph.graph["business_templates"] = [
        template.model_dump(mode="json") for template in templates
    ]
    graph.graph["business_template_count"] = len(templates)
    graph.graph["business_template_generation_failures"] = 0
    graph.graph["business_templates_generated_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    graph.graph["business_template_stats"] = {"count": len(templates)}

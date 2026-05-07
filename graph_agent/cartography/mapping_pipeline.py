from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from browser_use.browser.session import BrowserSession as Browser
from browser_use.llm.base import BaseChatModel

from graph_agent.cartography.browser_lifecycle import is_shutdown_requested
from graph_agent.cartography.config import (
    clean_url,
    is_http_url,
    is_login_url,
    resolve_knowledge_min_interval_sec,
    resolve_knowledge_on_demand_enabled,
    resolve_knowledge_query_timeout_ms,
    resolve_knowledge_topk,
    resolve_knowledge_trigger_profile,
    resolve_knowledge_trigger_score_threshold,
    resolve_layout_aware_enabled,
    resolve_layout_confidence_retry_enabled,
    resolve_layout_confidence_threshold,
    resolve_layout_snapshot_limit,
    resolve_orchestration_max_runtime_sec,
    resolve_pipeline_checkpoint_path,
    resolve_pipeline_resume_from_checkpoint,
    resolve_skip_advisor_enabled,
    resolve_skip_cache_ttl_sec,
    resolve_skip_policy_profile,
    resolve_skip_query_timeout_ms,
    resolve_skip_ttl_hours,
    same_origin,
)
from graph_agent.cartography.intervention_queue import evaluate_intervention_need
from graph_agent.cartography.knowledge_broker import (
    KnowledgeBroker,
    KnowledgeQueryInput,
)
from graph_agent.cartography.skip_advisor import (
    SkipAdvisor,
    SkipDecision,
    SkipKind,
    SkipPolicy,
)
from graph_agent.cartography.layout_snapshot import (
    build_layout_summary,
    capture_layout_snapshot,
    compute_layout_fingerprint,
    estimate_layout_confidence,
)
from graph_agent.cartography.llm_planning import (
    LLMPageAnalysis,
    analyze_page_with_llm,
    build_exploration_guidance,
    build_login_hint_from_env,
    plan_next_exploration_with_llm,
)
from graph_agent.cartography.types import (
    LayoutEvidenceItem,
    LayoutMetrics,
    LLMTransitionHint,
)
from graph_agent.lib.observability import observe

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import ZoneType

logger = logging.getLogger(__name__)


def _now_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _parse_evaluate_result(raw: object) -> dict[str, object]:
    import json

    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _serialize_skip_decision(decision: SkipDecision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "kind": decision.kind.value,
        "reason": decision.reason,
        "confidence": float(decision.confidence),
        "target_zone_selectors": list(decision.target_zone_selectors),
        "coverage": float(decision.coverage),
        "state_count": int(decision.state_count),
        "last_visited_age_h": float(decision.last_visited_age_h or 0.0),
        "last_explored_age_h": float(decision.last_explored_age_h or 0.0),
        "release_coverage": float(decision.release_coverage),
        "intent_confirm_total": int(decision.intent_confirm_total),
        "entity_confirm_total": int(decision.entity_confirm_total),
        "intent_confirmed_zone_count": int(decision.intent_confirmed_zone_count),
        "circuit_open": bool(decision.circuit_open),
    }


def _deserialize_skip_decision(raw: object) -> SkipDecision | None:
    if not isinstance(raw, dict):
        return None
    kind_raw = str(raw.get("kind") or "").strip().lower()
    kind = {
        SkipKind.SKIP_PAGE.value: SkipKind.SKIP_PAGE,
        SkipKind.EXPLORE_ZONES_ONLY.value: SkipKind.EXPLORE_ZONES_ONLY,
        SkipKind.FULL_EXPLORE.value: SkipKind.FULL_EXPLORE,
    }.get(kind_raw, SkipKind.FULL_EXPLORE)
    return SkipDecision(
        kind=kind,
        reason=str(raw.get("reason") or ""),
        confidence=float(raw.get("confidence") or 0.0),
        target_zone_selectors=list(raw.get("target_zone_selectors") or []),
        coverage=float(raw.get("coverage") or 0.0),
        state_count=int(raw.get("state_count") or 0),
        last_visited_age_h=float(raw.get("last_visited_age_h") or 0.0),
        last_explored_age_h=float(raw.get("last_explored_age_h") or 0.0),
        release_coverage=float(raw.get("release_coverage") or 0.0),
        intent_confirm_total=int(raw.get("intent_confirm_total") or 0),
        entity_confirm_total=int(raw.get("entity_confirm_total") or 0),
        intent_confirmed_zone_count=int(raw.get("intent_confirmed_zone_count") or 0),
        circuit_open=bool(raw.get("circuit_open")),
    )


def _classify_pipeline_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate" in text and "limit" in text:
        return "rate_limited"
    if "401" in text or "403" in text or "forbidden" in text or "unauthorized" in text:
        return "unauthorized"
    if "timeout" in text:
        return "timeout"
    return "unknown"


def _looks_rate_limited(dom_text: str, url: str) -> bool:
    sample = f"{dom_text or ''} {url or ''}".lower()
    return any(
        marker in sample
        for marker in (
            "429",
            "too many requests",
            "rate limit",
            "访问受限",
            "请求过于频繁",
        )
    )


async def click_menu_by_text(browser: Browser, text: str) -> bool:
    """Click a menu item by its visible text label."""
    target = (text or "").strip()
    if not target:
        return False
    script = (
        "(target) => {\n"
        "  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();\n"
        "  const candidates = [\n"
        "    'a[role=\"menuitem\"]', '[role=\"menuitem\"]',\n"
        "    '.ant-menu-item', '.el-menu-item', '.el-submenu__title',\n"
        "    '.ant-menu-submenu-title', '.menu-item', '[class*=\"menu-item\"]',\n"
        "    'aside a', 'aside button', 'nav a', 'nav button',\n"
        "    'a', 'button', '[role=\"button\"]'\n"
        "  ];\n"
        "  const seen = new Set();\n"
        "  for (const sel of candidates) {\n"
        "    const els = Array.from(document.querySelectorAll(sel));\n"
        "    for (const el of els) {\n"
        "      if (seen.has(el)) continue;\n"
        "      seen.add(el);\n"
        "      const t = norm(el.innerText || el.textContent);\n"
        "      if (!t) continue;\n"
        "      if (t === target || (t.length <= 40 && t.includes(target))) {\n"
        "        const rect = el.getBoundingClientRect();\n"
        "        if (rect.width === 0 || rect.height === 0) continue;\n"
        "        el.scrollIntoView({block: 'center'});\n"
        "        el.click();\n"
        "        return JSON.stringify({ok: true, tag: el.tagName, selector: sel});\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  return JSON.stringify({ok: false});\n"
        "}"
    )
    try:
        page = await browser.get_current_page()
        if page is None:
            return False
        raw = await page.evaluate(script, target)
        parsed = _parse_evaluate_result(raw)
        if parsed.get("ok"):
            logger.info(
                f"[PIPELINE] Menu '{target}' clicked "
                f"(via {parsed.get('selector', '?')}, tag={parsed.get('tag', '?')})"
            )
            return True
        logger.info(f"[PIPELINE] Menu '{target}' not found on current page")
        return False
    except Exception as e:
        logger.warning("[PIPELINE] click_menu_by_text error: %s", e)
        return False


def rank_warm_start_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """排序优先级（越靠前越先探索）：

    1. ExplorationScheduler 注入的高优先任务（按 ``scheduler_priority`` 降序）
    2. 历史发现但 zone 未探的（``zone_unexplored=True``）
    3. 历史 transition 置信度高
    """

    def _to_float(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    def _to_int(value: object) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    return sorted(
        candidates,
        key=lambda item: (
            # scheduler_priority 越大越靠前 → 取负
            -_to_int(item.get("scheduler_priority") or 0),
            0 if item.get("zone_unexplored") else 1,
            -(_to_float(item.get("confidence") or 0.0)),
            str(item.get("transition_id") or ""),
        ),
    )


def map_llm_zone_type(zone_type: str) -> "ZoneType | None":
    from graph_agent.models import ZoneType

    mapping: dict[str, "ZoneType"] = {
        "form": ZoneType.DETAIL_FORM,
        "table": ZoneType.DATA_TABLE,
        "nav": ZoneType.TREE_PANEL,
        "action-bar": ZoneType.ACTION_BAR,
        "action_bar": ZoneType.ACTION_BAR,
        "filter-panel": ZoneType.SEARCH_FORM,
        "filter_panel": ZoneType.SEARCH_FORM,
        "modal": ZoneType.MODAL,
        "tabs": ZoneType.TAB_PANEL,
        "tab_panel": ZoneType.TAB_PANEL,
        "pagination": ZoneType.DATA_TABLE,
        "card": ZoneType.DETAIL_FORM,
        "list": ZoneType.DATA_TABLE,
        "chart": ZoneType.DETAIL_FORM,
        "steps": ZoneType.TAB_PANEL,
        "content": ZoneType.DETAIL_FORM,
    }
    return mapping.get(zone_type.lower().strip())


def _build_runtime_zone_id(
    *,
    zone_type: str,
    selector: str,
    source_url: str = "",
) -> str:
    zone_type_norm = (zone_type or "").strip().lower() or "content"
    selector_norm = (selector or "").strip()
    source_key = clean_url(source_url or "") if source_url else ""
    seed = f"{zone_type_norm}|{selector_norm}|{source_key}"
    return f"zone:{hashlib.md5(seed.encode()).hexdigest()[:12]}"


async def ensure_browser_ready(browser: Browser, target_url: str) -> bool:
    try:
        current = await browser.get_current_page_url()
        if current:
            return True
    except Exception:
        pass
    try:
        logger.info("[PIPELINE] Browser session appears reset, attempting restart...")
        await browser.start()
        await browser.navigate_to(target_url)
        await asyncio.sleep(2)
        return True
    except Exception as e:
        logger.warning("[PIPELINE] Browser restart failed: %s", e)
        return False


async def collect_layout_context(
    browser: Browser,
    *,
    enabled: bool,
    limit: int,
) -> tuple[str, str, float]:
    if not enabled:
        return "", "", 0.0
    try:
        page = await browser.get_current_page()
        if page is None:
            return "", "", 0.0
        snapshot = await capture_layout_snapshot(page, limit=limit)
        return (
            build_layout_summary(snapshot),
            compute_layout_fingerprint(snapshot),
            estimate_layout_confidence(snapshot),
        )
    except Exception:
        return "", "", 0.0


def is_low_layout_confidence(
    confidence: float,
    *,
    enabled: bool,
    threshold: float,
) -> bool:
    return enabled and confidence > 0.0 and confidence < threshold


def summarize_layout_metrics(
    samples: list[float],
    low_confidence_hits: int,
    low_confidence_page_types: dict[str, int],
    evidence_count: int,
) -> LayoutMetrics:
    avg = round(sum(samples) / len(samples), 4) if samples else 0.0
    min_v = round(min(samples), 4) if samples else 0.0
    max_v = round(max(samples), 4) if samples else 0.0
    return {
        "layout_confidence_samples": len(samples),
        "layout_confidence_avg": avg,
        "layout_confidence_min": min_v,
        "layout_confidence_max": max_v,
        "layout_low_confidence_hits": low_confidence_hits,
        "layout_low_confidence_page_types": dict(low_confidence_page_types),
        "layout_evidence_count": evidence_count,
    }


def compute_knowledge_trigger_score(
    *,
    low_layout_confidence_hits: int,
    failed_action_count: int,
    semantic_conflict_count: int,
    stuck_steps: int,
    profile: str = "balanced",
) -> float:
    profile_norm = (profile or "balanced").strip().lower()
    if profile_norm == "aggressive":
        w_low = 1.3
        w_fail = 1.4
        w_conflict = 1.1
        w_stuck = 1.5
    elif profile_norm == "conservative":
        w_low = 0.8
        w_fail = 0.9
        w_conflict = 0.7
        w_stuck = 1.0
    else:
        w_low = 1.0
        w_fail = 1.0
        w_conflict = 1.0
        w_stuck = 1.0
    score = 0.0
    if low_layout_confidence_hits >= 2:
        score += (1.0 + min(1.0, (low_layout_confidence_hits - 2) * 0.2)) * w_low
    if failed_action_count >= 2:
        score += (1.0 + min(1.0, (failed_action_count - 2) * 0.2)) * w_fail
    if semantic_conflict_count >= 1:
        score += (0.8 + min(1.0, (semantic_conflict_count - 1) * 0.2)) * w_conflict
    if stuck_steps >= 2:
        score += (1.2 + min(1.0, (stuck_steps - 2) * 0.2)) * w_stuck
    return round(score, 3)


def should_query_knowledge(
    *,
    enabled: bool,
    now_ts: float,
    last_query_ts: float,
    min_interval_sec: float,
    score: float,
    threshold: float,
) -> bool:
    if not enabled:
        return False
    if score < threshold:
        return False
    if now_ts - last_query_ts < min_interval_sec:
        return False
    return True


def build_knowledge_hint_text(
    summary: str, transition_hints: list[dict[str, object]]
) -> str:
    if not summary and not transition_hints:
        return ""
    lines = [
        "HISTORICAL_HINTS (advisory only; always trust current DOM first):",
    ]
    if summary:
        lines.append(f"- summary: {summary}")
    for item in transition_hints[:3]:
        lines.append(
            "- action={action}, selector={selector}, conf={confidence:.2f}".format(
                action=str(item.get("action") or ""),
                selector=str(item.get("selector") or ""),
                confidence=float(item.get("confidence") or 0.0),  # type: ignore[arg-type]
            )
        )
    lines.append("- If hint conflicts with current page evidence, ignore the hint.")
    return "\n".join(lines)


async def _enqueue_ranked_warm_candidates(
    *,
    warm_start_candidates: list[dict[str, object]],
    current_url: str,
    start_url: str,
    primary_origin_url: str,
    enqueue_page,
) -> None:
    warm_candidates = rank_warm_start_candidates(warm_start_candidates or [])
    for item in warm_candidates:
        target_url = str(item.get("target_url") or "").strip()
        if not target_url:
            continue
        if clean_url(target_url) == clean_url(current_url or start_url):
            continue
        if primary_origin_url and not same_origin(target_url, primary_origin_url):
            continue
        sched_task_type = str(item.get("scheduler_task_type") or "")
        sched_hint = ""
        if sched_task_type == "explore_zone":
            sched_reason = str(item.get("scheduler_reason") or "")
            sched_hint = (
                "stale_re_explore"
                if sched_reason == "stale_re_explore"
                else "explore_zone"
            )
        reason = f"scheduler:{sched_task_type}" if sched_task_type else "warm-start"
        await enqueue_page(target_url, reason, scheduler_hint=sched_hint)


async def _maybe_inject_knowledge_hint(
    *,
    knowledge_broker: KnowledgeBroker | None,
    knowledge_enabled: bool,
    now_ts: float,
    last_query_ts: float,
    min_interval_sec: float,
    score: float,
    threshold: float,
    all_transitions,
    app_id: str,
    session_id: str,
    current_page_url: str,
    page_type: str,
    layout_fingerprint: str,
    knowledge_release_id: str,
    knowledge_topk_val: int,
    knowledge_timeout_ms: int,
) -> tuple[str, dict[str, object], float]:
    if not should_query_knowledge(
        enabled=knowledge_enabled,
        now_ts=now_ts,
        last_query_ts=last_query_ts,
        min_interval_sec=min_interval_sec,
        score=score,
        threshold=threshold,
    ) or knowledge_broker is None:
        return "", {}, last_query_ts

    latest_transition = all_transitions[-1] if all_transitions else None
    recent_selector = ""
    recent_action = ""
    if latest_transition is not None:
        recent_selector = str(getattr(latest_transition, "selector", "") or "")
        recent_action = str(getattr(latest_transition, "action", "") or "")

    result = await knowledge_broker.query(
        KnowledgeQueryInput(
            app_id=app_id,
            session_id=session_id,
            current_url=current_page_url,
            page_type=page_type,
            layout_fingerprint=layout_fingerprint,
            recent_selector=recent_selector,
            recent_action=recent_action,
            release_id=knowledge_release_id,
            signals={"trigger_score": score},
            top_k=knowledge_topk_val,
        ),
        timeout_ms=knowledge_timeout_ms,
    )
    metrics_delta: dict[str, object] = {
        "knowledge_query_count": 1,
        "knowledge_latency_ms": result.meta.query_latency_ms,
        "knowledge_cache_hit_count": 1 if result.meta.cache_hit else 0,
        "knowledge_timeout_count": 1 if result.meta.timed_out else 0,
        "knowledge_circuit_open_count": 1 if result.meta.circuit_open else 0,
        "knowledge_error_count": 1 if result.meta.error else 0,
        "knowledge_hit_count": 1
        if (result.transition_hints or result.intent_hints or result.state_hints)
        else 0,
    }
    hint_text = ""
    if metrics_delta["knowledge_hit_count"]:
        hint_text = build_knowledge_hint_text(result.summary, result.transition_hints)
    return hint_text, metrics_delta, now_ts


def _build_pipeline_result(
    *,
    all_states,
    all_transitions,
    all_zones,
    all_history,
    menu_items_discovered,
    zones_discovered,
    layout_evidence,
    layout_confidence_samples: list[float],
    low_layout_confidence_hits: int,
    low_layout_confidence_page_types: Counter[str],
    knowledge_query_count: int,
    knowledge_hit_count: int,
    knowledge_cache_hit_count: int,
    knowledge_timeout_count: int,
    knowledge_circuit_open_count: int,
    knowledge_error_count: int,
    knowledge_latency_total_ms: float,
    knowledge_profile: str,
    skip_advisor: SkipAdvisor | None,
    skip_metrics: dict[str, int],
    session_id: str,
    current_url: str,
    start_url: str,
    failed_action_count: int,
    semantic_conflict_count: int,
    cross_origin_seen: bool,
    iframe_seen: bool,
    captcha_seen: bool,
) -> "CartographyResult":
    from graph_agent.graph.merger import CartographyResult

    result = CartographyResult()
    result.states = all_states
    result.transitions = all_transitions
    result.zones = all_zones
    result.history = all_history
    result.menus = menu_items_discovered
    result.zone_hints = zones_discovered
    result.layout_evidence = [dict(item) for item in layout_evidence]
    result.layout_metrics = dict(
        summarize_layout_metrics(
            samples=layout_confidence_samples,
            low_confidence_hits=low_layout_confidence_hits,
            low_confidence_page_types=dict(low_layout_confidence_page_types),
            evidence_count=len(layout_evidence),
        )
    )
    result.layout_metrics.update(
        {
            "knowledge_query_count": knowledge_query_count,
            "knowledge_hit_count": knowledge_hit_count,
            "knowledge_cache_hit_count": knowledge_cache_hit_count,
            "knowledge_timeout_count": knowledge_timeout_count,
            "knowledge_circuit_open_count": knowledge_circuit_open_count,
            "knowledge_error_count": knowledge_error_count,
            "knowledge_avg_latency_ms": round(
                knowledge_latency_total_ms / knowledge_query_count, 3
            )
            if knowledge_query_count > 0
            else 0.0,
            "knowledge_trigger_profile": knowledge_profile,
        }
    )
    result.layout_metrics.update(
        {
            "skip_advisor_enabled": skip_advisor is not None,
            "skip_page_in_enqueue": skip_metrics["skip_page_in_enqueue"],
            "skip_page_in_loop": skip_metrics["skip_page_in_loop"],
            "skip_zones_only_in_loop": skip_metrics["zones_only_in_loop"],
            "pipeline_error_rate_limited_count": skip_metrics["error_rate_limited_count"],
            "pipeline_error_unauthorized_count": skip_metrics["error_unauthorized_count"],
            "pipeline_error_timeout_count": skip_metrics["error_timeout_count"],
            "pipeline_error_unknown_count": skip_metrics["error_unknown_count"],
        }
    )
    if skip_advisor is not None:
        result.layout_metrics.update(skip_advisor.metrics)
    intervention_tasks = evaluate_intervention_need(
        session_id=session_id,
        source_url=current_url or start_url,
        page_type="mixed",
        low_layout_confidence_hits=low_layout_confidence_hits,
        failed_action_count=failed_action_count,
        semantic_conflict_count=semantic_conflict_count,
        has_cross_origin=cross_origin_seen,
        has_iframe=iframe_seen,
        has_captcha=captcha_seen,
    )
    result.intervention_tasks = [
        {
            "task_id": task.task_id,
            "reason": task.reason,
            "source_url": task.source_url,
            "page_type": task.page_type,
            "context": task.context,
            "status": task.status,
        }
        for task in intervention_tasks
    ]
    result.semantic_conflict_count = semantic_conflict_count
    return result


def _update_page_zone_progress(
    *,
    page_analysis: LLMPageAnalysis,
    all_zones,
    zones_discovered: list[dict[str, object]],
    current_page_url: str,
    new_transitions_this_page: int,
) -> None:
    from graph_agent.models import ExplorationStatus

    page_zone_keys: set[tuple[str, str]] = {
        (str(z.zone_type), str(z.selector))
        for z in page_analysis.functional_zones
        if map_llm_zone_type(z.zone_type) is not None
    }
    if not page_zone_keys or new_transitions_this_page <= 0:
        return
    _status_priority = {
        ExplorationStatus.UNDISCOVERED.value: 0,
        ExplorationStatus.STALE.value: 0,
        ExplorationStatus.DISCOVERED.value: 1,
        ExplorationStatus.PARTIAL.value: 2,
        ExplorationStatus.EXPLORED.value: 3,
        ExplorationStatus.VALIDATED.value: 4,
    }
    target_status = (
        ExplorationStatus.EXPLORED
        if new_transitions_this_page >= 3
        else ExplorationStatus.PARTIAL
    )
    target_priority = _status_priority[target_status.value]
    now_ts = _now_utc()
    page_zone_id_set = {
        _build_runtime_zone_id(
            zone_type=ztype,
            selector=selector,
            source_url=current_page_url,
        )
        for ztype, selector in page_zone_keys
    }
    for z in all_zones:
        if z.id not in page_zone_id_set:
            continue
        current_val = (
            z.exploration_status.value
            if hasattr(z.exploration_status, "value")
            else str(z.exploration_status)
        )
        if _status_priority.get(current_val, 0) < target_priority:
            z.exploration_status = target_status
        z.last_explored = now_ts

    target_status_val = target_status.value
    now_iso = now_ts.isoformat()
    for hint in zones_discovered:
        if (
            str(hint.get("zone_type") or ""),
            str(hint.get("selector") or ""),
        ) not in page_zone_keys:
            continue
        if hint.get("source_url") and hint.get("source_url") != current_page_url:
            continue
        current_val = str(
            hint.get("exploration_status") or ExplorationStatus.DISCOVERED.value
        )
        if _status_priority.get(current_val, 0) < target_priority:
            hint["exploration_status"] = target_status_val
        hint["last_explored"] = now_iso


@observe(
    name="cartography.run_orchestrated_mapping",
    metadata={"component": "cartography", "stage": "pipeline"},
)
async def run_orchestrated_mapping(
    browser: Browser,
    llm: BaseChatModel,
    start_url: str,
    current_url: str,
    app_id: str,
    session_id: str,
    max_steps: int,
    time_budget_ms: int = 600_000,
    warm_start_candidates: list[dict[str, object]] | None = None,
    knowledge_on_demand_enabled: bool | None = None,
    knowledge_min_interval_sec: float | None = None,
    knowledge_trigger_score_threshold: float | None = None,
    knowledge_trigger_profile: str | None = None,
    knowledge_query_timeout_ms: int | None = None,
    knowledge_topk: int | None = None,
    knowledge_release_id: str = "",
    skip_advisor: SkipAdvisor | None = None,
    checkpoint_path: str = "",
    resume_from_checkpoint: bool | None = None,
    orchestration_max_runtime_sec: float | None = None,
) -> "CartographyResult":
    from graph_agent.cartography.react_explorer import ReActExplorer
    from graph_agent.cartography.snapshot import capture_composite_fingerprint
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import ExplorationStatus, State, Transition, Zone

    logger.info("[PIPELINE] === LLM-first orchestrated exploration starting ===")
    all_states: list[State] = []
    all_transitions: list[Transition] = []
    all_zones: list[Zone] = []
    all_history: list[dict[str, object]] = []
    explored_urls: list[str] = []
    menu_items_discovered: list[dict[str, object]] = []
    zones_discovered: list[dict[str, object]] = []
    pages_to_explore: list[tuple[str, str, SkipDecision | None]] = []
    pages_explored: set[str] = set()
    pending_menus: list[dict[str, str]] = []
    menu_scanned_urls: set[str] = set()
    menus_clicked: set[tuple[str, str]] = set()
    primary_origin_url: str = (start_url or current_url or "").strip()
    layout_enabled = resolve_layout_aware_enabled()
    layout_limit = resolve_layout_snapshot_limit()
    layout_conf_retry_enabled = resolve_layout_confidence_retry_enabled()
    layout_conf_threshold = resolve_layout_confidence_threshold()
    layout_evidence: list[LayoutEvidenceItem] = []
    layout_confidence_samples: list[float] = []
    low_layout_confidence_hits = 0
    low_layout_confidence_page_types: Counter[str] = Counter()
    failed_action_count = 0
    semantic_conflict_count = 0
    cross_origin_seen = False
    iframe_seen = False
    captcha_seen = False
    stuck_steps = 0
    knowledge_enabled = (
        resolve_knowledge_on_demand_enabled()
        if knowledge_on_demand_enabled is None
        else knowledge_on_demand_enabled
    )
    knowledge_interval = (
        resolve_knowledge_min_interval_sec()
        if knowledge_min_interval_sec is None
        else max(1.0, knowledge_min_interval_sec)
    )
    knowledge_score_threshold = (
        resolve_knowledge_trigger_score_threshold()
        if knowledge_trigger_score_threshold is None
        else max(0.1, knowledge_trigger_score_threshold)
    )
    knowledge_profile = (
        resolve_knowledge_trigger_profile()
        if knowledge_trigger_profile is None
        else (knowledge_trigger_profile or "balanced").strip().lower()
    )
    knowledge_timeout_ms = (
        resolve_knowledge_query_timeout_ms()
        if knowledge_query_timeout_ms is None
        else max(100, knowledge_query_timeout_ms)
    )
    knowledge_topk_val = (
        resolve_knowledge_topk() if knowledge_topk is None else max(1, knowledge_topk)
    )
    knowledge_broker = KnowledgeBroker() if knowledge_enabled else None
    last_knowledge_query_ts = 0.0
    knowledge_query_count = 0
    knowledge_hit_count = 0
    knowledge_cache_hit_count = 0
    knowledge_timeout_count = 0
    knowledge_circuit_open_count = 0
    knowledge_error_count = 0
    knowledge_latency_total_ms = 0.0

    skip_metrics: dict[str, int] = {
        "skip_page_in_enqueue": 0,
        "skip_page_in_loop": 0,
        "zones_only_in_loop": 0,
        "scheduler_zones_only_forced": 0,
        "error_rate_limited_count": 0,
        "error_unauthorized_count": 0,
        "error_timeout_count": 0,
        "error_unknown_count": 0,
    }
    checkpoint_file = checkpoint_path or resolve_pipeline_checkpoint_path()
    should_resume = (
        resolve_pipeline_resume_from_checkpoint()
        if resume_from_checkpoint is None
        else resume_from_checkpoint
    )
    runtime_limit_sec = (
        resolve_orchestration_max_runtime_sec()
        if orchestration_max_runtime_sec is None
        else max(60.0, orchestration_max_runtime_sec)
    )
    orchestration_step = 0

    def _persist_checkpoint() -> None:
        if not checkpoint_file:
            return
        payload = {
            "app_id": app_id,
            "session_id": session_id,
            "pages_to_explore": [
                {
                    "url": item[0],
                    "reason": item[1],
                    "skip_decision": _serialize_skip_decision(item[2]),
                }
                for item in pages_to_explore
            ],
            "pages_explored": sorted(pages_explored),
            "pending_menus": list(pending_menus),
            "orchestration_step": orchestration_step,
            "failed_action_count": failed_action_count,
            "semantic_conflict_count": semantic_conflict_count,
        }
        try:
            target = Path(checkpoint_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    if should_resume and checkpoint_file:
        try:
            raw = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and str(raw.get("app_id") or "") == app_id
                and str(raw.get("session_id") or "") == session_id
            ):
                restored_queue: list[tuple[str, str, SkipDecision | None]] = []
                for item in list(raw.get("pages_to_explore") or []):
                    if not isinstance(item, dict):
                        continue
                    restored_queue.append(
                        (
                            str(item.get("url") or ""),
                            str(item.get("reason") or "checkpoint-resume"),
                            _deserialize_skip_decision(item.get("skip_decision")),
                        )
                    )
                pages_to_explore = restored_queue
                pages_explored = set(str(x) for x in list(raw.get("pages_explored") or []))
                pending_menus = [
                    dict(m)
                    for m in list(raw.get("pending_menus") or [])
                    if isinstance(m, dict)
                ]
                failed_action_count = int(raw.get("failed_action_count") or 0)
                semantic_conflict_count = int(raw.get("semantic_conflict_count") or 0)
                orchestration_step = int(raw.get("orchestration_step") or 0)
                logger.info(
                    f"[PIPELINE] Resume from checkpoint: queue={len(pages_to_explore)}, explored={len(pages_explored)}"
                )
        except Exception:
            pass

    if not pages_to_explore:
        pages_to_explore = [(current_url or start_url, "start page", None)]

    def _force_zones_only(
        decision: SkipDecision, scheduler_hint: str
    ) -> SkipDecision:
        """Convert any decision into EXPLORE_ZONES_ONLY while preserving
        the long-term learning signals SkipAdvisor returned."""
        return SkipDecision(
            kind=SkipKind.EXPLORE_ZONES_ONLY,
            reason=f"scheduler_override:{scheduler_hint}",
            confidence=decision.confidence,
            target_zone_selectors=list(decision.target_zone_selectors),
            coverage=decision.coverage,
            state_count=decision.state_count,
            last_visited_age_h=decision.last_visited_age_h,
            last_explored_age_h=decision.last_explored_age_h,
            release_coverage=decision.release_coverage,
            intent_confirm_total=decision.intent_confirm_total,
            entity_confirm_total=decision.entity_confirm_total,
            intent_confirmed_zone_count=decision.intent_confirmed_zone_count,
        )

    async def _enqueue_page(
        url: str,
        reason: str,
        *,
        scheduler_hint: str = "",
    ) -> None:
        """``scheduler_hint`` 可选；当 candidate 来自 ExplorationScheduler 的
        ``explore_zone`` / ``stale_re_explore`` 任务时传入，会强制走 zones-only
        以避免对已知页面再做完整 ReAct 循环（仍允许 SkipAdvisor 选择 SKIP_PAGE
        外的所有结果，但 scheduler 总会要求至少补 zone）。"""
        clean = clean_url(url)
        if clean in pages_explored:
            return
        if clean in {clean_url(u) for u, _, _ in pages_to_explore}:
            return
        if primary_origin_url and not same_origin(url, primary_origin_url):
            logger.info("[PIPELINE] Skip foreign-origin URL: %s", url[:80])
            return
        decision: SkipDecision | None = None
        if skip_advisor is not None:
            decision = await skip_advisor.evaluate(url)
            if decision.kind is SkipKind.SKIP_PAGE:
                if scheduler_hint:
                    decision = _force_zones_only(decision, scheduler_hint)
                    skip_metrics["scheduler_zones_only_forced"] += 1
                else:
                    skip_metrics["skip_page_in_enqueue"] += 1
                    logger.info(
                        f"[PIPELINE] SkipAdvisor SKIP_PAGE -> {url[:80]} "
                        f"(coverage={decision.coverage:.2f}, reason={decision.reason})"
                    )
                    return
            elif (
                scheduler_hint in ("stale_re_explore", "explore_zone")
                and decision.kind is SkipKind.FULL_EXPLORE
            ):
                decision = _force_zones_only(decision, scheduler_hint)
                skip_metrics["scheduler_zones_only_forced"] += 1
        pages_to_explore.append((url, reason, decision))
        _persist_checkpoint()

    async def _cleanup_foreign_tabs() -> str:
        nonlocal cross_origin_seen
        if not primary_origin_url:
            return ""
        try:
            tabs = await browser.get_tabs()
        except Exception as e:
            logger.warning("[PIPELINE] get_tabs failed during cleanup: %s", e)
            return ""
        surviving = ""
        for t in tabs:
            tab_url = getattr(t, "url", "") or ""
            if same_origin(tab_url, primary_origin_url):
                surviving = surviving or tab_url
                continue
            target_id = getattr(t, "target_id", None)
            if not target_id:
                continue
            try:
                await browser.close_page(target_id)
                logger.info("[PIPELINE] Closed foreign tab (%s)", tab_url[:60] or "blank")
                cross_origin_seen = True
            except Exception as e:
                logger.warning(
                    "[PIPELINE] Failed to close foreign tab %s: %s", target_id, e
                )
        return surviving

    await _enqueue_ranked_warm_candidates(
        warm_start_candidates=warm_start_candidates or [],
        current_url=current_url,
        start_url=start_url,
        primary_origin_url=primary_origin_url,
        enqueue_page=_enqueue_page,
    )

    max_orchestration_steps = max(50, max_steps)
    loop = asyncio.get_event_loop()
    start_time = loop.time()
    first_page = True

    while (
        pages_to_explore or pending_menus
    ) and orchestration_step < max_orchestration_steps:
        if is_shutdown_requested():
            logger.info("[PIPELINE] Shutdown requested, stopping exploration.")
            break

        pending_menu_task: dict[str, str] | None = None
        skip_decision: SkipDecision | None = None
        if pages_to_explore:
            url, reason, skip_decision = pages_to_explore.pop(0)
            url_clean = clean_url(url)
            if url_clean in pages_explored:
                continue
            # 入队时若 advisor 不可用，pop 时再补一次 evaluate（命中缓存几乎零开销）
            if skip_decision is None and skip_advisor is not None:
                skip_decision = await skip_advisor.evaluate(url)
                if skip_decision.kind is SkipKind.SKIP_PAGE:
                    skip_metrics["skip_page_in_loop"] += 1
                    logger.info(
                        f"[PIPELINE] SkipAdvisor SKIP_PAGE (loop) -> {url[:80]} "
                        f"(coverage={skip_decision.coverage:.2f})"
                    )
                    pages_explored.add(url_clean)
                    continue
            if (
                skip_decision is not None
                and skip_decision.kind is SkipKind.EXPLORE_ZONES_ONLY
            ):
                skip_metrics["zones_only_in_loop"] += 1
        else:
            pending_menu_task = pending_menus.pop(0)
            key = (pending_menu_task["source_url"], pending_menu_task["text"])
            if key in menus_clicked:
                continue
            menus_clicked.add(key)
            url = pending_menu_task["source_url"]
            reason = f"menu-click: {pending_menu_task['text']}"
            url_clean = clean_url(url)

        orchestration_step += 1
        elapsed_ms = (loop.time() - start_time) * 1000
        if elapsed_ms >= runtime_limit_sec * 1000.0:
            logger.info(
                f"[PIPELINE] Runtime budget exhausted ({elapsed_ms/1000:.1f}s >= {runtime_limit_sec:.1f}s), stopping."
            )
            break
        logger.info(
            f"\n[PIPELINE] Step {orchestration_step}/{max_orchestration_steps}: {url[:80]} (reason: {reason}) [queue={len(pages_to_explore)}, pending_menus={len(pending_menus)}]"
        )

        if not await ensure_browser_ready(browser, url):
            logger.warning("[PIPELINE] Browser unrecoverable, skipping %s", url[:80])
            continue

        if first_page and pending_menu_task is None:
            first_page = False
        else:
            first_page = False
            try:
                await browser.navigate_to(url)
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning("[PIPELINE] Navigation failed: %s", e)
                continue

        if pending_menu_task is not None:
            clicked = await click_menu_by_text(browser, pending_menu_task["text"])
            if not clicked:
                logger.info(
                    f"[PIPELINE] Menu click failed for '{pending_menu_task['text']}', skipping"
                )
                failed_action_count += 1
                continue
            await asyncio.sleep(2)

        dom_text = ""
        page_title = ""
        try:
            bs_summary = await browser.get_browser_state_summary(
                include_screenshot=False, include_recent_events=False
            )
            dom_text = bs_summary.dom_state.llm_representation()
            page_title = getattr(bs_summary, "title", "") or ""
        except Exception as e:
            logger.warning("[PIPELINE] Failed to get DOM text: %s", e)

        current_page_url = await browser.get_current_page_url() or url
        if _looks_rate_limited(dom_text, current_page_url):
            skip_metrics["error_rate_limited_count"] += 1
            logger.info(
                f"[PIPELINE] Rate-limit signal detected on {current_page_url[:80]}, cooldown 5s."
            )
            await asyncio.sleep(5)
            await _enqueue_page(current_page_url, "rate-limit-retry")
            pages_explored.add(url_clean)
            _persist_checkpoint()
            continue
        if is_login_url(current_page_url):
            logger.info(
                "[PIPELINE] Login page detected; delegating login (including captcha) to LLM explorer."
            )

        layout_summary = ""
        layout_fingerprint = ""
        layout_confidence = 0.0
        (
            layout_summary,
            layout_fingerprint,
            layout_confidence,
        ) = await collect_layout_context(
            browser,
            enabled=layout_enabled,
            limit=layout_limit,
        )
        if layout_fingerprint:
            if layout_confidence > 0.0:
                layout_confidence_samples.append(layout_confidence)
            layout_evidence.append(
                {
                    "url": current_page_url,
                    "step": orchestration_step,
                    "layout_fingerprint": layout_fingerprint,
                    "layout_summary": layout_summary[:500],
                    "layout_confidence": layout_confidence,
                }
            )
        low_layout_conf = is_low_layout_confidence(
            layout_confidence,
            enabled=layout_conf_retry_enabled,
            threshold=layout_conf_threshold,
        )
        if low_layout_conf:
            low_layout_confidence_hits += 1
            logger.info(
                f"[PIPELINE] Low layout confidence ({layout_confidence:.2f} < {layout_conf_threshold:.2f}); "
                "enabling deeper in-page exploration for this page."
            )

        page_analysis: LLMPageAnalysis = await analyze_page_with_llm(
            llm,
            dom_text,
            current_page_url,
            page_title,
            layout_summary=layout_summary,
        )
        logger.info(
            f"[PIPELINE] LLM analysis: page_type={page_analysis.page_type}, "
            f"menus={len(page_analysis.menu_items)}, zones={len(page_analysis.functional_zones)}"
        )
        if "iframe" in dom_text.lower():
            iframe_seen = True
        if "captcha" in dom_text.lower() or page_analysis.page_type == "login":
            captcha_seen = True
        if low_layout_conf:
            low_layout_confidence_page_types[page_analysis.page_type] += 1

        for m in page_analysis.menu_items:
            menu_items_discovered.append(
                {
                    "text": m.text,
                    "href": m.href,
                    "level": m.level,
                    "source_url": current_page_url,
                }
            )
        for z in page_analysis.functional_zones:
            zones_discovered.append(
                {
                    "zone_type": z.zone_type,
                    "selector": z.selector,
                    "description": z.description,
                    "source_url": current_page_url,
                    "exploration_status": ExplorationStatus.DISCOVERED.value,
                    "last_explored": None,
                }
            )
            mapped_zone_type = map_llm_zone_type(z.zone_type)
            if mapped_zone_type is None:
                continue
            zone_id = _build_runtime_zone_id(
                zone_type=z.zone_type,
                selector=z.selector,
                source_url=current_page_url,
            )
            all_zones.append(
                Zone(
                    id=zone_id,
                    zone_type=mapped_zone_type,
                    root_selector=z.selector,
                    summary=z.description,
                    exploration_status=ExplorationStatus.DISCOVERED,
                )
            )

        enqueued_menu_count = 0
        pending_menu_added = 0
        menu_enqueue_debug: list[dict[str, object]] = []
        already_scanned = clean_url(current_page_url) in menu_scanned_urls
        is_login_context = (
            page_analysis.page_type == "login"
            or bool(page_analysis.is_login_page)
            or is_login_url(current_page_url)
        )
        if not already_scanned:
            menu_scanned_urls.add(clean_url(current_page_url))
            if is_login_context:
                pass
            else:
                for m in page_analysis.menu_items:
                    text = (m.text or "").strip()
                    href = (m.href or "").strip()
                    lower_text = text.lower()
                    if not href and lower_text in {"首页", "home"}:
                        # Runtime evidence shows this often represents current-page nav badge.
                        # Keep queue clean by skipping no-op menu clicks.
                        continue
                    full_href = (
                        urljoin(current_page_url, href)
                        if href and not href.startswith("http")
                        else href
                    )
                    if full_href and is_http_url(full_href):
                        before = len(pages_to_explore)
                        await _enqueue_page(full_href, f"menu: {text}")
                        if len(pages_to_explore) > before:
                            enqueued_menu_count += 1
                            if len(menu_enqueue_debug) < 8:
                                menu_enqueue_debug.append(
                                    {
                                        "type": "url",
                                        "text": text[:80],
                                        "href": full_href[:120],
                                    }
                                )
                    elif text:
                        key = (current_page_url, text)
                        if key in menus_clicked:
                            continue
                        if any(
                            p["source_url"] == current_page_url and p["text"] == text
                            for p in pending_menus
                        ):
                            continue
                        pending_menus.append(
                            {"text": text, "source_url": current_page_url}
                        )
                        pending_menu_added += 1
                        if len(menu_enqueue_debug) < 8:
                            menu_enqueue_debug.append(
                                {"type": "click", "text": text[:80], "href": ""}
                            )

        total_menu_targets = enqueued_menu_count + pending_menu_added
        if (
            total_menu_targets > 0
            and page_analysis.page_type in ("dashboard", "welcome", "login")
            and pending_menu_task is None
            and not low_layout_conf
        ):
            logger.info(
                f"[PIPELINE] Skip in-page exploration for {page_analysis.page_type}; {enqueued_menu_count} url-menu(s), {pending_menu_added} click-menu(s) queued."
            )
            pages_explored.add(clean_url(current_page_url))
            continue

        steps_remaining = max_steps - len(all_history)
        time_remaining = time_budget_ms - elapsed_ms
        if steps_remaining <= 0 or time_remaining <= 0:
            break

        explorer_hint = f"You are exploring the page at {current_page_url}."
        exploration_guidance = build_exploration_guidance(
            page_analysis,
            include_zone_order=not is_login_context,
        )
        if exploration_guidance:
            explorer_hint += "\n\n" + exploration_guidance

        trigger_score = compute_knowledge_trigger_score(
            low_layout_confidence_hits=low_layout_confidence_hits,
            failed_action_count=failed_action_count,
            semantic_conflict_count=semantic_conflict_count,
            stuck_steps=stuck_steps,
            profile=knowledge_profile,
        )
        knowledge_hint_text, knowledge_delta, last_knowledge_query_ts = (
            await _maybe_inject_knowledge_hint(
                knowledge_broker=knowledge_broker,
                knowledge_enabled=knowledge_enabled,
                now_ts=loop.time(),
                last_query_ts=last_knowledge_query_ts,
                min_interval_sec=knowledge_interval,
                score=trigger_score,
                threshold=knowledge_score_threshold,
                all_transitions=all_transitions,
                app_id=app_id,
                session_id=session_id,
                current_page_url=current_page_url,
                page_type=page_analysis.page_type,
                layout_fingerprint=layout_fingerprint,
                knowledge_release_id=knowledge_release_id,
                knowledge_topk_val=knowledge_topk_val,
                knowledge_timeout_ms=knowledge_timeout_ms,
            )
        )
        knowledge_query_count += int(knowledge_delta.get("knowledge_query_count", 0))
        knowledge_hit_count += int(knowledge_delta.get("knowledge_hit_count", 0))
        knowledge_cache_hit_count += int(
            knowledge_delta.get("knowledge_cache_hit_count", 0)
        )
        knowledge_timeout_count += int(knowledge_delta.get("knowledge_timeout_count", 0))
        knowledge_circuit_open_count += int(
            knowledge_delta.get("knowledge_circuit_open_count", 0)
        )
        knowledge_error_count += int(knowledge_delta.get("knowledge_error_count", 0))
        knowledge_latency_total_ms += float(
            knowledge_delta.get("knowledge_latency_ms", 0.0)
        )
        if knowledge_hint_text:
            explorer_hint += "\n\n" + knowledge_hint_text

        page_cap = {
            "dashboard": 20,
            "welcome": 15,
            "login": 10,
            "list": 40,
            "detail": 40,
            "form": 40,
            "settings": 40,
        }.get(page_analysis.page_type, 30)
        if low_layout_conf:
            page_cap = min(60, page_cap + 10)

        # 区域限定模式：缩小 page_cap，附 zone_filter 提示给 explorer
        target_zone_selectors: list[str] = []
        if (
            skip_decision is not None
            and skip_decision.kind is SkipKind.EXPLORE_ZONES_ONLY
            and skip_decision.target_zone_selectors
        ):
            target_zone_selectors = list(skip_decision.target_zone_selectors)
            zones_only_cap = min(15, page_cap)
            logger.info(
                f"[PIPELINE] EXPLORE_ZONES_ONLY -> {url[:80]} "
                f"(pending_zones={len(target_zone_selectors)}, "
                f"page_cap {page_cap}->{zones_only_cap})"
            )
            page_cap = zones_only_cap

        per_page_steps = min(
            steps_remaining,
            page_cap,
        )
        explorer = ReActExplorer(
            max_steps=per_page_steps,
            browser_session=browser,
            extra_system_prompt=build_login_hint_from_env() + "\n\n" + explorer_hint,
            target_zone_selectors=target_zone_selectors or None,
        )
        try:
            transition_count_before = len(all_transitions)
            try:
                page = await browser.get_current_page()
                if page is not None:
                    _ = await capture_composite_fingerprint(
                        page,
                        layout_enabled=layout_enabled,
                        layout_limit=layout_limit,
                    )
            except Exception:
                pass
            explore_result = await explorer.explore_page(
                session=browser,
                state_id=f"state:{current_page_url}",
                page_title=page_title,
            )
            seen_state_ids = {s.id for s in all_states}
            for state in explore_result.states:
                if state.id not in seen_state_ids:
                    all_states.append(state)
                    seen_state_ids.add(state.id)
            all_transitions.extend(explore_result.transitions)
            all_zones.extend(explore_result.zones)
            if explore_result.history:
                all_history.extend(explore_result.history)
            semantic_conflict_count += int(
                getattr(explore_result, "semantic_conflict_count", 0) or 0
            )
            new_transitions_this_page = (
                len(all_transitions) - transition_count_before
            )
            if len(all_transitions) == transition_count_before:
                stuck_steps += 1
            else:
                stuck_steps = 0

            _update_page_zone_progress(
                page_analysis=page_analysis,
                all_zones=all_zones,
                zones_discovered=zones_discovered,
                current_page_url=current_page_url,
                new_transitions_this_page=new_transitions_this_page,
            )

            explored_urls.append(current_page_url)
            pages_explored.add(url_clean)

            harvested = 0
            for state in explore_result.states:
                if getattr(state, "is_external", False):
                    continue
                s_url = (getattr(state, "url", "") or "").strip()
                if not is_http_url(s_url):
                    continue
                if primary_origin_url and not same_origin(s_url, primary_origin_url):
                    continue
                if clean_url(s_url) == url_clean:
                    continue
                before = len(pages_to_explore)
                await _enqueue_page(s_url, "in-page discovery")
                if len(pages_to_explore) > before:
                    harvested += 1
            if harvested:
                logger.info(
                    f"[PIPELINE] Harvested {harvested} same-origin URL(s) from in-page states"
                )

            surviving_url = await _cleanup_foreign_tabs()
            try:
                post_url = await browser.get_current_page_url() or ""
            except Exception:
                post_url = ""
            needs_recovery = (
                not post_url
                or post_url in ("about:blank", "chrome://newtab/")
                or (
                    primary_origin_url and not same_origin(post_url, primary_origin_url)
                )
            )
            if needs_recovery:
                recover_target = (
                    surviving_url or current_page_url or url or primary_origin_url
                )
                try:
                    await browser.navigate_to(recover_target)
                    await asyncio.sleep(1)
                except Exception:
                    pass
            if not await ensure_browser_ready(browser, url):
                break
        except Exception as e:
            logger.warning("[PIPELINE] ReActExplorer failed: %s", e)
            err_code = _classify_pipeline_exception(e)
            if err_code == "rate_limited":
                skip_metrics["error_rate_limited_count"] += 1
            elif err_code == "unauthorized":
                skip_metrics["error_unauthorized_count"] += 1
            elif err_code == "timeout":
                skip_metrics["error_timeout_count"] += 1
            else:
                skip_metrics["error_unknown_count"] += 1
            failed_action_count += 1
            stuck_steps += 1
            pages_explored.add(url_clean)
            await ensure_browser_ready(browser, url)
            _persist_checkpoint()
            continue

        if orchestration_step % 3 == 0 or not pages_to_explore:
            recent_transitions: list[LLMTransitionHint] = [
                {
                    "action": str(t.action),
                    "selector": t.selector,
                    "from_url": t.from_state_id or "",
                    "to_url": t.to_state_id or "",
                }
                for t in all_transitions[-20:]
            ]
            plan = await plan_next_exploration_with_llm(
                llm,
                current_page_url,
                page_title,
                page_analysis,
                explored_urls,
                recent_transitions,
                time_remaining,
                steps_remaining,
            )
            if plan.strategy == "stop":
                if not pages_to_explore and not pending_menus:
                    break
            for task in plan.tasks:
                if task.task_type == "explore_page" and task.target_url:
                    await _enqueue_page(
                        task.target_url, f"llm-plan: {task.description}"
                    )
        _persist_checkpoint()

    if checkpoint_file:
        try:
            Path(checkpoint_file).unlink(missing_ok=True)
        except Exception:
            pass

    return _build_pipeline_result(
        all_states=all_states,
        all_transitions=all_transitions,
        all_zones=all_zones,
        all_history=all_history,
        menu_items_discovered=menu_items_discovered,
        zones_discovered=zones_discovered,
        layout_evidence=layout_evidence,
        layout_confidence_samples=layout_confidence_samples,
        low_layout_confidence_hits=low_layout_confidence_hits,
        low_layout_confidence_page_types=low_layout_confidence_page_types,
        knowledge_query_count=knowledge_query_count,
        knowledge_hit_count=knowledge_hit_count,
        knowledge_cache_hit_count=knowledge_cache_hit_count,
        knowledge_timeout_count=knowledge_timeout_count,
        knowledge_circuit_open_count=knowledge_circuit_open_count,
        knowledge_error_count=knowledge_error_count,
        knowledge_latency_total_ms=knowledge_latency_total_ms,
        knowledge_profile=knowledge_profile,
        skip_advisor=skip_advisor,
        skip_metrics=skip_metrics,
        session_id=session_id,
        current_url=current_url,
        start_url=start_url,
        failed_action_count=failed_action_count,
        semantic_conflict_count=semantic_conflict_count,
        cross_origin_seen=cross_origin_seen,
        iframe_seen=iframe_seen,
        captcha_seen=captcha_seen,
    )

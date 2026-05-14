from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING

from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography.config import (
    clean_url,
    resolve_extra_system_prompt,
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
)
from graph_agent.cartography.layout_snapshot import (
    build_layout_summary,
    capture_layout_snapshot,
    compute_layout_fingerprint,
    estimate_layout_confidence,
)
from graph_agent.cartography.llm_planning import (
    LLMPageAnalysis,
    build_login_hint_from_env,
    normalize_llm_zone_type,
)
from graph_agent.cartography.types import (
    LayoutMetrics,
)

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import ZoneType

logger = logging.getLogger(__name__)

_CAPTCHA_RESULT_CODE_PATTERN = re.compile(r"(CAPTCHA_[A-Z_]+)")


def _empty_captcha_metrics() -> dict[str, int]:
    return {
        "captcha_action_count": 0,
        "captcha_ok_count": 0,
        "captcha_empty_count": 0,
        "captcha_manual_empty_count": 0,
        "captcha_fill_failed_count": 0,
        "captcha_manual_wait_ms_total": 0,
    }


def summarize_captcha_metrics_from_history(
    history: list[dict[str, object]] | None,
) -> dict[str, int]:
    metrics = _empty_captcha_metrics()
    if not history:
        return metrics
    for item in history:
        if not isinstance(item, dict):
            continue
        action_name = str(item.get("action_name") or item.get("action") or "").strip().lower()
        if action_name != "solve_captcha":
            continue
        metrics["captcha_action_count"] += 1
        result_text = str(item.get("action_result") or item.get("result") or "")
        upper = result_text.upper()
        match = _CAPTCHA_RESULT_CODE_PATTERN.search(upper)
        code = match.group(1) if match else "CAPTCHA_UNKNOWN"
        if code.startswith("CAPTCHA_OK"):
            metrics["captcha_ok_count"] += 1
        elif code == "CAPTCHA_EMPTY_CODE":
            metrics["captcha_empty_count"] += 1
        elif code == "CAPTCHA_MANUAL_EMPTY":
            metrics["captcha_manual_empty_count"] += 1
        elif code.startswith("CAPTCHA_FILL_FAILED"):
            metrics["captcha_fill_failed_count"] += 1
        wait_match = re.search(r"wait_ms=(\d+)", result_text, flags=re.IGNORECASE)
        if wait_match:
            metrics["captcha_manual_wait_ms_total"] += int(wait_match.group(1))
    return metrics


def _now_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _parse_evaluate_result(raw: object) -> dict[str, object]:

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




def _to_float(value: object) -> float:
    """Safely convert value to float with explicit type checks."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _to_int(value: object) -> int:
    """Safely convert value to int with explicit type checks."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def rank_warm_start_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """排序优先级（越靠前越先探索）：

    1. ExplorationScheduler 注入的高优先任务（按 ``scheduler_priority`` 降序）
    2. 历史发现但 zone 未探的（``zone_unexplored=True``）
    3. 历史 transition 置信度高
    """

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

    normalized = normalize_llm_zone_type(zone_type)
    mapping: dict[str, "ZoneType"] = {
        "form": ZoneType.DETAIL_FORM,
        "table": ZoneType.DATA_TABLE,
        "nav": ZoneType.TREE_PANEL,
        "action_bar": ZoneType.ACTION_BAR,
        "filter": ZoneType.SEARCH_FORM,
        "modal": ZoneType.MODAL,
        "tabs": ZoneType.TAB_PANEL,
        "pagination": ZoneType.DATA_TABLE,
        "card": ZoneType.DETAIL_FORM,
        "list": ZoneType.DATA_TABLE,
        "chart": ZoneType.DETAIL_FORM,
        "steps": ZoneType.TAB_PANEL,
        "content": ZoneType.DETAIL_FORM,
    }
    return mapping.get(normalized)


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
    return f"zone:{hashlib.md5(seed.encode(), usedforsecurity=False).hexdigest()[:12]}"


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
                confidence=_to_float(item.get("confidence")),
            )
        )
    lines.append("- If hint conflicts with current page evidence, ignore the hint.")
    return "\n".join(lines)


def _build_extra_system_prompt(explorer_hint: str) -> str:
    """Assemble the extra system prompt from env injection and exploration guidance."""
    parts: list[str] = []
    login_hint = build_login_hint_from_env()
    if login_hint:
        parts.append(login_hint)
    custom_hint = resolve_extra_system_prompt()
    if custom_hint:
        parts.append(custom_hint)
    if explorer_hint:
        parts.append(explorer_hint)
    return "\n\n".join(parts)


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
) -> tuple[str, dict[str, int | float], float]:
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
    metrics_delta: dict[str, int | float] = {
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
    low_layout_confidence_page_types: dict[str, int],
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
    captcha_metrics: dict[str, int],
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
    result.layout_metrics.update(captcha_metrics)
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
        captcha_action_count=int(captcha_metrics.get("captcha_action_count", 0)),
        captcha_empty_code_count=int(captcha_metrics.get("captcha_empty_count", 0)),
        captcha_manual_empty_count=int(captcha_metrics.get("captcha_manual_empty_count", 0)),
        captcha_fill_failed_count=int(captcha_metrics.get("captcha_fill_failed_count", 0)),
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



def _force_zones_only(decision, scheduler_hint: str):
    from graph_agent.cartography.skip_advisor import SkipDecision, SkipKind
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

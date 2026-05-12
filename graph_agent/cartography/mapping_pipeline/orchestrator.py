from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from browser_use.browser.session import BrowserSession as Browser
from browser_use.llm.base import BaseChatModel

from graph_agent.cartography.config import (
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
)
from graph_agent.cartography.knowledge_broker import KnowledgeBroker
from graph_agent.cartography.mapping_pipeline.helpers import (
    _deserialize_skip_decision,
    _empty_captcha_metrics,
    _enqueue_ranked_warm_candidates,
)
from graph_agent.cartography.mapping_pipeline.orchestrator_state import (
    OrchestratorState,
    cleanup_foreign_tabs,
    enqueue_page,
    persist_checkpoint,
)
from graph_agent.cartography.mapping_pipeline.orchestrator_loop import _run_main_loop
from graph_agent.cartography.skip_advisor import SkipAdvisor, SkipDecision
from graph_agent.lib.observability import observe

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


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
    inventory: list[dict] | None = None,
    initial_actions_log: list[dict[str, object]] | None = None,
    explorer_hint: str = "",
    warm_start_urls: set[str] | None = None,
    on_page_complete: Callable[..., Any] | None = None,
) -> "CartographyResult":

    logger.info("[PIPELINE] === LLM-first orchestrated exploration starting ===")
    _state = OrchestratorState()
    pages_to_explore = _state.pages_to_explore
    pages_explored = _state.pages_explored
    primary_origin_url: str = (start_url or current_url or "").strip()
    resolve_layout_aware_enabled()
    resolve_layout_snapshot_limit()
    resolve_layout_confidence_retry_enabled()
    resolve_layout_confidence_threshold()
    Counter()
    _empty_captcha_metrics()
    zone_state_map: dict = {}
    zone_rows: list = []
    zone_rows_by_id: dict = {}
    intervention_task_tracker: dict = {}
    _inventory = inventory or []
    _initial_actions_log = initial_actions_log or []
    _explorer_hint = explorer_hint or ""
    _warm_start_urls = warm_start_urls or set()
    knowledge_enabled = (
        resolve_knowledge_on_demand_enabled()
        if knowledge_on_demand_enabled is None
        else knowledge_on_demand_enabled
    )
    (
        resolve_knowledge_min_interval_sec()
        if knowledge_min_interval_sec is None
        else max(1.0, knowledge_min_interval_sec)
    )
    (
        resolve_knowledge_trigger_score_threshold()
        if knowledge_trigger_score_threshold is None
        else max(0.1, knowledge_trigger_score_threshold)
    )
    (
        resolve_knowledge_trigger_profile()
        if knowledge_trigger_profile is None
        else (knowledge_trigger_profile or "balanced").strip().lower()
    )
    (
        resolve_knowledge_query_timeout_ms()
        if knowledge_query_timeout_ms is None
        else max(100, knowledge_query_timeout_ms)
    )
    (
        resolve_knowledge_topk() if knowledge_topk is None else max(1, knowledge_topk)
    )
    KnowledgeBroker() if knowledge_enabled else None

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
    (
        resolve_orchestration_max_runtime_sec()
        if orchestration_max_runtime_sec is None
        else max(60.0, orchestration_max_runtime_sec)
    )

    def _persist_checkpoint() -> None:
        persist_checkpoint(_state, checkpoint_file=checkpoint_file,
                          app_id=app_id, session_id=session_id)

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
                _state.failed_action_count = int(raw.get("failed_action_count") or 0)
                _state.semantic_conflict_count = int(raw.get("semantic_conflict_count") or 0)
                _state.orchestration_step = int(raw.get("orchestration_step") or 0)
                logger.info(
                    f"[PIPELINE] Resume from checkpoint: queue={len(pages_to_explore)}, explored={len(pages_explored)}"
                )
        except Exception:
            pass

    if not pages_to_explore:
        pages_to_explore = [(current_url or start_url, "start page", None)]

    # _force_zones_only imported from helpers

    async def _enqueue_page(url: str, reason: str, *, scheduler_hint: str = "") -> None:
        if await enqueue_page(_state, url=url, reason=reason,
                              scheduler_hint=scheduler_hint,
                              primary_origin_url=primary_origin_url,
                              skip_advisor=skip_advisor,
                              skip_metrics=skip_metrics):
            _persist_checkpoint()

    async def _cleanup_foreign_tabs() -> str:
        return await cleanup_foreign_tabs(_state, browser=browser,
                                          primary_origin_url=primary_origin_url)

    await _enqueue_ranked_warm_candidates(
        warm_start_candidates=warm_start_candidates or [],
        current_url=current_url,
        start_url=start_url,
        primary_origin_url=primary_origin_url,
        enqueue_page=_enqueue_page,
    )

    max(50, max_steps)
    loop = asyncio.get_event_loop()
    loop.time()

    return await _run_main_loop(
        browser=browser,
        llm=llm,
        _state=_state,
        checkpoint_file=checkpoint_file,
        app_id=app_id,
        session_id=session_id,
        start_url=start_url,
        current_url=current_url,
        primary_origin_url=primary_origin_url,
        skip_advisor=skip_advisor,
        skip_metrics=skip_metrics,
        zone_state_map=zone_state_map,
        zone_rows=zone_rows,
        zone_rows_by_id=zone_rows_by_id,
        intervention_task_tracker=intervention_task_tracker,
        max_steps=max_steps,
        orchestration_max_runtime_sec=orchestration_max_runtime_sec or 3600.0,
        inventory=_inventory,
        initial_actions_log=_initial_actions_log,
        explorer_hint=_explorer_hint,
        warm_start_urls=_warm_start_urls,
        on_page_complete=on_page_complete,
        enqueue_page_fn=_enqueue_page,
        persist_checkpoint_fn=_persist_checkpoint,
        cleanup_foreign_tabs_fn=_cleanup_foreign_tabs,
    )

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from graph_agent.cartography.browser_lifecycle import is_shutdown_requested
from graph_agent.cartography.config import (
    clean_url,
    is_http_url,
    is_login_url,
    same_origin,
)
from graph_agent.cartography.mapping_pipeline.helpers import (
    _build_extra_system_prompt,
    _build_pipeline_result,
    _build_runtime_zone_id,
    _classify_pipeline_exception,
    _looks_rate_limited,
    _maybe_inject_knowledge_hint,
    _update_page_zone_progress,
    collect_layout_context,
    compute_knowledge_trigger_score,
    ensure_browser_ready,
    is_low_layout_confidence,
    map_llm_zone_type,
    summarize_captcha_metrics_from_history,
)
from graph_agent.cartography.llm_planning import (
    LLMPageAnalysis,
    LLMTransitionHint,
    analyze_page_with_llm,
    build_exploration_guidance,
    build_login_hint_from_env,
    normalize_page_type,
    plan_next_exploration_with_llm,
)

from graph_agent.cartography.mapping_pipeline.orchestrator_state import (
    OrchestratorState,
)
from graph_agent.cartography.skip_advisor import SkipDecision, SkipKind

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


async def _run_main_loop(
    *,
    browser,
    llm,
    _state: OrchestratorState,
    checkpoint_file: str | None,
    app_id: str,
    session_id: str,
    start_url: str,
    current_url: str,
    primary_origin_url: str | None,
    skip_advisor,
    skip_metrics: dict,
    zone_state_map: dict,
    zone_rows: list,
    zone_rows_by_id: dict,
    intervention_task_tracker: dict,
    orchestration_max_runtime_sec: float,
    inventory: list,
    initial_actions_log: list,
    explorer_hint: str,
    warm_start_urls: set,
    on_page_complete,
    enqueue_page_fn,
    persist_checkpoint_fn,
    cleanup_foreign_tabs_fn,
    persist_page_fn=None,
) -> "CartographyResult":
    from graph_agent.cartography.react_explorer import ReActExplorer
    from graph_agent.cartography.snapshot import capture_composite_fingerprint
    from graph_agent.models import ExplorationStatus, State, Transition, Zone

    pages_to_explore = _state.pages_to_explore
    pages_explored = _state.pages_explored
    # Local state initialized from parent function scope
    all_states: list[State] = []
    all_transitions: list[Transition] = []
    all_zones: list[Zone] = []
    all_history: list[dict[str, object]] = []
    explored_urls: list[str] = []
    menu_items_discovered: list[dict[str, object]] = []
    zones_discovered: list[dict[str, object]] = []
    layout_evidence: list = []
    layout_confidence_samples: list[float] = []
    low_layout_confidence_hits = 0
    low_layout_confidence_page_types: dict[str, int] = {}
    captcha_metrics = {}
    stuck_steps = 0
    loop = asyncio.get_event_loop()
    start_time = loop.time()
    first_page = True
    failed_action_count = _state.failed_action_count
    semantic_conflict_count = _state.semantic_conflict_count
    cross_origin_seen = _state.cross_origin_seen
    runtime_limit_sec = orchestration_max_runtime_sec
    knowledge_query_count = 0
    knowledge_hit_count = 0
    knowledge_cache_hit_count = 0
    knowledge_timeout_count = 0
    knowledge_circuit_open_count = 0
    knowledge_error_count = 0
    knowledge_latency_total_ms = 0.0
    layout_enabled = False
    layout_limit = 10
    layout_conf_retry_enabled = False
    layout_conf_threshold = 0.5
    knowledge_enabled = False
    knowledge_broker = None
    last_knowledge_query_ts = 0.0
    knowledge_interval = 60.0
    knowledge_score_threshold = 0.5
    knowledge_profile = "balanced"
    knowledge_timeout_ms = 15000
    knowledge_topk_val = 5
    knowledge_release_id = ""
    time_budget_ms = 3_600_000
    iframe_seen = False
    captcha_seen = False

    while pages_to_explore:
        if is_shutdown_requested():
            logger.info("[PIPELINE] Shutdown requested, stopping exploration.")
            break

        skip_decision: SkipDecision | None = None
        url, reason, skip_decision = pages_to_explore.pop(0)
        url_clean = clean_url(url)
        if url_clean in pages_explored:
            continue
        if skip_decision is None and skip_advisor is not None:
            skip_decision = await skip_advisor.evaluate(url)
            assert skip_decision is not None
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

        _state.orchestration_step += 1
        elapsed_ms = (loop.time() - start_time) * 1000
        if elapsed_ms >= runtime_limit_sec * 1000.0:
            logger.info(
                f"[PIPELINE] Runtime budget exhausted ({elapsed_ms/1000:.1f}s >= {runtime_limit_sec:.1f}s), stopping."
            )
            break
        logger.info(
            f"\n[PIPELINE] Step {_state.orchestration_step}: {url[:80]} (reason: {reason}) [queue={len(pages_to_explore)}]"
        )

        if not await ensure_browser_ready(browser, url):
            logger.warning("[PIPELINE] Browser unrecoverable, skipping %s", url[:80])
            continue

        if not first_page:
            try:
                await browser.navigate_to(url)
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning("[PIPELINE] Navigation failed: %s", e)
                continue

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
            await enqueue_page_fn(current_page_url, "rate-limit-retry")
            pages_explored.add(url_clean)
            persist_checkpoint_fn()
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
                    "step": _state.orchestration_step,
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
        page_analysis.page_type = normalize_page_type(page_analysis.page_type)
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

        is_login_context = (
            page_analysis.page_type == "login"
            or bool(page_analysis.is_login_page)
            or is_login_url(current_page_url)
        )

        time_remaining = time_budget_ms - elapsed_ms
        if time_remaining <= 0:
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
        knowledge_hint_text: str
        knowledge_delta: dict[str, int | float]
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

        # Per-page ReAct step budget — generous fixed cap.
        # The only global limiting factor is the time budget (time_budget_ms).
        per_page_steps = 60
        if low_layout_conf:
            per_page_steps = min(80, per_page_steps + 10)

        # 区域限定模式：缩小 budget，附 zone_filter 提示给 explorer
        target_zone_selectors: list[str] = []
        if (
            skip_decision is not None
            and skip_decision.kind is SkipKind.EXPLORE_ZONES_ONLY
            and skip_decision.target_zone_selectors
        ):
            target_zone_selectors = list(skip_decision.target_zone_selectors)
            zones_only_cap = min(20, per_page_steps)
            logger.info(
                f"[PIPELINE] EXPLORE_ZONES_ONLY -> {url[:80]} "
                f"(pending_zones={len(target_zone_selectors)}, "
                f"budget {per_page_steps}->{zones_only_cap})"
            )
            per_page_steps = zones_only_cap
        explorer = ReActExplorer(
            max_steps=per_page_steps,
            browser_session=browser,
            extra_system_prompt=_build_extra_system_prompt(explorer_hint) + "\n\n" + build_login_hint_from_env(),
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
                page_captcha_metrics = summarize_captcha_metrics_from_history(
                    explore_result.history
                )
                for key, value in page_captcha_metrics.items():
                    captcha_metrics[key] = int(captcha_metrics.get(key, 0)) + int(value)
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

            # Per-page flush to Neo4j for incremental durability.
            if persist_page_fn is not None:
                try:
                    await persist_page_fn(explore_result)
                except Exception as e:
                    logger.warning("[PIPELINE] Per-page persist failed: %s", e)

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
                await enqueue_page_fn(s_url, "in-page discovery")
                if len(pages_to_explore) > before:
                    harvested += 1
            if harvested:
                logger.info(
                    f"[PIPELINE] Harvested {harvested} same-origin URL(s) from in-page states"
                )

            surviving_url = await cleanup_foreign_tabs_fn()
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
            _state.failed_action_count += 1
            stuck_steps += 1
            pages_explored.add(url_clean)
            await ensure_browser_ready(browser, url)
            persist_checkpoint_fn()
            continue

        if _state.orchestration_step % 3 == 0 or not pages_to_explore:
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
                len(pages_to_explore),
            )
            if plan.strategy == "stop":
                if not pages_to_explore:
                    break
            for task in plan.tasks:
                if task.task_type == "explore_page" and task.target_url:
                    await enqueue_page_fn(
                        task.target_url, f"llm-plan: {task.description}"
                    )
        persist_checkpoint_fn()

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
        captcha_metrics=captcha_metrics,
    )

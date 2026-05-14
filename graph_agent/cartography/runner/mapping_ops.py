"""Run browser-use Agent for mapping: explore flow and save to Neo4j."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography import browser_lifecycle
from graph_agent.cartography import config as cartography_config
from graph_agent.cartography.captcha import resolve_captcha_solve_mode
from graph_agent.cartography.manual_capture import ManualCaptureSession
from graph_agent.cartography.mapping_pipeline import (
    rank_warm_start_candidates as orch_rank_warm_start_candidates,
)
from graph_agent.cartography.mapping_pipeline import (
    run_orchestrated_mapping,
)
from graph_agent.cartography.skip_advisor import SkipAdvisor, SkipPolicy
from graph_agent.cartography.types import (
    EvidenceBundleItem,
)
from graph_agent.cartography.runtime_watchdog import RuntimeWatchdogConfig
from graph_agent.lib.observability import initialize_laminar, observe
from graph_agent.llm import get_llm
from graph_agent.neo4j_client.manager import GraphManager

logger = logging.getLogger(__name__)

# Global registry for active browser sessions (for cleanup on Ctrl+C)
_active_browsers: list[Browser] = []
_shutdown_requested = False


def _register_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.register_browser(browser)


def _unregister_browser(browser: Browser) -> None:
    """Compatibility wrapper."""
    browser_lifecycle.unregister_browser(browser)


async def _cleanup_all_browsers() -> None:
    """Compatibility wrapper."""
    await browser_lifecycle.cleanup_all_browsers()


# Register signal handlers
browser_lifecycle.install_signal_handlers()


@asynccontextmanager
async def managed_browser(browser: Browser):
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
                await browser.close()  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("[RUNNER] Browser cleanup error: %s", e)
        finally:
            _unregister_browser(browser)


def _resolve_mapping_url(url: str | None) -> str:
    return cartography_config.resolve_mapping_url(url)


def _resolve_mapping_headless() -> bool:
    return cartography_config.resolve_mapping_headless()


def _resolve_mapping_channel() -> str | None:
    return cartography_config.resolve_mapping_channel()


def _setup_browser_use_timeouts():
    cartography_config.setup_browser_use_timeouts()


def _resolve_knowledge_on_demand_enabled() -> bool:
    return cartography_config.resolve_knowledge_on_demand_enabled()


def _resolve_knowledge_min_interval_sec() -> float:
    return cartography_config.resolve_knowledge_min_interval_sec()


def _resolve_knowledge_trigger_score_threshold() -> float:
    return cartography_config.resolve_knowledge_trigger_score_threshold()


def _resolve_knowledge_trigger_profile() -> str:
    return cartography_config.resolve_knowledge_trigger_profile()


def _resolve_knowledge_query_timeout_ms() -> int:
    return cartography_config.resolve_knowledge_query_timeout_ms()


def _resolve_knowledge_topk() -> int:
    return cartography_config.resolve_knowledge_topk()


def _resolve_knowledge_release_id() -> str:
    return cartography_config.resolve_knowledge_release_id()


def _resolve_orchestration_max_runtime_sec() -> float:
    return cartography_config.resolve_orchestration_max_runtime_sec()


def _resolve_pipeline_checkpoint_path() -> str:
    return cartography_config.resolve_pipeline_checkpoint_path()


def _resolve_pipeline_resume_from_checkpoint() -> bool:
    return cartography_config.resolve_pipeline_resume_from_checkpoint()


def _load_inventory(inventory_path: str | Path) -> list[dict[str, object]]:
    return cartography_config.load_inventory(inventory_path)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _classify_runner_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or ("rate" in text and "limit" in text):
        return "rate_limited"
    if "401" in text or "403" in text or "forbidden" in text or "unauthorized" in text:
        return "unauthorized"
    if "timeout" in text:
        return "timeout"
    return "unknown"


def _normalize_app_name(app_name: str) -> str:
    value = (app_name or "").strip().lower()
    if not value:
        return "unknown"
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "unknown"


def _stable_app_id_from_app_name(app_name: str) -> str:
    return f"app:{_normalize_app_name(app_name)}"


def rank_warm_start_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    return orch_rank_warm_start_candidates(candidates)


@observe(
    name="cartography.run_mapping",
    metadata={"component": "cartography", "stage": "entry"},
)
async def run_mapping(
    url: str | None = None,
    output_path: str | None = None,  # Kept for API compatibility, ignored
    task: str | None = None,
    inventory_path: str | Path | None = None,
    merge_existing: bool = False,  # Kept for API compatibility, ignored
) -> str:
    """Run LLM-first orchestrated exploration and store results in Neo4j.

    - url: Start URL (required via arg or MAPPING_URL env).
    - output_path: Kept for API compatibility, data now stored in Neo4j.
    - task: Kept for API compatibility, no longer used by the pipeline.
    - inventory_path: Required. Path to scout inventory JSON (run scout first).

    Returns the app_id of the newly created app in Neo4j.
    """
    if inventory_path is None or not str(inventory_path).strip():
        raise ValueError(
            "inventory_path is required. Run scout first, then pass --inventory to mapping."
        )
    initialize_laminar()
    _inventory = _load_inventory(inventory_path)

    from browser_use import Browser

    resolved_url = _resolve_mapping_url(url)
    pkg_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        # Resolve default relative to package: graph_agent/data/graph.json
        output_path = str(pkg_root / "data" / "graph.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    channel = _resolve_mapping_channel()
    base_headless = _resolve_mapping_headless()
    base_args = ["--incognito"]  # Force incognito mode: no session cache
    if channel:
        try:
            browser = Browser(headless=base_headless, args=base_args, channel=channel)
        except TypeError:
            # Older browser-use versions may not support channel keyword.
            browser = Browser(headless=base_headless, args=base_args)
    else:
        browser = Browser(headless=base_headless, args=base_args)

    async with managed_browser(browser):
        llm = get_llm()
        logger.info("[RUNNER] === Pre-navigation phase starting ===")
        _initial_actions_log: list[dict[str, object]] = []

        # Navigate to the target URL and wait for the page to settle.
        # Login (including captcha) is handled entirely by the LLM explorer
        # via prompt injection (build_login_hint_from_env).
        current_url = resolved_url
        logger.info("[PreNav] Starting browser and navigating to %s...", resolved_url)
        try:
            await browser.start()
            await browser.navigate_to(resolved_url)
            current_url = await browser.get_current_page_url() or resolved_url
            # Give the page time to load / redirect.
            await asyncio.sleep(4)
            current_url = await browser.get_current_page_url() or current_url
        except Exception as e:
            logger.warning(
                "[RUNNER] Pre-navigation failed: %s. Agent will attempt recovery.", e
            )

        logger.info("[PreNav] Browser is now at %s.", current_url)

        # Generate app_id and session_id early (needed by the pipeline and Neo4j)
        from urllib.parse import urlparse

        parsed = urlparse(resolved_url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        app_name = domain or "unknown"
        app_id = _stable_app_id_from_app_name(app_name)
        session_id = f"session:{app_id}:{datetime.now(timezone.utc).isoformat()}"
        warm_start_candidates: list[dict[str, object]] = []
        scheduler_candidates: list[dict[str, object]] = []
        try:
            async with GraphManager() as warm_manager:
                warm_start_candidates = await warm_manager.get_runner_warm_start_candidates(
                    app_id=app_id,
                    app_name=app_name,
                )
        except Exception as e:
            logger.warning("[RUNNER] Warm-start candidate query skipped: %s", e)

        # ExplorationScheduler —— 把"未探页面 / 未探 zone / stale zone"组成的优先级队列
        # 转换为 warm_start candidate，让本次 mapping 优先去补这些"已知缺口"。
        # 任务中带 ``scheduler_priority``、``scheduler_reason``、``scheduler_task_type``，
        # mapping_pipeline 据此把 stale 任务标记为 ``EXPLORE_ZONES_ONLY``。
        if cartography_config.resolve_scheduler_warm_start_enabled():
            try:
                from graph_agent.coverage.scheduler import ExplorationScheduler, ScheduledTask

                scheduler = ExplorationScheduler()
                async with GraphManager() as sched_manager:
                    scheduler_tasks: list[ScheduledTask] = await scheduler.schedule(
                        sched_manager.get_driver(),
                        focus="breadth",
                        max_tasks=cartography_config.resolve_scheduler_warm_start_topk(),
                        app_id=app_id,
                    )
                    seen_urls: set[str] = set()
                    for sched_task in scheduler_tasks:
                        ctx = sched_task.context or {}
                        target_url = str(ctx.get("url") or "").strip()
                        if not target_url and sched_task.type == "explore_zone":
                            sid = str(ctx.get("state_id") or "")
                            if sid:
                                target_url = (
                                    await sched_manager.get_state_url(sid)
                                ).strip()
                        if not target_url or target_url in seen_urls:
                            continue
                        seen_urls.add(target_url)
                        scheduler_candidates.append(
                            {
                                "transition_id": f"sched:{sched_task.target_id}",
                                "confidence": min(1.0, sched_task.priority / 100.0),
                                "target_url": target_url,
                                "zone_unexplored": sched_task.type == "discover_page",
                                "scheduler_priority": sched_task.priority,
                                "scheduler_reason": str(ctx.get("reason") or sched_task.type),
                                "scheduler_task_type": sched_task.type,
                            }
                        )
                if scheduler_candidates:
                    logger.info(
                        f"[RUNNER] ExplorationScheduler injected {len(scheduler_candidates)} candidate(s)"
                    )
            except Exception as e:
                logger.warning("[RUNNER] scheduler warm-start skipped: %s", e)
        # 调度任务排在静态 warm-start 之前（discover_page 优先级最高）
        if scheduler_candidates:
            warm_start_candidates = scheduler_candidates + (warm_start_candidates or [])

        knowledge_on_demand_enabled = _resolve_knowledge_on_demand_enabled()
        knowledge_min_interval_sec = _resolve_knowledge_min_interval_sec()
        knowledge_trigger_score_threshold = _resolve_knowledge_trigger_score_threshold()
        knowledge_trigger_profile = _resolve_knowledge_trigger_profile()
        knowledge_query_timeout_ms = _resolve_knowledge_query_timeout_ms()
        knowledge_topk = _resolve_knowledge_topk()
        knowledge_release_id = _resolve_knowledge_release_id()
        captcha_mode = resolve_captcha_solve_mode()
        guard_cfg = RuntimeWatchdogConfig.from_env()
        logger.info(
            "[RUNNER] config_snapshot captcha_mode=%s guard_captcha_attempts=%d guard_captcha_failures=%d",
            captcha_mode,
            guard_cfg.max_captcha_attempts,
            guard_cfg.max_captcha_failures,
        )
        # 若没有显式设置 MAPPING_RELEASE_ID，则尝试拉取该 app 当前最新的 active release，
        # 让 KnowledgeBroker 优先走 release-first 路径（学习沉淀的最权威基线）。
        if not knowledge_release_id and app_id:
            try:
                async with GraphManager() as _km:
                    knowledge_release_id = (
                        await _km.get_latest_active_release_by_app_id(app_id)
                    )
                if knowledge_release_id:
                    logger.info(
                        f"[RUNNER] auto-resolved active release_id={knowledge_release_id}"
                    )
            except Exception as e:
                logger.warning("[RUNNER] release_id auto-resolve skipped: %s", e)

        # SkipAdvisor — 跨 session 跳过已探索区域；失败/超时一律退化为 FULL_EXPLORE
        skip_advisor: SkipAdvisor | None = None
        if cartography_config.resolve_skip_advisor_enabled():
            policy = SkipPolicy.from_profile(
                cartography_config.resolve_skip_policy_profile()
            )
            ttl_override = cartography_config.resolve_skip_ttl_hours()
            if ttl_override is not None:
                policy.ttl_hours = ttl_override
            policy.query_timeout_ms = (
                cartography_config.resolve_skip_query_timeout_ms()
            )
            policy.cache_ttl_sec = cartography_config.resolve_skip_cache_ttl_sec()
            skip_advisor = SkipAdvisor(
                app_name=app_name,
                app_id=app_id,
                primary_origin_url=resolved_url,
                policy=policy,
            )
            logger.info(
                f"[RUNNER] SkipAdvisor enabled (profile={policy.profile}, "
                f"ttl_h={policy.ttl_hours}, timeout_ms={policy.query_timeout_ms}, "
                f"cache_s={policy.cache_ttl_sec})"
            )

        # Check for shutdown request before running
        if browser_lifecycle.is_shutdown_requested():
            logger.info("[RUNNER] Shutdown requested before agent run, cleaning up...")
            return ""

        logger.info("[RUNNER] === LLM-first orchestrated mode ===")
        retry_count = 0
        while True:
            try:
                _result = await run_orchestrated_mapping(
                    browser=browser,
                    llm=llm,
                    start_url=resolved_url,
                    current_url=current_url or resolved_url,
                    app_id=app_id,
                    app_name=app_name,
                    session_id=session_id,
                    warm_start_candidates=warm_start_candidates,
                    knowledge_on_demand_enabled=knowledge_on_demand_enabled,
                    knowledge_min_interval_sec=knowledge_min_interval_sec,
                    knowledge_trigger_score_threshold=knowledge_trigger_score_threshold,
                    knowledge_trigger_profile=knowledge_trigger_profile,
                    knowledge_query_timeout_ms=knowledge_query_timeout_ms,
                    knowledge_topk=knowledge_topk,
                    knowledge_release_id=knowledge_release_id,
                    skip_advisor=skip_advisor,
                    checkpoint_path=_resolve_pipeline_checkpoint_path(),
                    resume_from_checkpoint=_resolve_pipeline_resume_from_checkpoint(),
                    orchestration_max_runtime_sec=_resolve_orchestration_max_runtime_sec(),
                )
                break
            except asyncio.CancelledError:
                logger.info("[RUNNER] Orchestrated exploration cancelled, cleaning up...")
                raise
            except Exception as e:
                err_code = _classify_runner_exception(e)
                if err_code in {"rate_limited", "unauthorized"} and retry_count < 1:
                    retry_count += 1
                    cooldown = 5 if err_code == "rate_limited" else 3
                    logger.warning(
                        f"[RUNNER] transient pipeline error ({err_code}), retry after {cooldown}s: {e}"
                    )
                    await asyncio.sleep(cooldown)
                    continue
                raise

    # Persistence is done incrementally inside run_orchestrated_mapping
    # (per-page flush via PersistenceSession), so no final persist call here.
    return app_id


async def run_manual_mapping(
    *,
    app_name: str,
    session_id: str,
    events: list[dict[str, object]],
    mode: str = "manual_raw",
    app_id: str | None = None,
    operator_id: str = "human:operator",
    start_state_hint: str = "",
    start_url: str = "",
) -> str:
    """Persist manual capture events through the same cartography pipeline."""
    explicit_app_id = (app_id or "").strip()
    resolved_app_id = explicit_app_id or _stable_app_id_from_app_name(app_name)
    if mode == "manual_graph_assisted":
        manual_session = ManualCaptureSession.from_graph_context(
            session_id=session_id,
            operator_id=operator_id,
            app_id=resolved_app_id,
            start_state_hint=start_state_hint or start_url,
            start_url=start_url,
        )
    else:
        manual_session = ManualCaptureSession.from_raw(
            session_id=session_id,
            operator_id=operator_id,
            start_url=start_url,
        )
    for event in events:
        raw_evidence = event.get("evidence_bundle")
        evidence_bundle = raw_evidence if isinstance(raw_evidence, list) else []
        manual_session.record_action(
            action=str(event.get("action") or "unknown"),
            selector=str(event.get("selector") or "[manual]"),
            url_before=str(event.get("url_before") or start_url),
            url_after=str(event.get("url_after") or start_url),
            thought=str(event.get("thought") or ""),
            action_value=str(event.get("action_value") or ""),
            param_name=str(event.get("param_name") or ""),
            confidence_hint=_as_float(event.get("confidence_hint") or 0.9, 0.9),
            evidence_bundle=cast(list[EvidenceBundleItem], evidence_bundle),
            from_state_hint=str(event.get("from_state_hint") or ""),
            to_state_hint=str(event.get("to_state_hint") or ""),
        )
    result = await manual_session.to_cartography_result()

    from graph_agent.cartography.persistence import PersistenceSession
    persister = PersistenceSession(
        app_id=resolved_app_id,
        app_name=app_name,
        session_id=session_id,
        resolved_url=start_url,
        current_url=start_url,
        inventory=[],
        initial_actions_log=[],
        mode=mode,
    )
    await persister.init()
    await persister.persist_page(result)
    await persister.finalize(result)
    return resolved_app_id


async def run_post_mapping_maintenance(
    *,
    app_id: str,
    retention_execute: bool = False,
    keep_releases: int = 5,
    min_age_days: int = 14,
    batch_size: int = 500,
) -> dict[str, object]:
    async with GraphManager() as manager:
        pre_gate = await manager.evaluate_semantic_stability_gate(app_id=app_id)
        if retention_execute:
            retention_report = await manager.run_retention(
                app_id=app_id,
                keep_releases=keep_releases,
                min_age_days=min_age_days,
                batch_size=batch_size,
            )
        else:
            retention_report = await manager.evaluate_retention_plan(
                app_id=app_id,
                keep_releases=keep_releases,
                min_age_days=min_age_days,
            )
        post_gate = await manager.evaluate_semantic_stability_gate(app_id=app_id)
    return {
        "app_id": app_id,
        "retention_execute": retention_execute,
        "pre_gate": pre_gate,
        "retention": retention_report,
        "post_gate": post_gate,
    }



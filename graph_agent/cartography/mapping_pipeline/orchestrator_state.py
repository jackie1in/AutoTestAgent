"""Mutable state holder for run_orchestrated_mapping closures."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from graph_agent.cartography.mapping_pipeline.helpers import _serialize_skip_decision

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    pages_to_explore: list = field(default_factory=list)
    pages_explored: set = field(default_factory=set)
    orchestration_step: int = 0
    failed_action_count: int = 0
    semantic_conflict_count: int = 0
    cross_origin_seen: bool = False


def persist_checkpoint(state: OrchestratorState, *, checkpoint_file: str | None,
                       app_id: str, session_id: str) -> None:
    if not checkpoint_file:
        return
    payload = {
        "app_id": app_id,
        "session_id": session_id,
        "pages_to_explore": [
            {"url": item[0], "reason": item[1],
             "skip_decision": _serialize_skip_decision(item[2])}
            for item in state.pages_to_explore
        ],
        "pages_explored": sorted(state.pages_explored),
        "orchestration_step": state.orchestration_step,
        "failed_action_count": state.failed_action_count,
        "semantic_conflict_count": state.semantic_conflict_count,
    }
    try:
        target = Path(checkpoint_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


async def enqueue_page(state: OrchestratorState, *, url: str, reason: str,
                       scheduler_hint: str = "", primary_origin_url: str | None = None,
                       skip_advisor=None, skip_metrics: dict | None = None) -> bool:
    """Returns True if page was enqueued, False if skipped."""
    from graph_agent.cartography.config import clean_url, same_origin
    from graph_agent.cartography.mapping_pipeline.helpers import _force_zones_only
    from graph_agent.cartography.skip_advisor import SkipKind

    clean = clean_url(url)
    if clean in state.pages_explored:
        return False
    if clean in {clean_url(u) for u, _, _ in state.pages_to_explore}:
        return False
    if primary_origin_url and not same_origin(url, primary_origin_url):
        logger.info("[PIPELINE] Skip foreign-origin URL: %s", url[:80])
        return False
    decision = None
    if skip_advisor is not None:
        decision = await skip_advisor.evaluate(url)
        if decision.kind is SkipKind.SKIP_PAGE:
            if scheduler_hint:
                decision = _force_zones_only(decision, scheduler_hint)
                if skip_metrics is not None:
                    skip_metrics["scheduler_zones_only_forced"] += 1
            else:
                if skip_metrics is not None:
                    skip_metrics["skip_page_in_enqueue"] += 1
                logger.info(
                    f"[PIPELINE] SkipAdvisor SKIP_PAGE -> {url[:80]} "
                    f"(coverage={decision.coverage:.2f}, reason={decision.reason})"
                )
                return False
        elif (scheduler_hint in ("stale_re_explore", "explore_zone")
              and decision.kind is SkipKind.FULL_EXPLORE):
            decision = _force_zones_only(decision, scheduler_hint)
            if skip_metrics is not None:
                skip_metrics["scheduler_zones_only_forced"] += 1
    state.pages_to_explore.append((url, reason, decision))
    return True


async def cleanup_foreign_tabs(state: OrchestratorState, *, browser,
                                primary_origin_url: str | None) -> str:
    if not primary_origin_url:
        return ""
    try:
        tabs = await browser.get_tabs()
    except Exception as e:
        logger.warning("[PIPELINE] get_tabs failed during cleanup: %s", e)
        return ""
    surviving = ""
    from graph_agent.cartography.config import same_origin
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
            state.cross_origin_seen = True
        except Exception as e:
            logger.warning("[PIPELINE] Failed to close foreign tab %s: %s", target_id, e)
    return surviving

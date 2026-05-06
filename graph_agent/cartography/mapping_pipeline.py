from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
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
    same_origin,
)
from graph_agent.cartography.intervention_queue import evaluate_intervention_need
from graph_agent.cartography.knowledge_broker import (
    KnowledgeBroker,
    KnowledgeQueryInput,
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
            print(
                f"[PIPELINE] Menu '{target}' clicked "
                f"(via {parsed.get('selector', '?')}, tag={parsed.get('tag', '?')})"
            )
            return True
        print(f"[PIPELINE] Menu '{target}' not found on current page")
        return False
    except Exception as e:
        print(f"[PIPELINE] click_menu_by_text error: {e}")
        return False


def rank_warm_start_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    def _to_float(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    return sorted(
        candidates,
        key=lambda item: (
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


async def ensure_browser_ready(browser: Browser, target_url: str) -> bool:
    try:
        current = await browser.get_current_page_url()
        if current:
            return True
    except Exception:
        pass
    try:
        print("[PIPELINE] Browser session appears reset, attempting restart...")
        await browser.start()
        await browser.navigate_to(target_url)
        await asyncio.sleep(2)
        return True
    except Exception as e:
        print(f"[PIPELINE] Browser restart failed: {e}")
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
) -> "CartographyResult":
    from graph_agent.cartography.react_explorer import ReActExplorer
    from graph_agent.cartography.snapshot import capture_composite_fingerprint
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.models import State, Transition, Zone

    print("[PIPELINE] === LLM-first orchestrated exploration starting ===")
    all_states: list[State] = []
    all_transitions: list[Transition] = []
    all_zones: list[Zone] = []
    all_history: list[dict[str, object]] = []
    explored_urls: list[str] = []
    menu_items_discovered: list[dict[str, object]] = []
    zones_discovered: list[dict[str, object]] = []
    pages_to_explore: list[tuple[str, str]] = [(current_url or start_url, "start page")]
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

    def _enqueue_page(url: str, reason: str) -> None:
        clean = clean_url(url)
        if clean in pages_explored:
            return
        if clean in {clean_url(u) for u, _ in pages_to_explore}:
            return
        if primary_origin_url and not same_origin(url, primary_origin_url):
            print(f"[PIPELINE] Skip foreign-origin URL: {url[:80]}")
            return
        pages_to_explore.append((url, reason))

    async def _cleanup_foreign_tabs() -> str:
        nonlocal cross_origin_seen
        if not primary_origin_url:
            return ""
        try:
            tabs = await browser.get_tabs()
        except Exception as e:
            print(f"[PIPELINE] get_tabs failed during cleanup: {e}")
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
                print(f"[PIPELINE] Closed foreign tab ({tab_url[:60] or 'blank'})")
                cross_origin_seen = True
            except Exception as e:
                print(f"[PIPELINE] Failed to close foreign tab {target_id}: {e}")
        return surviving

    warm_candidates = rank_warm_start_candidates(warm_start_candidates or [])
    for item in warm_candidates:
        target_url = str(item.get("target_url") or "").strip()
        if not target_url:
            continue
        if clean_url(target_url) == clean_url(current_url or start_url):
            continue
        if primary_origin_url and not same_origin(target_url, primary_origin_url):
            continue
        pages_to_explore.append((target_url, "warm-start"))

    orchestration_step = 0
    max_orchestration_steps = 50
    loop = asyncio.get_event_loop()
    start_time = loop.time()
    first_page = True

    while (
        pages_to_explore or pending_menus
    ) and orchestration_step < max_orchestration_steps:
        if is_shutdown_requested():
            print("[PIPELINE] Shutdown requested, stopping exploration.")
            break

        pending_menu_task: dict[str, str] | None = None
        if pages_to_explore:
            url, reason = pages_to_explore.pop(0)
            url_clean = clean_url(url)
            if url_clean in pages_explored:
                continue
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
        print(
            f"\n[PIPELINE] Step {orchestration_step}/{max_orchestration_steps}: {url[:80]} (reason: {reason}) [queue={len(pages_to_explore)}, pending_menus={len(pending_menus)}]"
        )

        if not await ensure_browser_ready(browser, url):
            print(f"[PIPELINE] Browser unrecoverable, skipping {url[:80]}")
            continue

        if first_page and pending_menu_task is None:
            first_page = False
        else:
            first_page = False
            try:
                await browser.navigate_to(url)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"[PIPELINE] Navigation failed: {e}")
                continue

        if pending_menu_task is not None:
            clicked = await click_menu_by_text(browser, pending_menu_task["text"])
            if not clicked:
                print(
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
            print(f"[PIPELINE] Failed to get DOM text: {e}")

        current_page_url = await browser.get_current_page_url() or url
        if is_login_url(current_page_url):
            print(
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
            print(
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
        print(
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
                }
            )
            mapped_zone_type = map_llm_zone_type(z.zone_type)
            if mapped_zone_type is None:
                continue
            zone_id = (
                f"zone:{z.zone_type}:{hashlib.md5(z.selector.encode()).hexdigest()[:8]}"
            )
            all_zones.append(
                Zone(
                    id=zone_id,
                    zone_type=mapped_zone_type,
                    root_selector=z.selector,
                    summary=z.description,
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
                        _enqueue_page(full_href, f"menu: {text}")
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
            print(
                f"[PIPELINE] Skip in-page exploration for {page_analysis.page_type}; {enqueued_menu_count} url-menu(s), {pending_menu_added} click-menu(s) queued."
            )
            pages_explored.add(clean_url(current_page_url))
            continue

        steps_remaining = max_steps - len(all_history)
        time_remaining = time_budget_ms - elapsed_ms
        if steps_remaining <= 0 or time_remaining <= 0:
            break

        explorer_hint = f"You are exploring the page at {current_page_url}."
        exploration_guidance = build_exploration_guidance(page_analysis)
        if exploration_guidance:
            explorer_hint += "\n\n" + exploration_guidance

        trigger_score = compute_knowledge_trigger_score(
            low_layout_confidence_hits=low_layout_confidence_hits,
            failed_action_count=failed_action_count,
            semantic_conflict_count=semantic_conflict_count,
            stuck_steps=stuck_steps,
            profile=knowledge_profile,
        )
        now_ts = loop.time()
        if knowledge_broker and should_query_knowledge(
            enabled=knowledge_enabled,
            now_ts=now_ts,
            last_query_ts=last_knowledge_query_ts,
            min_interval_sec=knowledge_interval,
            score=trigger_score,
            threshold=knowledge_score_threshold,
        ):
            latest_transition = all_transitions[-1] if all_transitions else None
            recent_selector = ""
            recent_action = ""
            if latest_transition is not None:
                recent_selector = str(getattr(latest_transition, "selector", "") or "")
                recent_action = str(getattr(latest_transition, "action", "") or "")
            knowledge_query_count += 1
            knowledge_result = await knowledge_broker.query(
                KnowledgeQueryInput(
                    app_id=app_id,
                    session_id=session_id,
                    current_url=current_page_url,
                    page_type=page_analysis.page_type,
                    layout_fingerprint=layout_fingerprint,
                    recent_selector=recent_selector,
                    recent_action=recent_action,
                    release_id=knowledge_release_id,
                    signals={"trigger_score": trigger_score},
                    top_k=knowledge_topk_val,
                ),
                timeout_ms=knowledge_timeout_ms,
            )
            last_knowledge_query_ts = now_ts
            knowledge_latency_total_ms += knowledge_result.meta.query_latency_ms
            if knowledge_result.meta.cache_hit:
                knowledge_cache_hit_count += 1
            if knowledge_result.meta.timed_out:
                knowledge_timeout_count += 1
            if knowledge_result.meta.circuit_open:
                knowledge_circuit_open_count += 1
            if knowledge_result.meta.error:
                knowledge_error_count += 1
            if (
                knowledge_result.transition_hints
                or knowledge_result.intent_hints
                or knowledge_result.state_hints
            ):
                knowledge_hit_count += 1
                knowledge_hint_text = build_knowledge_hint_text(
                    knowledge_result.summary,
                    knowledge_result.transition_hints,
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

        per_page_steps = min(
            steps_remaining,
            page_cap,
        )
        explorer = ReActExplorer(
            max_steps=per_page_steps,
            browser_session=browser,
            extra_system_prompt=build_login_hint_from_env() + "\n\n" + explorer_hint,
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
            if len(all_transitions) == transition_count_before:
                stuck_steps += 1
            else:
                stuck_steps = 0

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
                _enqueue_page(s_url, "in-page discovery")
                if len(pages_to_explore) > before:
                    harvested += 1
            if harvested:
                print(
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
            print(f"[PIPELINE] ReActExplorer failed: {e}")
            failed_action_count += 1
            stuck_steps += 1
            pages_explored.add(url_clean)
            await ensure_browser_ready(browser, url)
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
                    _enqueue_page(task.target_url, f"llm-plan: {task.description}")

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

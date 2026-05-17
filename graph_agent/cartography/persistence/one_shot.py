from __future__ import annotations

import hashlib
import json
import logging
from typing import cast
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from graph_agent.cartography.config import clean_url, is_http_url
from graph_agent.lib.observability import observe
from graph_agent.cartography.persistence.semantic_stability import (
    SEMANTIC_STABILITY_THRESHOLD,
    _as_float,
    _as_int,
    _as_str,
    _build_semantic_metrics,
    _calculate_semantic_stability,
    _clean_selector_suffix,
    _empty_stats,
    _source_priority,
    _transition_stable_key,
)

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


@observe(
    name="cartography.persist_mapping_result",
    metadata={"component": "cartography", "stage": "persistence"},
)
async def persist_mapping_result(
    *,
    app_id: str,
    app_name: str,
    session_id: str,
    resolved_url: str,
    current_url: str,
    inventory: list[dict[str, object]],
    initial_actions_log: list[dict[str, object]],
    result: "CartographyResult",
) -> None:
    from graph_agent.models import (
        ActionType,
        App,
        CoverageSnapshot,
        Evidence,
        EvidenceType,
        GraphRelease,
        IngestionRun,
        Session,
        State,
        Transition,
        TransitionEntity,
        TransitionRevision,
        TransitionSourceType,
    )
    from graph_agent.coverage.analyzer import CoverageAnalyzer
    from graph_agent.neo4j_client.manager import GraphManager
    from graph_agent.neo4j_client.queries import CypherQueries

    async with GraphManager() as manager:
        runtime: dict[str, object] = {
            "ingest_version_id": "",
            "active_revision_ids": [],
            "menu_rows": [],
            "stats": _empty_stats(),
            "state_ids_by_url": {},
        }

        def _stats() -> dict[str, int]:
            return cast(dict[str, int], runtime["stats"])

        def _ingest_version_id() -> str:
            return str(runtime["ingest_version_id"] or "")

        def _active_revision_ids() -> list[str]:
            return cast(list[str], runtime["active_revision_ids"])

        def _menu_rows() -> list[dict[str, object]]:
            return cast(list[dict[str, object]], runtime["menu_rows"])

        def _state_ids_by_url() -> dict[str, set[str]]:
            return cast(dict[str, set[str]], runtime["state_ids_by_url"])

        def _index_state_url(state_id: str, url: str) -> None:
            raw = _as_str(url).strip()
            if not raw:
                return
            keys = {raw, clean_url(raw)}
            for key in keys:
                if not key:
                    continue
                _state_ids_by_url().setdefault(key, set()).add(state_id)

        async def _persist_core_nodes() -> None:
            app = App(id=app_id, name=app_name, entry_url=resolved_url)
            await manager.add_app(app)
            session = Session(id=session_id, app_id=app_id)
            await manager.add_session(session)
            await manager.link_app_session(app_id, session_id)

            source_types = {
                str(getattr(t, "source_type", "auto") or "auto")
                for t in result.transitions
            }
            if source_types and all(v.startswith("manual") for v in source_types):
                mode = "manual"
            elif any(v.startswith("manual") for v in source_types):
                mode = "mixed"
            else:
                mode = "auto"

            ingest_seed = (
                f"{app_id}|{session_id}|{mode}|{len(result.states)}|{len(result.transitions)}|"
                f"{datetime.now(UTC).isoformat()}"
            )
            runtime["ingest_version_id"] = (
                f"ingest:{hashlib.md5(ingest_seed.encode(), usedforsecurity=False).hexdigest()[:16]}"
            )
            await manager.add_ingestion_run(
                IngestionRun(
                    id=_ingest_version_id(),
                    app_id=app_id,
                    session_id=session_id,
                    mode=mode,
                    source="cartography",
                    status="completed",
                )
            )
            stage_ops: list[tuple[str, dict[str, object]]] = [
                (
                    CypherQueries.LINK_SESSION_INGESTION_RUN,
                    {"session_id": session_id, "ingest_id": _ingest_version_id()},
                )
            ]
            if inventory:
                stage_ops.append(
                    (
                        CypherQueries.SET_SESSION_INVENTORY,
                        {"session_id": session_id, "inventory": {"elements": inventory}},
                    )
                )
            await manager.run_write_transaction(stage_ops)

            login_url = resolved_url
            post_login_url = current_url or resolved_url
            if login_url != post_login_url and initial_actions_log:
                login_fp = hashlib.md5(login_url.encode(), usedforsecurity=False).hexdigest()[:12]
                post_fp = hashlib.md5(post_login_url.encode(), usedforsecurity=False).hexdigest()[:12]
                from_state = State(
                    id=f"state:prelogin:{login_fp}",
                    url=login_url,
                    title="Login page",
                    ingest_version_id=_ingest_version_id(),
                )
                to_state = State(
                    id=f"state:prelogin:{post_fp}",
                    url=post_login_url,
                    title="Post-login page",
                    ingest_version_id=_ingest_version_id(),
                )
                prelogin_transition = Transition(
                    id=f"{session_id}:prelogin",
                    selector="[pre-login]",
                    action=ActionType.CLICK,
                    from_state_id=from_state.id,
                    to_state_id=to_state.id,
                    thought="Auto-login: filled credentials and submitted",
                    confidence=0.9,
                    ingest_version_id=_ingest_version_id(),
                )
                await manager.add_state(from_state)
                await manager.link_app_state(app_id, from_state.id)
                await manager.link_session_discovered(session_id, from_state.id)
                await manager.link_ingestion_emits_state(_ingest_version_id(), from_state.id)
                _index_state_url(from_state.id, from_state.url)
                _stats()["states_added"] += 1
                await manager.add_state(to_state)
                await manager.link_app_state(app_id, to_state.id)
                await manager.link_session_discovered(session_id, to_state.id)
                await manager.link_ingestion_emits_state(_ingest_version_id(), to_state.id)
                _index_state_url(to_state.id, to_state.url)
                _stats()["states_added"] += 1
                await manager.add_transition(prelogin_transition)
                await manager.link_session_transition(session_id, prelogin_transition.id)
                await manager.link_ingestion_emits_transition(
                    _ingest_version_id(), prelogin_transition.id
                )
                _stats()["transitions_added"] += 1

            seen_state_ids: set[str] = set()
            for state in result.states:
                if state.id in seen_state_ids:
                    continue
                state_to_store = state.model_copy(
                    update={
                        "ingest_version_id": state.ingest_version_id
                        or _ingest_version_id()
                    }
                )
                await manager.add_state(state_to_store)
                await manager.link_app_state(app_id, state.id)
                await manager.link_session_discovered(session_id, state.id)
                await manager.link_ingestion_emits_state(_ingest_version_id(), state.id)
                _index_state_url(state.id, state.url)
                seen_state_ids.add(state.id)
                _stats()["states_added"] += 1

        async def _persist_state_transition_revision() -> None:
            for transition in result.transitions:
                transition_to_store = transition.model_copy(
                    update={
                        "ingest_version_id": transition.ingest_version_id
                        or _ingest_version_id()
                    }
                )
                await manager.add_transition(transition_to_store)
                await manager.link_session_transition(session_id, transition.id)
                await manager.link_ingestion_emits_transition(
                    _ingest_version_id(), transition.id
                )
                _stats()["transitions_added"] += 1
                source_type = str(getattr(transition, "source_type", "auto") or "auto")
                if source_type.startswith("manual"):
                    _stats()["manual_transition_count"] += 1
                else:
                    _stats()["auto_transition_count"] += 1

                stable_key = _transition_stable_key(transition_to_store)
                action_value = str(transition_to_store.action.value)
                intent_key = ""
                if transition_to_store.intent is not None:
                    intent_key = _as_str(transition_to_store.intent.key).strip()
                if not intent_key:
                    intent_key = _as_str(transition_to_store.semantic_action_key).strip()
                await manager.add_transition_entity_with_session(
                    TransitionEntity(
                        stable_key=stable_key,
                        app_id=app_id,
                        from_state_id=_as_str(transition_to_store.from_state_id),
                        to_state_id=_as_str(transition_to_store.to_state_id),
                        action=action_value,
                        semantic_action_key=_as_str(
                            transition_to_store.semantic_action_key
                        )
                        or None,
                    ),
                    session_id=session_id,
                )
                existing_active = await manager.get_active_transition_revision(stable_key)
                source_enum = (
                    transition_to_store.source_type
                    if isinstance(
                        transition_to_store.source_type, TransitionSourceType
                    )
                    else TransitionSourceType.AUTO
                )
                revision_seed = f"{stable_key}|{transition.id}|{session_id}|{datetime.now(UTC).isoformat()}"
                revision_digest = hashlib.md5(revision_seed.encode(), usedforsecurity=False).hexdigest()[:16]
                revision_id = f"trev:{revision_digest}"
                incoming_conf = max(
                    0.0, min(1.0, _as_float(transition_to_store.confidence, 0.5))
                )
                should_activate = True
                supersedes_ids: list[str] = []
                if existing_active is not None:
                    existing_priority = _source_priority(existing_active.source_type.value)
                    incoming_priority = _source_priority(source_enum.value)
                    existing_conf = _as_float(existing_active.confidence, 0.0)
                    if incoming_priority > existing_priority:
                        should_activate = True
                    elif incoming_priority < existing_priority:
                        should_activate = False
                    else:
                        should_activate = incoming_conf >= existing_conf
                    if should_activate:
                        supersedes_ids.append(existing_active.revision_id)
                revision = TransitionRevision(
                    revision_id=revision_id,
                    stable_key=stable_key,
                    transition_id=transition_to_store.id,
                    confidence=incoming_conf,
                    intent_key=intent_key or None,
                    source_type=source_enum,
                    operator_id=_as_str(
                        getattr(transition_to_store, "operator_id", "agent")
                    )
                    or "agent",
                    selector=_as_str(transition_to_store.selector),
                    action=action_value,
                    from_state_id=_as_str(transition_to_store.from_state_id),
                    to_state_id=_as_str(transition_to_store.to_state_id),
                    session_id=session_id,
                    ingest_version_id=_ingest_version_id(),
                    is_active=should_activate,
                    action_value=_as_str(getattr(transition_to_store, "action_value", None)) or None,
                    param_name=_as_str(getattr(transition_to_store, "param_name", None)) or None,
                    element_snapshot=_as_str(getattr(transition_to_store, "element_snapshot", None)) or None,
                    frame_path=_as_str(getattr(transition_to_store, "frame_path", None)) or None,
                    tab_id=_as_str(getattr(transition_to_store, "tab_id", "tab-0")) or "tab-0",
                    target_tab_id=_as_str(getattr(transition_to_store, "target_tab_id", None)) or None,
                    tab_action=_as_str(getattr(transition_to_store, "tab_action", None)) or None,
                    thought=_as_str(getattr(transition_to_store, "thought", None)) or None,
                    step_index=getattr(transition_to_store, "step_index", None),
                    intent_failure_reason=_as_str(getattr(transition_to_store, "intent_failure_reason", None)) or None,
                )
                await manager.add_transition_revision(revision)
                await manager.link_ingestion_emits_revision(
                    _ingest_version_id(), revision.revision_id
                )
                if should_activate:
                    await manager.activate_transition_revision(
                        stable_key,
                        revision.revision_id,
                        supersedes_ids,
                    )
                    _active_revision_ids().append(revision.revision_id)
                else:
                    await manager.attach_transition_revision(
                        stable_key, revision.revision_id
                    )
            _stats()["semantic_mismatch_warnings"] = int(
                getattr(result, "semantic_conflict_count", 0) or 0
            )

        async def _persist_artifacts() -> None:
            for transition in result.transitions:
                intent_obj = transition.intent
                if intent_obj is not None:
                    # Derive stable intent identity from graph structure, not LLM output.
                    from_id = _as_str(transition.from_state_id).strip()
                    sel = _as_str(transition.selector).strip()
                    action_val = _as_str(transition.action.value if hasattr(transition.action, "value") else transition.action).strip().lower()
                    structural_id = f"intent:{from_id}:{sel}:{action_val}"
                    structural_key = f"{action_val}.{_clean_selector_suffix(sel)}"
                    resolved_intent_id = (
                        _as_str(intent_obj.id).strip()
                        or structural_id
                    )
                    if resolved_intent_id:
                        if not resolved_intent_id.startswith("intent:"):
                            resolved_intent_id = f"intent:{resolved_intent_id}"
                        intent_to_store = intent_obj.model_copy(
                            update={
                                "id": resolved_intent_id,
                                "key": _as_str(intent_obj.key).strip() or structural_key,
                            }
                        )
                        await manager.add_intent(intent_to_store)
                        await manager.link_transition_intent(
                            transition.id, resolved_intent_id
                        )
                        if transition.from_state_id and transition.selector:
                            await manager.link_zone_covers_intent(
                                from_state_id=from_id,
                                selector=sel,
                                intent_id=resolved_intent_id,
                                confidence=_as_float(intent_obj.confidence, 0.5),
                                session_id=session_id,
                            )

                for evidence_id in transition.evidence_ids:
                    evidence_type = (
                        EvidenceType.URL_CHANGE
                        if evidence_id.endswith("url_change")
                        else EvidenceType.DOM_DIFF
                    )
                    evidence = Evidence(
                        id=evidence_id,
                        transition_id=transition.id,
                        session_id=session_id,
                        evidence_type=evidence_type,
                        summary=f"{evidence_type.value} evidence for {transition.selector}",
                        payload=json.dumps(
                            {
                                "from_state_id": transition.from_state_id,
                                "to_state_id": transition.to_state_id,
                                "selector": transition.selector,
                                "step_index": transition.step_index,
                            },
                            ensure_ascii=False,
                        ),
                        confidence=min(1.0, max(0.1, transition.confidence)),
                        ingest_version_id=_ingest_version_id(),
                    )
                    await manager.add_evidence(evidence)
                    await manager.link_transition_evidence(transition.id, evidence.id)
                    await manager.link_session_evidence(session_id, evidence.id)
                    await manager.link_ingestion_emits_evidence(
                        _ingest_version_id(), evidence.id
                    )

            if getattr(result, "layout_evidence", None):
                for idx, ev in enumerate(result.layout_evidence):
                    if not isinstance(ev, dict):
                        continue
                    url = str(ev.get("url") or "")
                    layout_fp = str(ev.get("layout_fingerprint") or "")
                    if not layout_fp:
                        continue
                    evidence_id = f"evidence:layout:{session_id}:{idx}:{layout_fp[:8]}"
                    confidence = _as_float(ev.get("layout_confidence") or 0.5, 0.5)
                    layout_evidence = Evidence(
                        id=evidence_id,
                        transition_id="",
                        session_id=session_id,
                        evidence_type=EvidenceType.LAYOUT,
                        summary=f"layout evidence at step {ev.get('step', idx)}",
                        payload=json.dumps(
                            {
                                "url": url,
                                "step": _as_int(ev.get("step") or idx, idx),
                                "layout_fingerprint": layout_fp,
                                "layout_summary": str(ev.get("layout_summary") or ""),
                            },
                            ensure_ascii=False,
                        ),
                        confidence=max(0.1, min(1.0, confidence)),
                        ingest_version_id=_ingest_version_id(),
                    )
                    await manager.add_evidence(layout_evidence)
                    await manager.link_session_evidence(session_id, evidence_id)
                    await manager.link_ingestion_emits_evidence(
                        _ingest_version_id(), evidence_id
                    )

            if getattr(result, "menus", None):
                for i, menu in enumerate(result.menus):
                    if not isinstance(menu, dict):
                        continue
                    text = _as_str(menu.get("text")).strip()
                    href = _as_str(menu.get("href")).strip()
                    level = _as_int(menu.get("level") or 0, 0)
                    source_url = _as_str(menu.get("source_url")).strip()
                    if not text and not href:
                        continue
                    menu_id_src = f"{app_id}|{source_url}|{level}|{text}|{href}"
                    _menu_rows().append(
                        {
                            "id": f"menu:{hashlib.md5(menu_id_src.encode(), usedforsecurity=False).hexdigest()[:12]}",
                            "text": text or href,
                            "href": href,
                            "level": level,
                            "order": i,
                            "is_active": True,
                            "ingest_version_id": _ingest_version_id(),
                        }
                    )
                if _menu_rows():
                    await manager.add_menus(
                        app_id=app_id,
                        menus=_menu_rows(),
                        page_url=current_url or resolved_url,
                        session_id=session_id,
                    )
                    for menu in _menu_rows():
                        menu_id = _as_str(menu.get("id"))
                        if menu_id:
                            await manager.link_ingestion_emits_menu(
                                _ingest_version_id(), menu_id
                            )

            if getattr(result, "zone_hints", None):
                _status_priority = {
                    "undiscovered": 0,
                    "stale": 0,
                    "discovered": 1,
                    "partial": 2,
                    "explored": 3,
                    "validated": 4,
                }
                zone_state_map: dict[tuple[str, str, str], dict[str, object]] = {}
                for zone in result.zone_hints:
                    if not isinstance(zone, dict):
                        continue
                    selector = _as_str(zone.get("selector")).strip()
                    z_type = _as_str(zone.get("zone_type") or "content").strip()
                    source_url = _as_str(zone.get("source_url")).strip()
                    source_url_key = clean_url(source_url) if source_url else ""
                    if not selector:
                        continue
                    key = (z_type, selector, source_url_key)
                    status_raw = (
                        _as_str(zone.get("exploration_status") or "discovered")
                        .strip()
                        .lower()
                        or "discovered"
                    )
                    last_explored_iso = (
                        zone.get("last_explored") if zone.get("last_explored") else None
                    )
                    existing = zone_state_map.get(key)
                    if existing is None or _status_priority.get(
                        status_raw, 0
                    ) > _status_priority.get(str(existing.get("status") or ""), 0):
                        zone_state_map[key] = {
                            "status": status_raw,
                            "last_explored": last_explored_iso,
                        }
                    elif last_explored_iso and not existing.get("last_explored"):
                        existing["last_explored"] = last_explored_iso

                zone_rows: list[dict[str, object]] = []
                zone_rows_by_id: dict[str, dict[str, object]] = {}
                for zone in result.zone_hints:
                    if not isinstance(zone, dict):
                        continue
                    selector = _as_str(zone.get("selector")).strip()
                    z_type = _as_str(zone.get("zone_type") or "content").strip()
                    summary = _as_str(zone.get("description")).strip()
                    source_url = _as_str(zone.get("source_url")).strip()
                    source_url_key = clean_url(source_url) if source_url else ""
                    if not selector:
                        continue
                    zid_src = f"{app_id}|{z_type}|{selector}|{source_url_key}"
                    zone_id = f"zone:{hashlib.md5(zid_src.encode(), usedforsecurity=False).hexdigest()[:12]}"
                    related_state_ids: set[str] = set()
                    for key in (source_url, clean_url(source_url)):
                        if not key:
                            continue
                        related_state_ids.update(_state_ids_by_url().get(key, set()))
                    state_info = zone_state_map.get((z_type, selector, source_url_key), {})
                    existing_row = zone_rows_by_id.get(zone_id)
                    if existing_row is None:
                        row = {
                            "id": zone_id,
                            "type": z_type,
                            "selector": selector,
                            "element_count": 0,
                            "bounds": "",
                            "text_sample": summary,
                            "ingest_version_id": _ingest_version_id(),
                            "exploration_status": str(
                                state_info.get("status") or "discovered"
                            ),
                            "last_explored": state_info.get("last_explored"),
                            "state_ids": sorted(related_state_ids),
                        }
                        zone_rows.append(row)
                        zone_rows_by_id[zone_id] = row
                        continue

                    raw_ids = existing_row.get("state_ids") or []
                    merged_state_ids = set(raw_ids if isinstance(raw_ids, list) else [])
                    merged_state_ids.update(related_state_ids)
                    existing_row["state_ids"] = sorted(
                        _as_str(v).strip() for v in merged_state_ids if _as_str(v).strip()
                    )
                    if summary and not _as_str(existing_row.get("text_sample")).strip():
                        existing_row["text_sample"] = summary
                    existing_status = (
                        _as_str(existing_row.get("exploration_status") or "discovered")
                        .strip()
                        .lower()
                    )
                    incoming_status = (
                        _as_str(state_info.get("status") or "discovered")
                        .strip()
                        .lower()
                    )
                    if _status_priority.get(incoming_status, 0) > _status_priority.get(
                        existing_status, 0
                    ):
                        existing_row["exploration_status"] = incoming_status
                    if (
                        state_info.get("last_explored")
                        and not existing_row.get("last_explored")
                    ):
                        existing_row["last_explored"] = state_info.get("last_explored")
                if zone_rows:
                    await manager.add_zones(app_id=app_id, zones=zone_rows)
                    for zone in zone_rows:
                        zone_id = _as_str(zone.get("id"))
                        if zone_id:
                            await manager.link_ingestion_emits_zone(
                                _ingest_version_id(), zone_id
                            )

            if _menu_rows():
                for transition in result.transitions:
                    signal = f"{transition.selector} {transition.thought or ''}".lower()
                    for menu in _menu_rows():
                        text = str(menu.get("text") or "").strip().lower()
                        if text and text in signal:
                            await manager.link_transition_navigated_via(
                                transition.id, str(menu["id"])
                            )
                            break

        async def _persist_release_and_snapshot() -> None:
            visited_urls: list[str] = []
            if result.history:
                for h in result.history:
                    if not isinstance(h, dict):
                        continue
                    url = _as_str(h.get("url", ""))
                    if url and is_http_url(url):
                        visited_urls.append(url)
            visited_urls = list(dict.fromkeys(visited_urls))

            last_history = result.history[-1] if result.history else {}
            final_text_str = _as_str(
                last_history.get("result", "") if isinstance(last_history, dict) else ""
            )
            mapping_stopped = False
            stop_reason = None
            for marker in ("Stopped:", "停止："):
                if marker in final_text_str:
                    idx = final_text_str.find(marker)
                    stop_reason = final_text_str[idx + len(marker) :].strip()
                    if len(stop_reason) > 200:
                        stop_reason = stop_reason[:200] + "..."
                    mapping_stopped = True
                    break

            release_id = (
                f"release:{app_id}:{hashlib.md5(_ingest_version_id().encode(), usedforsecurity=False).hexdigest()[:12]}"
            )
            await manager.add_graph_release(
                GraphRelease(
                    id=release_id,
                    app_id=app_id,
                    base_ingest_ids=[_ingest_version_id()],
                    status="active",
                )
            )
            await manager.deactivate_other_active_releases(
                app_id=app_id,
                keep_release_id=release_id,
            )
            for revision_id in _active_revision_ids():
                await manager.link_release_revision(release_id, revision_id)

            coverage_snapshot_id: str | None = None
            try:
                analyzer = CoverageAnalyzer(manager.get_driver())
                report = await analyzer.compute(app_id=app_id)
                coverage_snapshot_id = (
                    f"cov:{session_id}:"
                    f"{hashlib.md5(release_id.encode(), usedforsecurity=False).hexdigest()[:8]}"
                )
                snapshot = CoverageSnapshot(
                    id=coverage_snapshot_id,
                    app_id=app_id,
                    session_id=session_id,
                    release_id=release_id,
                    menu_coverage=report.menu_coverage,
                    zone_coverage=report.zone_coverage,
                    interaction_coverage=report.interaction_coverage,
                    state_coverage=report.state_coverage,
                    overall_completeness=report.overall_completeness,
                    transition_high=report.transition_confidence.high,
                    transition_medium=report.transition_confidence.medium,
                    transition_low=report.transition_confidence.low,
                    recommendation=report.recommendation,
                )
                await manager.add_coverage_snapshot(snapshot)
                await manager.link_session_coverage(session_id, coverage_snapshot_id)
                await manager.link_release_coverage(release_id, coverage_snapshot_id)
            except Exception as e:
                logger.warning("[PERSIST] coverage snapshot skipped: %s", e)

            semantic_metrics = _build_semantic_metrics(result)
            baseline_row = await manager.get_latest_semantic_baseline(
                app_id=app_id,
                exclude_session_id=session_id,
            )
            semantic_stability_metrics = _calculate_semantic_stability(
                semantic_metrics=semantic_metrics,
                baseline_row=baseline_row,
                threshold=SEMANTIC_STABILITY_THRESHOLD,
            )
            layout_metrics_payload = (
                dict(result.layout_metrics)
                if isinstance(result.layout_metrics, dict)
                else {}
            )
            for key, default in {
                "captcha_action_count": 0,
                "captcha_ok_count": 0,
                "captcha_empty_count": 0,
                "captcha_manual_empty_count": 0,
                "captcha_fill_failed_count": 0,
                "captcha_manual_wait_ms_total": 0,
            }.items():
                layout_metrics_payload.setdefault(key, default)

            await manager.update_session_stats(
                session_id=session_id,
                stats={
                    "visited_urls": visited_urls,
                    "mapping_stopped": mapping_stopped,
                    "stop_reason": stop_reason,
                    "start_url": resolved_url,
                    "current_release_id": release_id,
                    "latest_ingest_version_id": _ingest_version_id(),
                    "current_coverage_snapshot_id": coverage_snapshot_id or "",
                    "intervention_task_count": len(
                        getattr(result, "intervention_tasks", []) or []
                    ),
                    "intervention_tasks": list(
                        getattr(result, "intervention_tasks", []) or []
                    ),
                    **_stats(),
                    **layout_metrics_payload,
                    **semantic_metrics,
                    **semantic_stability_metrics,
                },
            )
            await manager.touch_app_last_session(app_id)
            await manager.update_app_stats(app_id)

        await _persist_core_nodes()
        await _persist_state_transition_revision()
        await _persist_artifacts()
        await _persist_release_and_snapshot()
        logger.info(
            f"Graph stored in Neo4j: app_id={app_id} "
            f"(states={_stats().get('states_added', 0)}, transitions={_stats().get('transitions_added', 0)})"
        )

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING

from graph_agent.cartography.config import is_http_url
from graph_agent.cartography.types import LayoutEvidenceItem, LayoutMetrics
from graph_agent.lib.observability import observe

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult


def _as_str(value: object) -> str:
    return str(value or "")


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_priority(source_type: object) -> int:
    value = str(source_type or "auto").strip().lower()
    if value == "manual_graph_assisted":
        return 3
    if value == "manual_raw":
        return 2
    return 1


def _transition_stable_key(transition: object) -> str:
    from_id = _as_str(getattr(transition, "from_state_id", "")).strip()
    to_id = _as_str(getattr(transition, "to_state_id", "")).strip()
    action = _as_str(getattr(transition, "action", "")).strip().lower()
    semantic = _as_str(getattr(transition, "semantic_action_key", "")).strip()
    if not semantic:
        semantic = _as_str(getattr(transition, "selector", "")).strip()
    return f"{from_id}|{to_id}|{action}|{semantic}"


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
    inventory: list[dict],
    initial_actions_log: list[dict[str, object]],
    result: "CartographyResult",
) -> None:
    from graph_agent.models import (
        ActionType,
        App,
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
    from graph_agent.neo4j_client.manager import GraphManager

    async with GraphManager() as manager:
        app = App(id=app_id, name=app_name, entry_url=resolved_url)
        await manager.add_app(app)
        session = Session(id=session_id, app_id=app_id)
        await manager.add_session(session)
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
            f"{datetime.utcnow().isoformat()}"
        )
        ingest_version_id = f"ingest:{hashlib.md5(ingest_seed.encode()).hexdigest()[:16]}"
        await manager.add_ingestion_run(
            IngestionRun(
                id=ingest_version_id,
                app_id=app_id,
                session_id=session_id,
                mode=mode,
                source="cartography",
                status="completed",
            )
        )
        await manager.link_session_ingestion_run(session_id, ingest_version_id)
        if inventory:
            await manager.set_session_inventory(session_id, {"elements": inventory})

        stats = {
            "states_added": 0,
            "transitions_added": 0,
            "filtered_non_ui_edges": 0,
            "semantic_mismatch_warnings": 0,
            "url_discontinuity_warnings": 0,
            "frame_context_transition_warnings": 0,
            "manual_transition_count": 0,
            "auto_transition_count": 0,
        }

        login_url = resolved_url
        post_login_url = current_url or resolved_url
        if login_url != post_login_url and initial_actions_log:
            login_fp = hashlib.md5(login_url.encode()).hexdigest()[:12]
            post_fp = hashlib.md5(post_login_url.encode()).hexdigest()[:12]
            from_state = State(
                id=f"state:prelogin:{login_fp}",
                url=login_url,
                title="Login page",
                ingest_version_id=ingest_version_id,
            )
            to_state = State(
                id=f"state:prelogin:{post_fp}",
                url=post_login_url,
                title="Post-login page",
                ingest_version_id=ingest_version_id,
            )
            prelogin_transition = Transition(
                id=f"{session_id}:prelogin",
                selector="[pre-login]",
                action=ActionType.CLICK,
                from_state_id=from_state.id,
                to_state_id=to_state.id,
                thought="Auto-login: filled credentials and submitted",
                confidence=0.9,
                ingest_version_id=ingest_version_id,
            )
            await manager.add_state(from_state)
            await manager.link_app_state(app_id, from_state.id)
            await manager.link_session_discovered(session_id, from_state.id)
            await manager.link_ingestion_emits_state(ingest_version_id, from_state.id)
            stats["states_added"] += 1
            await manager.add_state(to_state)
            await manager.link_app_state(app_id, to_state.id)
            await manager.link_session_discovered(session_id, to_state.id)
            await manager.link_ingestion_emits_state(ingest_version_id, to_state.id)
            stats["states_added"] += 1
            await manager.add_transition(prelogin_transition)
            await manager.link_session_transition(session_id, prelogin_transition.id)
            await manager.link_ingestion_emits_transition(
                ingest_version_id, prelogin_transition.id
            )
            stats["transitions_added"] += 1

        seen_state_ids: set[str] = set()
        for state in result.states:
            if state.id in seen_state_ids:
                continue
            state_to_store = state.model_copy(
                update={"ingest_version_id": state.ingest_version_id or ingest_version_id}
            )
            await manager.add_state(state_to_store)
            await manager.link_app_state(app_id, state.id)
            await manager.link_session_discovered(session_id, state.id)
            await manager.link_ingestion_emits_state(ingest_version_id, state.id)
            seen_state_ids.add(state.id)
            stats["states_added"] += 1

        active_revision_ids: list[str] = []
        for transition in result.transitions:
            transition_to_store = transition.model_copy(
                update={
                    "ingest_version_id": transition.ingest_version_id or ingest_version_id
                }
            )
            await manager.add_transition(transition_to_store)
            await manager.link_session_transition(session_id, transition.id)
            await manager.link_ingestion_emits_transition(ingest_version_id, transition.id)
            stats["transitions_added"] += 1
            source_type = str(getattr(transition, "source_type", "auto") or "auto")
            if source_type.startswith("manual"):
                stats["manual_transition_count"] += 1
            else:
                stats["auto_transition_count"] += 1

            stable_key = _transition_stable_key(transition_to_store)
            action_value = str(transition_to_store.action.value)
            intent_key = ""
            if transition_to_store.intent is not None:
                intent_key = _as_str(transition_to_store.intent.key).strip()
            if not intent_key:
                intent_key = _as_str(transition_to_store.semantic_action_key).strip()
            await manager.add_transition_entity(
                TransitionEntity(
                    stable_key=stable_key,
                    app_id=app_id,
                    from_state_id=_as_str(transition_to_store.from_state_id),
                    to_state_id=_as_str(transition_to_store.to_state_id),
                    action=action_value,
                    semantic_action_key=_as_str(transition_to_store.semantic_action_key)
                    or None,
                )
            )
            existing_active = await manager.get_active_transition_revision(stable_key)
            source_enum = (
                transition_to_store.source_type
                if isinstance(transition_to_store.source_type, TransitionSourceType)
                else TransitionSourceType.AUTO
            )
            revision_id = (
                f"trev:{hashlib.md5(f'{stable_key}|{transition.id}|{session_id}|{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:16]}"
            )
            incoming_conf = max(0.0, min(1.0, _as_float(transition_to_store.confidence, 0.5)))
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
                operator_id=_as_str(getattr(transition_to_store, "operator_id", "agent")) or "agent",
                selector=_as_str(transition_to_store.selector),
                action=action_value,
                from_state_id=_as_str(transition_to_store.from_state_id),
                to_state_id=_as_str(transition_to_store.to_state_id),
                session_id=session_id,
                ingest_version_id=ingest_version_id,
                is_active=should_activate,
            )
            await manager.add_transition_revision(revision)
            await manager.link_ingestion_emits_revision(ingest_version_id, revision.revision_id)
            if should_activate:
                await manager.activate_transition_revision(
                    stable_key,
                    revision.revision_id,
                    supersedes_ids,
                )
                active_revision_ids.append(revision.revision_id)
            else:
                await manager.attach_transition_revision(stable_key, revision.revision_id)
        stats["semantic_mismatch_warnings"] = int(
            getattr(result, "semantic_conflict_count", 0) or 0
        )
        for transition in result.transitions:
            for evidence_id in transition.evidence_ids:
                evidence_type = (
                    EvidenceType.URL_CHANGE if evidence_id.endswith("url_change") else EvidenceType.DOM_DIFF
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
                    ingest_version_id=ingest_version_id,
                )
                await manager.add_evidence(evidence)
                await manager.link_transition_evidence(transition.id, evidence.id)
                await manager.link_session_evidence(session_id, evidence.id)
                await manager.link_ingestion_emits_evidence(ingest_version_id, evidence.id)

        # Persist layout evidence emitted by mapping pipeline (session-level).
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
                    ingest_version_id=ingest_version_id,
                )
                await manager.add_evidence(layout_evidence)
                await manager.link_session_evidence(session_id, evidence_id)
                await manager.link_ingestion_emits_evidence(ingest_version_id, evidence_id)

        menu_rows: list[dict[str, object]] = []
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
                menu_rows.append(
                    {
                        "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
                        "text": text or href,
                        "href": href,
                        "level": level,
                        "order": i,
                        "is_active": False,
                        "ingest_version_id": ingest_version_id,
                    }
                )
            if menu_rows:
                await manager.add_menus(
                    app_id=app_id,
                    menus=menu_rows,
                    page_url=current_url or resolved_url,
                    session_id=session_id,
                )
                for menu in menu_rows:
                    menu_id = _as_str(menu.get("id"))
                    if menu_id:
                        await manager.link_ingestion_emits_menu(ingest_version_id, menu_id)

        if getattr(result, "zone_hints", None):
            zone_rows: list[dict[str, object]] = []
            for zone in result.zone_hints:
                if not isinstance(zone, dict):
                    continue
                selector = _as_str(zone.get("selector")).strip()
                z_type = _as_str(zone.get("zone_type") or "content").strip()
                summary = _as_str(zone.get("description")).strip()
                if not selector:
                    continue
                zid_src = f"{app_id}|{z_type}|{selector}"
                zone_rows.append(
                    {
                        "id": f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}",
                        "type": z_type,
                        "selector": selector,
                        "element_count": 0,
                        "bounds": "",
                        "text_sample": summary,
                        "ingest_version_id": ingest_version_id,
                    }
                )
            if zone_rows:
                await manager.add_zones(app_id=app_id, zones=zone_rows)
                for zone in zone_rows:
                    zone_id = _as_str(zone.get("id"))
                    if zone_id:
                        await manager.link_ingestion_emits_zone(ingest_version_id, zone_id)

        if menu_rows:
            for transition in result.transitions:
                signal = f"{transition.selector} {transition.thought or ''}".lower()
                for menu in menu_rows:
                    text = str(menu.get("text") or "").strip().lower()
                    if text and text in signal:
                        await manager.link_transition_navigated_via(transition.id, str(menu["id"]))
                        break

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
        final_text_str = _as_str(last_history.get("result", "") if isinstance(last_history, dict) else "")
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

        release_id = f"release:{app_id}:{hashlib.md5(ingest_version_id.encode()).hexdigest()[:12]}"
        await manager.add_graph_release(
            GraphRelease(
                id=release_id,
                app_id=app_id,
                base_ingest_ids=[ingest_version_id],
                status="active",
            )
        )
        for revision_id in active_revision_ids:
            await manager.link_release_revision(release_id, revision_id)

        await manager.update_session_stats(
            session_id=session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": resolved_url,
                "current_release_id": release_id,
                "latest_ingest_version_id": ingest_version_id,
                "intervention_task_count": len(getattr(result, "intervention_tasks", []) or []),
                "intervention_tasks": list(getattr(result, "intervention_tasks", []) or []),
                **stats,
                **(result.layout_metrics if isinstance(result.layout_metrics, dict) else {}),
            },
        )
        print(
            f"Graph stored in Neo4j: app_id={app_id} "
            f"(states={stats.get('states_added', 0)}, transitions={stats.get('transitions_added', 0)})"
        )

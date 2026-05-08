from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from graph_agent.cartography.config import clean_url, is_http_url
from graph_agent.cartography.types import LayoutEvidenceItem, LayoutMetrics
from graph_agent.lib.observability import observe

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


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
    # Include steps digest for intent-level transitions
    steps = getattr(transition, "steps", None)
    steps_digest = ""
    if steps:
        steps_summary = "|".join(
            f"{getattr(s, 'action', '')}:{getattr(s, 'selector', '')}"
            for s in steps
        )
        steps_digest = hashlib.md5(steps_summary.encode("utf-8")).hexdigest()[:8]
    return f"{from_id}|{to_id}|{action}|{semantic}|{steps_digest}"


def _state_semantic_key(state: object) -> str:
    route = _as_str(getattr(state, "spa_route", "")).strip().lower()
    cleaned = clean_url(_as_str(getattr(state, "url", "")).strip())
    url = _as_str(cleaned or "").split("#", 1)[0].strip().lower()
    title = _as_str(getattr(state, "title", "")).strip().lower()
    if route:
        return f"route:{route}"
    if url:
        return f"url:{url}"
    if title:
        return f"title:{title}"
    return _as_str(getattr(state, "id", "")).strip().lower()


def _stable_signature(items: set[str]) -> str:
    if not items:
        return ""
    payload = "\n".join(sorted(item for item in items if item))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


SEMANTIC_STABILITY_THRESHOLD = 90.0


def _json_list_payload(items: set[str]) -> str:
    return json.dumps(sorted(item for item in items if item), ensure_ascii=False)


def _parse_json_list(raw: object) -> set[str]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(item).strip() for item in parsed if str(item).strip()}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def _calculate_semantic_stability(
    *,
    semantic_metrics: dict[str, object],
    baseline_row: dict[str, object] | None,
    threshold: float = SEMANTIC_STABILITY_THRESHOLD,
) -> dict[str, object]:
    baseline = baseline_row or {}
    baseline_session_id = _as_str(baseline.get("session_id")).strip()
    current_state = _parse_json_list(semantic_metrics.get("semantic_state_keys_json"))
    current_transition = _parse_json_list(
        semantic_metrics.get("semantic_transition_keys_json")
    )
    current_intent = _parse_json_list(semantic_metrics.get("semantic_intent_keys_json"))
    baseline_state = _parse_json_list(baseline.get("semantic_state_keys_json"))
    baseline_transition = _parse_json_list(baseline.get("semantic_transition_keys_json"))
    baseline_intent = _parse_json_list(baseline.get("semantic_intent_keys_json"))

    if not baseline_session_id:
        return {
            "semantic_baseline_session_id": "",
            "semantic_stability_state_similarity": 1.0,
            "semantic_stability_transition_similarity": 1.0,
            "semantic_stability_intent_similarity": 1.0,
            "semantic_stability_score": 100.0,
            "semantic_stability_threshold": float(threshold),
            "semantic_stability_passed": True,
            "semantic_stability_mode": "bootstrap",
        }

    state_similarity = _jaccard_similarity(current_state, baseline_state)
    transition_similarity = _jaccard_similarity(current_transition, baseline_transition)
    intent_similarity = _jaccard_similarity(current_intent, baseline_intent)
    score = (
        0.4 * state_similarity
        + 0.4 * transition_similarity
        + 0.2 * intent_similarity
    ) * 100.0
    score = round(score, 2)
    threshold_value = float(threshold)
    return {
        "semantic_baseline_session_id": baseline_session_id,
        "semantic_stability_state_similarity": round(state_similarity, 4),
        "semantic_stability_transition_similarity": round(transition_similarity, 4),
        "semantic_stability_intent_similarity": round(intent_similarity, 4),
        "semantic_stability_score": score,
        "semantic_stability_threshold": threshold_value,
        "semantic_stability_passed": score >= threshold_value,
        "semantic_stability_mode": "compare",
    }


def _build_semantic_metrics(result: object) -> dict[str, object]:
    states = list(getattr(result, "states", []) or [])
    transitions = list(getattr(result, "transitions", []) or [])
    state_key_by_id: dict[str, str] = {}
    state_keys: set[str] = set()
    for state in states:
        key = _state_semantic_key(state)
        sid = _as_str(getattr(state, "id", "")).strip()
        if sid:
            state_key_by_id[sid] = key
        if key:
            state_keys.add(key)

    transition_keys: set[str] = set()
    intent_keys: set[str] = set()
    for transition in transitions:
        from_id = _as_str(getattr(transition, "from_state_id", "")).strip()
        to_id = _as_str(getattr(transition, "to_state_id", "")).strip()
        from_key = state_key_by_id.get(from_id, from_id)
        to_key = state_key_by_id.get(to_id, to_id)
        action = _as_str(getattr(transition, "action", "")).strip().lower()
        semantic = _as_str(getattr(transition, "semantic_action_key", "")).strip()
        if not semantic:
            semantic = _as_str(getattr(transition, "selector", "")).strip()
        transition_keys.add(f"{from_key}|{to_key}|{action}|{semantic}")

        intent = getattr(transition, "intent", None)
        intent_key = ""
        if intent is not None:
            intent_key = _as_str(getattr(intent, "key", "")).strip()
        if not intent_key:
            intent_key = semantic
        if intent_key:
            intent_keys.add(intent_key)

    return {
        "semantic_state_count": len(state_keys),
        "semantic_transition_count": len(transition_keys),
        "semantic_intent_count": len(intent_keys),
        "semantic_state_signature": _stable_signature(state_keys),
        "semantic_transition_signature": _stable_signature(transition_keys),
        "semantic_intent_signature": _stable_signature(intent_keys),
        "semantic_state_keys_json": _json_list_payload(state_keys),
        "semantic_transition_keys_json": _json_list_payload(transition_keys),
        "semantic_intent_keys_json": _json_list_payload(intent_keys),
    }


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
            "stats": {
                "states_added": 0,
                "transitions_added": 0,
                "filtered_non_ui_edges": 0,
                "semantic_mismatch_warnings": 0,
                "url_discontinuity_warnings": 0,
                "frame_context_transition_warnings": 0,
                "manual_transition_count": 0,
                "auto_transition_count": 0,
            },
            "state_ids_by_url": {},
        }

        def _stats() -> dict[str, int]:
            return runtime["stats"]  # type: ignore[return-value]

        def _ingest_version_id() -> str:
            return str(runtime["ingest_version_id"] or "")

        def _active_revision_ids() -> list[str]:
            return runtime["active_revision_ids"]  # type: ignore[return-value]

        def _menu_rows() -> list[dict[str, object]]:
            return runtime["menu_rows"]  # type: ignore[return-value]

        def _state_ids_by_url() -> dict[str, set[str]]:
            return runtime["state_ids_by_url"]  # type: ignore[return-value]

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
                f"ingest:{hashlib.md5(ingest_seed.encode()).hexdigest()[:16]}"
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
                login_fp = hashlib.md5(login_url.encode()).hexdigest()[:12]
                post_fp = hashlib.md5(post_login_url.encode()).hexdigest()[:12]
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
                revision_id = (
                    f"trev:{hashlib.md5(f'{stable_key}|{transition.id}|{session_id}|{datetime.now(UTC).isoformat()}'.encode()).hexdigest()[:16]}"
                )
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
                    resolved_intent_id = (
                        _as_str(intent_obj.id).strip()
                        or _as_str(intent_obj.key).strip()
                        or _as_str(intent_obj.summary).strip()
                    )
                    if resolved_intent_id:
                        if not resolved_intent_id.startswith("intent:"):
                            resolved_intent_id = f"intent:{resolved_intent_id}"
                        intent_to_store = intent_obj.model_copy(
                            update={"id": resolved_intent_id}
                        )
                        await manager.add_intent(intent_to_store)
                        await manager.link_transition_intent(
                            transition.id, resolved_intent_id
                        )
                        if transition.from_state_id and transition.selector:
                            await manager.link_zone_covers_intent(
                                from_state_id=_as_str(transition.from_state_id),
                                selector=_as_str(transition.selector),
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
                            "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
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
                    zone_id = f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}"
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

                    merged_state_ids = set(existing_row.get("state_ids") or [])
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
                f"release:{app_id}:{hashlib.md5(_ingest_version_id().encode()).hexdigest()[:12]}"
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
                    f"{hashlib.md5(release_id.encode()).hexdigest()[:8]}"
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
            except Exception as e:  # noqa: BLE001
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

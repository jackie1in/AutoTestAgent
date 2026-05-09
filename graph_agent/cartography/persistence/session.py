from __future__ import annotations

import json
import logging
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
    _empty_stats,
    _jaccard_similarity,
    _json_list_payload,
    _parse_json_list,
    _source_priority,
    _stable_signature,
    _state_semantic_key,
    _transition_stable_key,
)

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


class PersistenceSession:
    """Manages an incremental persistence session across multiple page explorations.

    Usage::

        session = PersistenceSession(...)
        await session.init()
        for page_result in page_results:
            await session.persist_page(page_result)
        await session.finalize(final_result)
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_name: str,
        session_id: str,
        resolved_url: str,
        current_url: str,
        inventory: list[dict],
        initial_actions_log: list[dict[str, object]],
        mode: str = "auto",
    ) -> None:
        self.app_id = app_id
        self.app_name = app_name
        self.session_id = session_id
        self.resolved_url = resolved_url
        self.current_url = current_url
        self.inventory = inventory
        self.initial_actions_log = initial_actions_log
        self.mode = mode

        self._manager: object | None = None
        self._ingest_version_id: str = ""
        self._active_revision_ids: list[str] = []
        self._menu_rows: list[dict[str, object]] = []
        self._stats: dict[str, int] = _empty_stats()
        self._state_ids_by_url: dict[str, set[str]] = {}
        self._seen_state_ids: set[str] = set()
        self._seen_transition_ids: set[str] = set()
        self._initialized = False
        self._finalized = False
        self._accumulated_result: "CartographyResult | None" = None

    @property
    def ingest_version_id(self) -> str:
        return self._ingest_version_id

    @property
    def stats(self) -> dict[str, int]:
        return self._stats

    def _index_state_url(self, state_id: str, url: str) -> None:
        raw = _as_str(url).strip()
        if not raw:
            return
        keys = {raw, clean_url(raw)}
        for key in keys:
            if not key:
                continue
            self._state_ids_by_url.setdefault(key, set()).add(state_id)

    async def init(self) -> None:
        """Phase 1: Create App, Session, IngestionRun, and optional prelogin nodes."""
        from graph_agent.models import (
            ActionType,
            App,
            IngestionRun,
            Session,
            State,
            Transition,
        )
        from graph_agent.neo4j_client.manager import GraphManager
        from graph_agent.neo4j_client.queries import CypherQueries

        from graph_agent.graph.merger import CartographyResult

        self._accumulated_result = CartographyResult()

        manager = GraphManager()
        await manager.__aenter__()
        self._manager = manager

        app = App(id=self.app_id, name=self.app_name, entry_url=self.resolved_url)
        await manager.add_app(app)
        session = Session(id=self.session_id, app_id=self.app_id)
        await manager.add_session(session)
        await manager.link_app_session(self.app_id, self.session_id)

        ingest_seed = (
            f"{self.app_id}|{self.session_id}|{self.mode}|0|0|"
            f"{datetime.now(UTC).isoformat()}"
        )
        self._ingest_version_id = (
            f"ingest:{hashlib.md5(ingest_seed.encode()).hexdigest()[:16]}"
        )
        await manager.add_ingestion_run(
            IngestionRun(
                id=self._ingest_version_id,
                app_id=self.app_id,
                session_id=self.session_id,
                mode=self.mode,
                source="cartography",
                status="in_progress",
            )
        )
        stage_ops: list[tuple[str, dict[str, object]]] = [
            (
                CypherQueries.LINK_SESSION_INGESTION_RUN,
                {"session_id": self.session_id, "ingest_id": self._ingest_version_id},
            )
        ]
        if self.inventory:
            stage_ops.append(
                (
                    CypherQueries.SET_SESSION_INVENTORY,
                    {"session_id": self.session_id, "inventory": {"elements": self.inventory}},
                )
            )
        await manager.run_write_transaction(stage_ops)

        login_url = self.resolved_url
        post_login_url = self.current_url or self.resolved_url
        if login_url != post_login_url and self.initial_actions_log:
            login_fp = hashlib.md5(login_url.encode()).hexdigest()[:12]
            post_fp = hashlib.md5(post_login_url.encode()).hexdigest()[:12]
            from_state = State(
                id=f"state:prelogin:{login_fp}",
                url=login_url,
                title="Login page",
                ingest_version_id=self._ingest_version_id,
            )
            to_state = State(
                id=f"state:prelogin:{post_fp}",
                url=post_login_url,
                title="Post-login page",
                ingest_version_id=self._ingest_version_id,
            )
            prelogin_transition = Transition(
                id=f"{self.session_id}:prelogin",
                selector="[pre-login]",
                action=ActionType.CLICK,
                from_state_id=from_state.id,
                to_state_id=to_state.id,
                thought="Auto-login: filled credentials and submitted",
                confidence=0.9,
                ingest_version_id=self._ingest_version_id,
            )
            await manager.add_state(from_state)
            await manager.link_app_state(self.app_id, from_state.id)
            await manager.link_session_discovered(self.session_id, from_state.id)
            await manager.link_ingestion_emits_state(self._ingest_version_id, from_state.id)
            self._index_state_url(from_state.id, from_state.url)
            self._stats["states_added"] += 1

            await manager.add_state(to_state)
            await manager.link_app_state(self.app_id, to_state.id)
            await manager.link_session_discovered(self.session_id, to_state.id)
            await manager.link_ingestion_emits_state(self._ingest_version_id, to_state.id)
            self._index_state_url(to_state.id, to_state.url)
            self._stats["states_added"] += 1

            await manager.add_transition(prelogin_transition)
            await manager.link_session_transition(self.session_id, prelogin_transition.id)
            await manager.link_ingestion_emits_transition(
                self._ingest_version_id, prelogin_transition.id
            )
            self._stats["transitions_added"] += 1

        self._initialized = True
        logger.info(
            "[PERSIST] Session initialized: app_id=%s ingest=%s",
            self.app_id, self._ingest_version_id,
        )

    @observe(
        name="cartography.persist_page_result",
        metadata={"component": "cartography", "stage": "persistence"},
    )
    async def persist_page(self, page_result: "CartographyResult") -> None:
        """Phase 2 (incremental): Write a single page's states, transitions, intents, evidence."""
        if not self._initialized or self._manager is None:
            logger.warning("[PERSIST] persist_page called before init, skipping")
            return

        from graph_agent.models import (
            Evidence,
            EvidenceType,
            TransitionEntity,
            TransitionRevision,
            TransitionSourceType,
        )

        manager = self._manager

        # --- States ---
        for state in page_result.states:
            if state.id in self._seen_state_ids:
                continue
            state_to_store = state.model_copy(
                update={
                    "ingest_version_id": state.ingest_version_id
                    or self._ingest_version_id
                }
            )
            await manager.add_state(state_to_store)
            await manager.link_app_state(self.app_id, state.id)
            await manager.link_session_discovered(self.session_id, state.id)
            await manager.link_ingestion_emits_state(self._ingest_version_id, state.id)
            self._index_state_url(state.id, state.url)
            self._seen_state_ids.add(state.id)
            self._stats["states_added"] += 1

        # --- Transitions + Entities + Revisions ---
        for transition in page_result.transitions:
            if transition.id in self._seen_transition_ids:
                continue
            transition_to_store = transition.model_copy(
                update={
                    "ingest_version_id": transition.ingest_version_id
                    or self._ingest_version_id
                }
            )
            await manager.add_transition(transition_to_store)
            await manager.link_session_transition(self.session_id, transition.id)
            await manager.link_ingestion_emits_transition(
                self._ingest_version_id, transition.id
            )
            self._stats["transitions_added"] += 1
            self._seen_transition_ids.add(transition.id)

            source_type = str(getattr(transition, "source_type", "auto") or "auto")
            if source_type.startswith("manual"):
                self._stats["manual_transition_count"] += 1
            else:
                self._stats["auto_transition_count"] += 1

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
                    app_id=self.app_id,
                    from_state_id=_as_str(transition_to_store.from_state_id),
                    to_state_id=_as_str(transition_to_store.to_state_id),
                    action=action_value,
                    semantic_action_key=_as_str(
                        transition_to_store.semantic_action_key
                    )
                    or None,
                ),
                session_id=self.session_id,
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
                f"trev:{hashlib.md5(f'{stable_key}|{transition.id}|{self.session_id}|{datetime.now(UTC).isoformat()}'.encode()).hexdigest()[:16]}"
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
                session_id=self.session_id,
                ingest_version_id=self._ingest_version_id,
                is_active=should_activate,
            )
            await manager.add_transition_revision(revision)
            await manager.link_ingestion_emits_revision(
                self._ingest_version_id, revision.revision_id
            )
            if should_activate:
                await manager.activate_transition_revision(
                    stable_key,
                    revision.revision_id,
                    supersedes_ids,
                )
                self._active_revision_ids.append(revision.revision_id)
            else:
                await manager.attach_transition_revision(
                    stable_key, revision.revision_id
                )

        # --- Intents ---
        for transition in page_result.transitions:
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
                            session_id=self.session_id,
                        )

        # --- Evidence ---
        for transition in page_result.transitions:
            for evidence_id in transition.evidence_ids:
                evidence_type = (
                    EvidenceType.URL_CHANGE
                    if evidence_id.endswith("url_change")
                    else EvidenceType.DOM_DIFF
                )
                evidence = Evidence(
                    id=evidence_id,
                    transition_id=transition.id,
                    session_id=self.session_id,
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
                    ingest_version_id=self._ingest_version_id,
                )
                await manager.add_evidence(evidence)
                await manager.link_transition_evidence(transition.id, evidence.id)
                await manager.link_session_evidence(self.session_id, evidence.id)
                await manager.link_ingestion_emits_evidence(
                    self._ingest_version_id, evidence.id
                )

        # --- Layout evidence ---
        if getattr(page_result, "layout_evidence", None):
            layout_ev_base = len(self._accumulated_result.layout_evidence) if self._accumulated_result else 0
            for idx, ev in enumerate(page_result.layout_evidence):
                if not isinstance(ev, dict):
                    continue
                url = str(ev.get("url") or "")
                layout_fp = str(ev.get("layout_fingerprint") or "")
                if not layout_fp:
                    continue
                evidence_id = f"evidence:layout:{self.session_id}:{layout_ev_base + idx}:{layout_fp[:8]}"
                confidence = _as_float(ev.get("layout_confidence") or 0.5, 0.5)
                layout_evidence = Evidence(
                    id=evidence_id,
                    transition_id="",
                    session_id=self.session_id,
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
                    ingest_version_id=self._ingest_version_id,
                )
                await manager.add_evidence(layout_evidence)
                await manager.link_session_evidence(self.session_id, evidence_id)
                await manager.link_ingestion_emits_evidence(
                    self._ingest_version_id, evidence_id
                )

        # Accumulate for final stats
        if self._accumulated_result is not None:
            seen_sids = {s.id for s in self._accumulated_result.states}
            for state in page_result.states:
                if state.id not in seen_sids:
                    self._accumulated_result.states.append(state)
                    seen_sids.add(state.id)
            self._accumulated_result.transitions.extend(page_result.transitions)
            if page_result.history:
                self._accumulated_result.history.extend(page_result.history)
            if page_result.layout_evidence:
                self._accumulated_result.layout_evidence.extend(page_result.layout_evidence)

        logger.info(
            "[PERSIST] Page persisted: +states=%d +transitions=%d (total: states=%d transitions=%d)",
            sum(1 for s in page_result.states if s.id not in self._seen_state_ids),
            sum(1 for t in page_result.transitions if t.id not in self._seen_transition_ids),
            self._stats["states_added"],
            self._stats["transitions_added"],
        )

    @observe(
        name="cartography.finalize_persistence_session",
        metadata={"component": "cartography", "stage": "persistence"},
    )
    async def finalize(self, final_result: "CartographyResult") -> None:
        """Phase 3: Write menus, zones, release, coverage snapshot, session stats."""
        if not self._initialized or self._manager is None:
            logger.warning("[PERSIST] finalize called before init, skipping")
            return
        if self._finalized:
            return

        from graph_agent.models import (
            CoverageSnapshot,
            GraphRelease,
        )
        from graph_agent.coverage.analyzer import CoverageAnalyzer

        manager = self._manager

        # Merge final_result into accumulated for complete metrics
        if self._accumulated_result is not None and final_result is not None:
            seen_sids = {s.id for s in self._accumulated_result.states}
            for state in final_result.states:
                if state.id not in seen_sids:
                    self._accumulated_result.states.append(state)
                    seen_sids.add(state.id)
            seen_tids = {t.id for t in self._accumulated_result.transitions}
            for transition in final_result.transitions:
                if transition.id not in seen_tids:
                    self._accumulated_result.transitions.append(transition)
                    seen_tids.add(transition.id)
            if final_result.history:
                self._accumulated_result.history.extend(final_result.history)
            if final_result.layout_evidence:
                self._accumulated_result.layout_evidence.extend(final_result.layout_evidence)
            if final_result.menus:
                self._accumulated_result.menus = final_result.menus
            if final_result.zone_hints:
                self._accumulated_result.zone_hints = final_result.zone_hints
            if final_result.layout_metrics:
                self._accumulated_result.layout_metrics = final_result.layout_metrics
            if final_result.intervention_tasks:
                self._accumulated_result.intervention_tasks = final_result.intervention_tasks
            self._accumulated_result.semantic_conflict_count = final_result.semantic_conflict_count

        result = self._accumulated_result

        # --- Menus ---
        for i, menu in enumerate(result.menus):
            if not isinstance(menu, dict):
                continue
            text = _as_str(menu.get("text")).strip()
            href = _as_str(menu.get("href")).strip()
            level = _as_int(menu.get("level") or 0, 0)
            source_url = _as_str(menu.get("source_url")).strip()
            if not text and not href:
                continue
            menu_id_src = f"{self.app_id}|{source_url}|{level}|{text}|{href}"
            self._menu_rows.append(
                {
                    "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
                    "text": text or href,
                    "href": href,
                    "level": level,
                    "order": i,
                    "is_active": True,
                    "ingest_version_id": self._ingest_version_id,
                }
            )
        if self._menu_rows:
            await manager.add_menus(
                app_id=self.app_id,
                menus=self._menu_rows,
                page_url=self.current_url or self.resolved_url,
                session_id=self.session_id,
            )
            for menu in self._menu_rows:
                menu_id = _as_str(menu.get("id"))
                if menu_id:
                    await manager.link_ingestion_emits_menu(
                        self._ingest_version_id, menu_id
                    )

        # --- Zones ---
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
                zid_src = f"{self.app_id}|{z_type}|{selector}|{source_url_key}"
                zone_id = f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}"
                related_state_ids: set[str] = set()
                for key in (source_url, clean_url(source_url)):
                    if not key:
                        continue
                    related_state_ids.update(self._state_ids_by_url.get(key, set()))
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
                        "ingest_version_id": self._ingest_version_id,
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
                await manager.add_zones(app_id=self.app_id, zones=zone_rows)
                for zone in zone_rows:
                    zid = _as_str(zone.get("id"))
                    if zid:
                        await manager.link_ingestion_emits_zone(
                            self._ingest_version_id, zid
                        )

        # --- Menu-transition links ---
        if self._menu_rows:
            for transition in result.transitions:
                signal = f"{transition.selector} {transition.thought or ''}".lower()
                for menu in self._menu_rows:
                    text = str(menu.get("text") or "").strip().lower()
                    if text and text in signal:
                        await manager.link_transition_navigated_via(
                            transition.id, str(menu["id"])
                        )
                        break

        # --- Release ---
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
            f"release:{self.app_id}:{hashlib.md5(self._ingest_version_id.encode()).hexdigest()[:12]}"
        )
        await manager.add_graph_release(
            GraphRelease(
                id=release_id,
                app_id=self.app_id,
                base_ingest_ids=[self._ingest_version_id],
                status="active",
            )
        )
        await manager.deactivate_other_active_releases(
            app_id=self.app_id,
            keep_release_id=release_id,
        )
        for revision_id in self._active_revision_ids:
            await manager.link_release_revision(release_id, revision_id)

        # --- Coverage snapshot ---
        coverage_snapshot_id: str | None = None
        try:
            analyzer = CoverageAnalyzer(manager.get_driver())
            report = await analyzer.compute(app_id=self.app_id)
            coverage_snapshot_id = (
                f"cov:{self.session_id}:"
                f"{hashlib.md5(release_id.encode()).hexdigest()[:8]}"
            )
            snapshot = CoverageSnapshot(
                id=coverage_snapshot_id,
                app_id=self.app_id,
                session_id=self.session_id,
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
            await manager.link_session_coverage(self.session_id, coverage_snapshot_id)
            await manager.link_release_coverage(release_id, coverage_snapshot_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[PERSIST] coverage snapshot skipped: %s", e)

        # --- Semantic metrics ---
        self._stats["semantic_mismatch_warnings"] = int(
            getattr(result, "semantic_conflict_count", 0) or 0
        )
        semantic_metrics = _build_semantic_metrics(result)
        baseline_row = await manager.get_latest_semantic_baseline(
            app_id=self.app_id,
            exclude_session_id=self.session_id,
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

        # --- Update ingestion run status ---
        await manager.run_write_transaction([
            (
                "MATCH (ir:IngestionRun {id: $id}) SET ir.status = $status",
                {"id": self._ingest_version_id, "status": "completed"},
            )
        ])

        await manager.update_session_stats(
            session_id=self.session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": self.resolved_url,
                "current_release_id": release_id,
                "latest_ingest_version_id": self._ingest_version_id,
                "current_coverage_snapshot_id": coverage_snapshot_id or "",
                "intervention_task_count": len(
                    getattr(result, "intervention_tasks", []) or []
                ),
                "intervention_tasks": list(
                    getattr(result, "intervention_tasks", []) or []
                ),
                **self._stats,
                **layout_metrics_payload,
                **semantic_metrics,
                **semantic_stability_metrics,
            },
        )
        await manager.touch_app_last_session(self.app_id)
        await manager.update_app_stats(self.app_id)

        self._finalized = True
        logger.info(
            "[PERSIST] Session finalized: app_id=%s ingest=%s "
            "(states=%d, transitions=%d)",
            self.app_id, self._ingest_version_id,
            self._stats["states_added"],
            self._stats["transitions_added"],
        )

    async def close(self) -> None:
        """Close the underlying GraphManager connection."""
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("[PERSIST] Error closing manager: %s", e)
            self._manager = None


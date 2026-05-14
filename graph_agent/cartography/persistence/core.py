from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_agent.neo4j_client.manager import GraphManager

from graph_agent.cartography.config import clean_url
from graph_agent.lib.observability import observe
from graph_agent.cartography.persistence.semantic_stability import (
    _as_float,
    _as_int,
    _as_str,
    _empty_stats,
    _source_priority,
    _transition_stable_key,
)

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult

logger = logging.getLogger(__name__)


class _PersistenceSessionCore:
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
        inventory: list[dict[str, object]],
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

        self._manager: GraphManager | None = None
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
            f"ingest:{hashlib.md5(ingest_seed.encode(), usedforsecurity=False).hexdigest()[:16]}"
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
            login_fp = hashlib.md5(login_url.encode(), usedforsecurity=False).hexdigest()[:12]
            post_fp = hashlib.md5(post_login_url.encode(), usedforsecurity=False).hexdigest()[:12]
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
            revision_seed = f"{stable_key}|{transition.id}|{self.session_id}|{datetime.now(UTC).isoformat()}"
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


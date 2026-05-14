from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from neo4j import AsyncDriver

from graph_agent.neo4j_client.repository.core import _model_to_props
from graph_agent.neo4j_client.queries import CypherQueries

if TYPE_CHECKING:
    from graph_agent.models import (
        State, Intent, Entity, EntityInstance,
        Checkpoint, FieldConstraint, TestCase, Evidence, Session,
        IngestionRun, TransitionEntity, TransitionRevision,
        GraphRelease, CoverageSnapshot, Menu,
    )

logger = logging.getLogger(__name__)


class GraphRepositoryExtended:
    _driver: AsyncDriver  # provided by GraphRepositoryCore via multiple inheritance
    async def upsert_intent(self, intent: Intent) -> None:
        logger.info("[Neo4j] upsert Intent id=%s key=%s", intent.id, getattr(intent, "key", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_INTENT,
                id=intent.id, props=_model_to_props(intent),
            )

    async def link_transition_intent(self, transition_id: str, intent_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_INTENT,
                transition_id=transition_id, intent_id=intent_id,
            )

    async def link_intent_requires_entity(
        self, intent_id: str, entity_id: str, field_mapping: str = "{}"
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_REQUIRES_ENTITY,
                intent_id=intent_id, entity_id=entity_id,
                field_mapping=field_mapping,
            )

    async def link_intent_produces_entity(
        self, intent_id: str, entity_id: str, field_mapping: str = "{}"
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_PRODUCES_ENTITY,
                intent_id=intent_id, entity_id=entity_id,
                field_mapping=field_mapping,
            )

    async def link_intent_precondition(self, intent_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_PRECONDITION,
                intent_id=intent_id, checkpoint_id=checkpoint_id,
            )

    async def link_intent_postcondition(self, intent_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_POSTCONDITION,
                intent_id=intent_id, checkpoint_id=checkpoint_id,
            )

    async def link_intent_data_flow(
        self, from_intent_id: str, to_intent_id: str,
        from_slot: str = "", to_slot: str = "",
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_DATA_FLOW,
                from_intent_id=from_intent_id, to_intent_id=to_intent_id,
                from_slot=from_slot, to_slot=to_slot,
            )

    async def link_intent_alternative(self, intent_id_a: str, intent_id_b: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_ALTERNATIVE,
                intent_id_a=intent_id_a, intent_id_b=intent_id_b,
            )

    async def link_intent_must_precede(self, before_intent_id: str, after_intent_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INTENT_MUST_PRECEDE,
                before_intent_id=before_intent_id, after_intent_id=after_intent_id,
            )

    # ===== Entity =====

    async def upsert_entity(self, entity: Entity) -> None:
        logger.info("[Neo4j] upsert Entity id=%s name=%s", entity.id, getattr(entity, "name", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_ENTITY,
                id=entity.id, props=_model_to_props(entity),
            )

    async def upsert_entity_instance(self, instance: EntityInstance, entity_id: str) -> None:
        logger.info("[Neo4j] upsert EntityInstance id=%s entity=%s", instance.id, entity_id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_ENTITY_INSTANCE,
                id=instance.id, props=_model_to_props(instance),
                entity_id=entity_id,
            )

    async def link_entity_instance_intent(self, instance_id: str, intent_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_ENTITY_INSTANCE_INTENT,
                instance_id=instance_id, intent_id=intent_id,
            )

    # ===== Checkpoint =====

    async def upsert_checkpoint(self, checkpoint: Checkpoint) -> None:
        logger.info("[Neo4j] upsert Checkpoint id=%s rule_type=%s", checkpoint.id, getattr(checkpoint, "rule_type", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_CHECKPOINT,
                id=checkpoint.id, props=_model_to_props(checkpoint),
            )

    async def link_check_before(self, transition_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_CHECK_BEFORE,
                transition_id=transition_id, checkpoint_id=checkpoint_id,
            )

    async def link_check_after(self, transition_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_CHECK_AFTER,
                transition_id=transition_id, checkpoint_id=checkpoint_id,
            )

    async def link_state_check(self, state_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_STATE_CHECK,
                state_id=state_id, checkpoint_id=checkpoint_id,
            )

    async def link_checkpoint_entity(self, checkpoint_id: str, entity_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_CHECKPOINT_ENTITY,
                checkpoint_id=checkpoint_id, entity_id=entity_id,
            )

    # ===== Session =====

    async def upsert_session(self, sess: Session) -> None:
        logger.info("[Neo4j] upsert Session id=%s", sess.id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_SESSION,
                id=sess.id, props=_model_to_props(sess),
            )

    async def link_session_discovered_state(self, session_id: str, state_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_DISCOVERED_STATE,
                session_id=session_id, state_id=state_id,
            )

    async def link_session_discovered_transition(self, session_id: str, transition_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_DISCOVERED_TRANSITION,
                session_id=session_id, transition_id=transition_id,
            )

    async def upsert_ingestion_run(self, run: IngestionRun) -> None:
        logger.info("[Neo4j] upsert IngestionRun id=%s", run.id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_INGESTION_RUN,
                id=run.id,
                props=_model_to_props(run),
            )

    async def link_session_ingestion_run(self, session_id: str, ingest_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_INGESTION_RUN,
                session_id=session_id,
                ingest_id=ingest_id,
            )

    async def link_ingestion_emits_state(self, ingest_id: str, state_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_STATE,
                ingest_id=ingest_id,
                state_id=state_id,
            )

    async def link_ingestion_emits_transition(
        self, ingest_id: str, transition_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_TRANSITION,
                ingest_id=ingest_id,
                transition_id=transition_id,
            )

    async def link_ingestion_emits_evidence(self, ingest_id: str, evidence_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_EVIDENCE,
                ingest_id=ingest_id,
                evidence_id=evidence_id,
            )

    async def link_ingestion_emits_menu(self, ingest_id: str, menu_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_MENU,
                ingest_id=ingest_id,
                menu_id=menu_id,
            )

    async def link_ingestion_emits_zone(self, ingest_id: str, zone_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_ZONE,
                ingest_id=ingest_id,
                zone_id=zone_id,
            )

    async def upsert_transition_entity(self, entity: TransitionEntity) -> None:
        logger.info("[Neo4j] upsert TransitionEntity stable_key=%s", entity.stable_key)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_TRANSITION_ENTITY,
                stable_key=entity.stable_key,
                props=_model_to_props(entity),
            )

    async def upsert_transition_entity_with_session(
        self,
        entity: TransitionEntity,
        session_id: str,
    ) -> None:
        """Upsert TransitionEntity 并按 session 累计 confirmed_session_count。

        与普通 upsert 的区别：本方法对 ``confirmed_session_count`` 做去重累加
        （同一 session 重复提交不再 +1），用于 SkipAdvisor 跨 session 学习沉淀。
        """
        logger.info("[Neo4j] upsert TransitionEntity+session stable_key=%s session=%s", entity.stable_key, session_id)
        props = _model_to_props(entity)
        # 这些字段由 Cypher 自己负责递增/初始化，不从 props 覆盖。
        for k in (
            "confirmed_session_count",
            "last_confirm_session",
            "last_confirmed_at",
            "first_seen_session",
        ):
            props.pop(k, None)
        # 用空串补齐 ON MATCH 里 coalesce 取值时的 NULL 防御。
        props.setdefault("app_id", "")
        props.setdefault("from_state_id", "")
        props.setdefault("to_state_id", "")
        props.setdefault("action", "")
        props.setdefault("semantic_action_key", "")
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_TRANSITION_ENTITY_WITH_SESSION,
                stable_key=entity.stable_key,
                props=props,
                session_id=session_id,
                now=datetime.now(UTC).isoformat(),
            )

    async def upsert_coverage_snapshot(self, snapshot: CoverageSnapshot) -> None:
        logger.info("[Neo4j] upsert CoverageSnapshot id=%s", snapshot.id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_COVERAGE_SNAPSHOT,
                id=snapshot.id,
                props=_model_to_props(snapshot),
            )

    async def link_session_achieved_coverage(
        self, session_id: str, coverage_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_ACHIEVED_COVERAGE,
                session_id=session_id,
                coverage_id=coverage_id,
            )

    async def link_release_has_coverage(
        self, release_id: str, coverage_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_RELEASE_HAS_COVERAGE,
                release_id=release_id,
                coverage_id=coverage_id,
            )

    async def link_zone_covers_intent(
        self,
        *,
        from_state_id: str,
        selector: str,
        intent_id: str,
        confidence: float,
        session_id: str,
    ) -> None:
        if not from_state_id or not selector or not intent_id:
            return
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_ZONE_COVERS_INTENT,
                from_state_id=from_state_id,
                selector=selector,
                intent_id=intent_id,
                confidence=max(0.0, min(1.0, float(confidence or 0.0))),
                session_id=session_id or "",
                now=datetime.now(UTC).isoformat(),
            )

    async def upsert_transition_revision(self, revision: TransitionRevision) -> None:
        logger.info("[Neo4j] upsert TransitionRevision id=%s stable_key=%s", revision.revision_id, revision.stable_key)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_TRANSITION_REVISION,
                revision_id=revision.revision_id,
                props=_model_to_props(revision),
            )

    async def get_active_transition_revision(
        self, stable_key: str
    ) -> TransitionRevision | None:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_ACTIVE_REVISION_BY_STABLE_KEY,
                stable_key=stable_key,
            )
            record = await result.single()
            if record and record.get("rev") is not None:
                return TransitionRevision(**dict(record["rev"]))
            return None

    async def activate_transition_revision(
        self,
        stable_key: str,
        revision_id: str,
        supersedes_revision_ids: list[str] | None = None,
    ) -> None:
        supersedes_revision_ids = supersedes_revision_ids or []
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.DEACTIVATE_ACTIVE_REVISIONS,
                stable_key=stable_key,
            )
            await session.run(
                CypherQueries.LINK_ENTITY_HAS_REVISION,
                stable_key=stable_key,
                revision_id=revision_id,
            )
            for old_revision_id in supersedes_revision_ids:
                await session.run(
                    CypherQueries.LINK_REVISION_SUPERSEDES,
                    new_revision_id=revision_id,
                    old_revision_id=old_revision_id,
                )

    async def attach_transition_revision(self, stable_key: str, revision_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_ENTITY_HAS_REVISION,
                stable_key=stable_key,
                revision_id=revision_id,
            )

    async def link_ingestion_emits_revision(
        self, ingest_id: str, revision_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_INGESTION_EMITS_REVISION,
                ingest_id=ingest_id,
                revision_id=revision_id,
            )

    async def upsert_graph_release(self, release: GraphRelease) -> None:
        logger.info("[Neo4j] upsert GraphRelease id=%s", release.id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_GRAPH_RELEASE,
                id=release.id,
                props=_model_to_props(release),
            )

    async def link_release_revision(self, release_id: str, revision_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_RELEASE_INCLUDES_REVISION,
                release_id=release_id,
                revision_id=revision_id,
            )

    async def deactivate_other_active_releases(
        self,
        app_id: str,
        keep_release_id: str,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.DEACTIVATE_OTHER_ACTIVE_RELEASES,
                app_id=app_id,
                keep_release_id=keep_release_id,
                now=datetime.now(UTC).isoformat(),
            )

    async def link_session_generated_checkpoint(self, session_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_GENERATED_CHECKPOINT,
                session_id=session_id, checkpoint_id=checkpoint_id,
            )

    async def link_session_validated(self, session_id: str, transition_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_VALIDATED,
                session_id=session_id, transition_id=transition_id,
            )

    async def link_session_invalidated(self, session_id: str, transition_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_INVALIDATED,
                session_id=session_id, transition_id=transition_id,
            )

    async def link_session_created_entity_instance(self, session_id: str, instance_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_CREATED_ENTITY_INSTANCE,
                session_id=session_id, instance_id=instance_id,
            )

    # ===== TestCase =====

    async def upsert_test_case(self, tc: TestCase) -> None:
        logger.info("[Neo4j] upsert TestCase id=%s name=%s", tc.id, getattr(tc, "name", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_TEST_CASE,
                id=tc.id, props=_model_to_props(tc),
            )

    async def link_test_case_transition(self, test_case_id: str, transition_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TEST_CASE_TRANSITION,
                test_case_id=test_case_id, transition_id=transition_id,
            )

    async def link_test_case_checkpoint(self, test_case_id: str, checkpoint_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TEST_CASE_CHECKPOINT,
                test_case_id=test_case_id, checkpoint_id=checkpoint_id,
            )

    async def link_test_case_covers(self, test_case_id: str, field_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TEST_CASE_COVERS,
                test_case_id=test_case_id, field_id=field_id,
            )

    # ===== Evidence =====
    async def upsert_evidence(self, evidence: Evidence) -> None:
        logger.info("[Neo4j] upsert Evidence id=%s type=%s", evidence.id, getattr(evidence, "evidence_type", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_EVIDENCE,
                id=evidence.id,
                props=_model_to_props(evidence),
            )

    async def link_transition_evidence(self, transition_id: str, evidence_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_EVIDENCE,
                transition_id=transition_id,
                evidence_id=evidence_id,
            )

    async def link_session_evidence(self, session_id: str, evidence_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_EVIDENCE,
                session_id=session_id,
                evidence_id=evidence_id,
            )

    async def get_transition_evidence(self, transition_id: str) -> list[Evidence]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_TRANSITION_EVIDENCE,
                transition_id=transition_id,
            )
            records = await result.data()
            out: list[Evidence] = []
            for record in records:
                node = record.get("e")
                if node is None:
                    continue
                out.append(Evidence(**dict(node)))
            return out

    # ===== FieldConstraint =====

    async def upsert_field_constraint(self, fc: FieldConstraint, zone_id: str | None = None) -> None:
        logger.info("[Neo4j] upsert FieldConstraint id=%s field=%s", fc.id, getattr(fc, "field_name", ""))
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_FIELD_CONSTRAINT,
                id=fc.id, props=_model_to_props(fc),
            )
            if zone_id:
                await session.run(
                    CypherQueries.LINK_ZONE_FIELD,
                    zone_id=zone_id, field_id=fc.id,
                )

    async def link_transition_operates_on(self, transition_id: str, field_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_OPERATES_ON,
                transition_id=transition_id, field_id=field_id,
            )

    # ===== Menu Navigation (New Schema: Menu as separate node) =====

    async def upsert_menu(self, menu: Menu) -> None:
        logger.info("[Neo4j] upsert Menu id=%s", menu.id)
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_MENU,
                id=menu.id, props=_model_to_props(menu),
            )

    async def link_app_menu(self, app_id: str, menu_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_APP_MENU,
                app_id=app_id, menu_id=menu_id,
            )

    async def link_menu_child_of(
        self, child_id: str, parent_id: str, order_index: int = 0
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_MENU_CHILD_OF,
                child_id=child_id, parent_id=parent_id, order_index=order_index,
            )

    async def link_menu_leads_to(
        self, menu_id: str, state_id: str, session_id: str | None = None
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_MENU_LEADS_TO,
                menu_id=menu_id,
                state_id=state_id,
                first_seen=datetime.now(UTC).isoformat(),
                session_id=session_id or "",
            )

    async def link_transition_navigated_via(
        self, transition_id: str, menu_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_NAVIGATED_VIA,
                transition_id=transition_id, menu_id=menu_id,
            )

    async def link_session_discovered_menu(
        self, session_id: str, menu_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_SESSION_DISCOVERED_MENU,
                session_id=session_id, menu_id=menu_id,
            )

    async def get_menu(self, menu_id: str) -> Menu | None:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_MENU_BY_ID, id=menu_id)
            record = await result.single()
            if record:
                return Menu(**dict(record["m"]))
        return None

    async def get_app_menus(self, app_id: str) -> list[Menu]:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_APP_MENUS, app_id=app_id)
            records = await result.data()
            return [Menu(**dict(r["m"])) for r in records]

    async def get_menu_tree(self, app_id: str) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_MENU_TREE, app_id=app_id)
            records = await result.data()
            return [
                {
                    "menu": Menu(**dict(r["m"])),
                    "parent_id": r.get("parent_id"),
                    "state_id": r.get("state_id"),
                }
                for r in records
            ]

    async def mark_page_menus_inactive(self, app_id: str, page_url: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.MARK_PAGE_MENUS_INACTIVE,
                app_id=app_id,
                page_url=page_url,
                inactive_at=datetime.now(UTC).isoformat(),
            )

    # ===== Legacy Menu Navigation (Deprecated) =====

    async def link_menu_parent(self, child_id: str, parent_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_MENU_PARENT_LEGACY,
                child_id=child_id, parent_id=parent_id,
            )

    async def link_menu_next(self, state_id: str, next_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_MENU_NEXT_LEGACY,
                state_id=state_id, next_id=next_id,
            )

    # ===== Graph operations =====

    async def get_shortest_path(self, from_state_id: str, to_state_id: str) -> list[State] | None:
        """Find shortest path between two states."""
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.SHORTEST_PATH,
                from_id=from_state_id, to_id=to_state_id,
            )
            record = await result.single()
            if record and record.get("path"):
                nodes = record["path"].nodes
                return [State(**dict(n)) for n in nodes if n.labels == {"State"}]
            return None

    async def get_state_count(self) -> int:
        """Get total number of states."""
        async with self._driver.session() as session:
            result = await session.run("MATCH (s:State) RETURN count(s) as cnt")
            record = await result.single()
            return record["cnt"] if record else 0

    async def get_transition_count(self) -> int:
        """Get total number of transitions."""
        async with self._driver.session() as session:
            result = await session.run("MATCH (t:Transition) RETURN count(t) as cnt")
            record = await result.single()
            return record["cnt"] if record else 0

    async def get_app_states(self, app_id: str) -> list[State]:
        """Get all states for an app."""
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_APP_STATES, app_id=app_id)
            records = await result.data()
            return [State(**dict(r["s"])) for r in records]

    async def delete_app_data(self, app_id: str) -> None:
        """Delete all data for an app."""
        async with self._driver.session() as session:
            # Delete transitions first (to avoid orphan edges)
            await session.run("""
                MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)
                DETACH DELETE t
            """, app_id=app_id)
            # Delete states
            await session.run("""
                MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
                DETACH DELETE s
            """, app_id=app_id)
            # Delete app
            await session.run("""
                MATCH (a:App {id: $app_id})
                DELETE a
            """, app_id=app_id)

    async def clear_all(self) -> None:
        """Clear all graph data (use with caution)."""
        async with self._driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

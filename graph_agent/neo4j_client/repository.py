from __future__ import annotations
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from neo4j import AsyncDriver
from graph_agent.models import (
    App, State, Transition, Zone, FrameNode, Entity, EntityInstance,
    Intent, Checkpoint, FieldConstraint, TestCase, Session, Menu, Evidence,
    GraphRelease, IngestionRun, TransitionEntity, TransitionRevision,
    CoverageSnapshot,
)
from graph_agent.neo4j_client.queries import CypherQueries

logger = logging.getLogger(__name__)


def _model_to_props(model: Any) -> dict[str, Any]:
    """Convert Pydantic model to Neo4j-compatible property dict."""
    data = model.model_dump(exclude_none=True, exclude={"from_state_id", "to_state_id"})
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, list):
            result[k] = v
        elif isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif v is not None:
            result[k] = str(v)
    return result


class GraphRepository:
    """High-level repository for graph CRUD operations."""

    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def execute_batch(
        self,
        operations: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Execute multiple write operations in one transaction."""
        if not operations:
            return
        async with self._driver.session() as session:
            async def _tx_write(tx):
                for query, params in operations:
                    await tx.run(query, **params)
            await session.execute_write(_tx_write)

    # ===== App =====

    async def upsert_app(self, app: App) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_APP,
                id=app.id, props=_model_to_props(app),
            )

    async def get_app(self, app_id: str) -> App | None:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_APP_BY_ID, id=app_id)
            record = await result.single()
            if record:
                return App(**dict(record["a"]))
        return None

    async def get_all_apps(self) -> list[App]:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_ALL_APPS)
            records = await result.data()
            return [App(**dict(r["a"])) for r in records]

    async def link_app_state(self, app_id: str, state_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_APP_STATE,
                app_id=app_id, state_id=state_id,
            )

    async def link_app_session(self, app_id: str, session_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_APP_SESSION,
                app_id=app_id, session_id=session_id,
            )

    async def touch_app_last_session(
        self,
        app_id: str,
        last_session_at: datetime | None = None,
    ) -> None:
        ts = (last_session_at or datetime.now(UTC)).isoformat()
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.TOUCH_APP_LAST_SESSION,
                app_id=app_id,
                last_session_at=ts,
            )

    async def update_app_stats(self, app_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(CypherQueries.UPDATE_APP_STATS, app_id=app_id)

    # ===== State =====

    async def upsert_state(self, state: State) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_STATE,
                id=state.id, props=_model_to_props(state),
            )

    async def get_state(self, state_id: str) -> State | None:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_STATE_BY_ID, id=state_id)
            record = await result.single()
            if record:
                return State(**dict(record["s"]))
        return None

    async def get_states_by_url(self, url: str) -> list[State]:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_STATE_BY_URL, url=url)
            records = await result.data()
            return [State(**dict(r["s"])) for r in records]

    async def get_all_states(self) -> list[State]:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.GET_ALL_STATES)
            records = await result.data()
            return [State(**dict(r["s"])) for r in records]

    # ===== Transition =====

    async def upsert_transition(
        self, transition: Transition, from_state_id: str, to_state_id: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_TRANSITION,
                id=transition.id,
                props=_model_to_props(transition),
                from_state_id=from_state_id,
                to_state_id=to_state_id,
            )

    async def get_transitions_from_state(self, state_id: str) -> list[tuple[Transition, State]]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_TRANSITIONS_FROM_STATE, state_id=state_id
            )
            records = await result.data()
            pairs = []
            for r in records:
                t = Transition(**dict(r["t"]))
                target = State(**dict(r["target"]))
                pairs.append((t, target))
            return pairs

    async def get_all_transitions(self, app_id: str) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_ALL_TRANSITIONS, app_id=app_id
            )
            return await result.data()

    async def get_all_transitions_by_release(
        self, app_id: str, release_id: str
    ) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_ALL_TRANSITIONS_BY_RELEASE,
                app_id=app_id,
                release_id=release_id,
            )
            return await result.data()

    async def get_states_with_intents(self, app_id: str) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.GET_STATES_WITH_INTENTS, app_id=app_id
            )
            return await result.data()

    # ===== Zone =====

    async def upsert_zone(self, zone: Zone, state_id: str | None = None) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_ZONE,
                id=zone.id, props=_model_to_props(zone),
            )
            if state_id:
                await session.run(
                    CypherQueries.LINK_STATE_ZONE,
                    state_id=state_id, zone_id=zone.id,
                )

    async def link_transition_zone(self, transition_id: str, zone_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_ZONE,
                transition_id=transition_id, zone_id=zone_id,
            )

    async def upsert_zones_for_app(
        self,
        app_id: str,
        zones: list[dict[str, Any]],
        state_id: str | None = None,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_ZONES_FOR_APP,
                app_id=app_id,
                zones=zones,
            )
            if state_id:
                await session.run(
                    CypherQueries.LINK_STATE_ZONES_DIRECT,
                    zones=zones,
                    state_id=state_id,
                )
            await session.run(
                CypherQueries.LINK_STATE_ZONES_BY_IDS,
                zones=zones,
            )

    async def get_skip_advisor_coverage(
        self,
        *,
        url_clean: str,
        app_id: str = "",
        app_name: str = "",
    ) -> dict[str, Any]:
        async with self._driver.session() as session:
            if app_id:
                base_query = CypherQueries.SKIP_ADVISOR_BASE_BY_APP_ID
                entity_query = CypherQueries.SKIP_ADVISOR_ENTITY_BY_APP_ID
                params = {"app_id": app_id, "url_clean": url_clean}
            else:
                base_query = CypherQueries.SKIP_ADVISOR_BASE_BY_APP_NAME
                entity_query = CypherQueries.SKIP_ADVISOR_ENTITY_BY_APP_NAME
                params = {"app_name": app_name, "url_clean": url_clean}

            base_result = await session.run(base_query, **params)
            base_row = await base_result.single()
            if not base_row:
                return {
                    "state_count": 0,
                    "last_visited": None,
                    "zones": [],
                    "release_coverage": 0.0,
                    "entity_confirm_total": 0,
                }
            row = dict(base_row)
            entity_result = await session.run(entity_query, **params)
            entity_row = await entity_result.single()
            row["entity_confirm_total"] = (
                int(entity_row.get("entity_confirm_total") or 0) if entity_row else 0
            )
            return row

    async def get_knowledge_release_rows(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        release_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            if app_id:
                query = CypherQueries.KNOWLEDGE_RELEASE_ROWS_BY_APP_ID
                params = {"app_id": app_id, "release_id": release_id, "limit": limit}
            else:
                query = CypherQueries.KNOWLEDGE_RELEASE_ROWS_BY_APP_NAME
                params = {
                    "app_name": app_name,
                    "release_id": release_id,
                    "limit": limit,
                }
            result = await session.run(query, **params)
            return await result.data()

    async def get_knowledge_legacy_rows(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        limit: int,
    ) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            if app_id:
                query = CypherQueries.KNOWLEDGE_LEGACY_ROWS_BY_APP_ID
                params = {"app_id": app_id, "limit": limit}
            else:
                query = CypherQueries.KNOWLEDGE_LEGACY_ROWS_BY_APP_NAME
                params = {"app_name": app_name, "limit": limit}
            result = await session.run(query, **params)
            return await result.data()

    async def get_runner_warm_start_candidates(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            if app_id:
                result = await session.run(
                    CypherQueries.RUNNER_WARM_START_CANDIDATES_BY_APP_ID,
                    app_id=app_id,
                    limit=limit,
                )
            else:
                result = await session.run(
                    CypherQueries.RUNNER_WARM_START_CANDIDATES_BY_APP_NAME,
                    app_name=app_name,
                    limit=limit,
                )
            return await result.data()

    async def get_state_url(self, sid: str) -> str:
        async with self._driver.session() as session:
            result = await session.run(CypherQueries.RUNNER_STATE_URL_BY_ID, sid=sid)
            record = await result.single()
            return str(record.get("url") or "") if record else ""

    async def get_latest_active_release_by_app_name(self, app_name: str) -> str:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.RUNNER_ACTIVE_RELEASE_BY_APP_NAME,
                app_name=app_name,
            )
            record = await result.single()
            return str(record.get("id") or "") if record else ""

    async def get_latest_active_release_by_app_id(self, app_id: str) -> str:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.RUNNER_ACTIVE_RELEASE_BY_APP_ID,
                app_id=app_id,
            )
            record = await result.single()
            return str(record.get("id") or "") if record else ""

    async def get_latest_semantic_baseline(
        self,
        *,
        app_id: str,
        exclude_session_id: str,
    ) -> dict[str, Any]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.SESSION_LATEST_SEMANTIC_BASELINE_BY_APP_ID,
                app_id=app_id,
                exclude_session_id=exclude_session_id,
            )
            row = await result.single()
            return dict(row) if row else {}

    async def get_semantic_stability_trend(
        self,
        *,
        app_id: str,
        limit: int = 10,
        default_threshold: float = 90.0,
    ) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(
                CypherQueries.SESSION_SEMANTIC_STABILITY_TREND_BY_APP_ID,
                app_id=app_id,
                limit=max(1, int(limit)),
                default_threshold=float(default_threshold),
            )
            return await result.data()

    async def plan_retention(
        self,
        *,
        app_id: str,
        keep_releases: int = 5,
        min_age_days: int = 14,
    ) -> dict[str, Any]:
        keep_n = max(1, int(keep_releases))
        min_days = max(0, int(min_age_days))
        before_ts = (datetime.now(UTC) - timedelta(days=min_days)).isoformat()
        async with self._driver.session() as session:
            keep_result = await session.run(
                CypherQueries.RETENTION_KEEP_RELEASE_IDS_BY_APP,
                app_id=app_id,
                keep_releases=keep_n,
            )
            keep_row = await keep_result.single()
            keep_release_ids = (
                list(keep_row.get("keep_release_ids") or []) if keep_row else []
            )

            candidate_result = await session.run(
                CypherQueries.RETENTION_CANDIDATE_RELEASE_IDS_BY_APP,
                app_id=app_id,
                before_ts=before_ts,
                keep_release_ids=keep_release_ids,
            )
            candidate_row = await candidate_result.single()
            candidate_release_ids = (
                list(candidate_row.get("candidate_release_ids") or [])
                if candidate_row
                else []
            )

            revisions_result = await session.run(
                CypherQueries.RETENTION_COUNT_REVISIONS_FOR_RELEASES,
                release_ids=candidate_release_ids,
            )
            revisions_row = await revisions_result.single()

            sessions_result = await session.run(
                CypherQueries.RETENTION_COUNT_SESSIONS_FOR_RELEASES,
                app_id=app_id,
                release_ids=candidate_release_ids,
                before_ts=before_ts,
            )
            sessions_row = await sessions_result.single()

            ingests_result = await session.run(
                CypherQueries.RETENTION_COUNT_INGEST_RUNS_FOR_RELEASE_SESSIONS,
                app_id=app_id,
                release_ids=candidate_release_ids,
                before_ts=before_ts,
            )
            ingests_row = await ingests_result.single()

            active_result = await session.run(
                CypherQueries.RETENTION_COUNT_ACTIVE_RELEASES_BY_APP,
                app_id=app_id,
            )
            active_row = await active_result.single()

        return {
            "app_id": app_id,
            "before_ts": before_ts,
            "keep_releases": keep_n,
            "min_age_days": min_days,
            "keep_release_ids": keep_release_ids,
            "candidate_release_ids": candidate_release_ids,
            "candidate_counts": {
                "releases": len(candidate_release_ids),
                "inactive_revisions": int(revisions_row.get("count") or 0)
                if revisions_row
                else 0,
                "sessions": int(sessions_row.get("count") or 0) if sessions_row else 0,
                "ingestion_runs": int(ingests_row.get("count") or 0)
                if ingests_row
                else 0,
            },
            "protected_counts": {
                "active_releases": int(active_row.get("active_count") or 0)
                if active_row
                else 0,
            },
        }

    async def _delete_in_batches(
        self,
        query: str,
        *,
        batch_size: int,
        params: dict[str, Any],
    ) -> int:
        deleted_total = 0
        while True:
            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    batch_size=max(1, int(batch_size)),
                    **params,
                )
                row = await result.single()
            deleted = int(row.get("deleted_count") or 0) if row else 0
            deleted_total += deleted
            if deleted <= 0:
                break
        return deleted_total

    async def execute_retention(
        self,
        *,
        app_id: str,
        keep_releases: int = 5,
        min_age_days: int = 14,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        plan = await self.plan_retention(
            app_id=app_id,
            keep_releases=keep_releases,
            min_age_days=min_age_days,
        )
        release_ids = list(plan.get("candidate_release_ids") or [])
        before_ts = str(plan.get("before_ts") or datetime.now(UTC).isoformat())
        deleted_counts = {
            "sessions": 0,
            "releases": 0,
            "inactive_revisions": 0,
            "orphan_coverage_snapshots": 0,
            "orphan_ingestion_runs": 0,
            "orphan_evidence": 0,
        }
        if release_ids:
            deleted_counts["sessions"] = await self._delete_in_batches(
                CypherQueries.RETENTION_DELETE_SESSIONS_BY_RELEASE_IDS,
                batch_size=batch_size,
                params={
                    "app_id": app_id,
                    "release_ids": release_ids,
                    "before_ts": before_ts,
                },
            )
            deleted_counts["releases"] = await self._delete_in_batches(
                CypherQueries.RETENTION_DELETE_RELEASES_BY_IDS,
                batch_size=batch_size,
                params={"release_ids": release_ids},
            )
        deleted_counts["inactive_revisions"] = await self._delete_in_batches(
            CypherQueries.RETENTION_DELETE_ORPHAN_INACTIVE_REVISIONS,
            batch_size=batch_size,
            params={},
        )
        deleted_counts["orphan_coverage_snapshots"] = await self._delete_in_batches(
            CypherQueries.RETENTION_DELETE_ORPHAN_COVERAGE_SNAPSHOTS,
            batch_size=batch_size,
            params={"before_ts": before_ts},
        )
        deleted_counts["orphan_ingestion_runs"] = await self._delete_in_batches(
            CypherQueries.RETENTION_DELETE_ORPHAN_INGEST_RUNS,
            batch_size=batch_size,
            params={"before_ts": before_ts},
        )
        deleted_counts["orphan_evidence"] = await self._delete_in_batches(
            CypherQueries.RETENTION_DELETE_ORPHAN_EVIDENCE,
            batch_size=batch_size,
            params={"before_ts": before_ts},
        )

        post_plan = await self.plan_retention(
            app_id=app_id,
            keep_releases=keep_releases,
            min_age_days=min_age_days,
        )
        return {
            **plan,
            "deleted_counts": deleted_counts,
            "post_candidate_counts": post_plan.get("candidate_counts", {}),
            "protected_counts": post_plan.get("protected_counts", {}),
        }

    # ===== Frame =====

    async def upsert_frame(
        self, frame: FrameNode, state_id: str | None = None
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_FRAME,
                id=frame.id, props=_model_to_props(frame),
            )
            if state_id:
                await session.run(
                    CypherQueries.LINK_STATE_FRAME,
                    state_id=state_id, frame_id=frame.id, depth=frame.depth,
                )
            if frame.parent_frame_id:
                await session.run(
                    CypherQueries.LINK_FRAME_PARENT,
                    parent_id=frame.parent_frame_id, child_id=frame.id,
                )

    async def link_transition_frame(self, transition_id: str, frame_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.LINK_TRANSITION_FRAME,
                transition_id=transition_id, frame_id=frame_id,
            )

    # ===== Intent =====

    async def upsert_intent(self, intent: Intent) -> None:
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
        async with self._driver.session() as session:
            await session.run(
                CypherQueries.UPSERT_ENTITY,
                id=entity.id, props=_model_to_props(entity),
            )

    async def upsert_entity_instance(self, instance: EntityInstance, entity_id: str) -> None:
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

    # ===== Graph Operations (for networkx migration) =====

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

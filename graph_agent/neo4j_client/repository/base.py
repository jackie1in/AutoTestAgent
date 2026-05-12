from __future__ import annotations
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from neo4j import AsyncDriver
from graph_agent.models import (
    App, State, Transition, Zone, FrameNode, Entity, EntityInstance,
    Intent,
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
        logger.info("[Neo4j] upsert App id=%s name=%s", app.id, getattr(app, "name", ""))
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
        logger.info("[Neo4j] upsert State id=%s url=%s", state.id, getattr(state, "url", ""))
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
        logger.info(
            "[Neo4j] upsert Transition id=%s action=%s %s→%s",
            transition.id, getattr(transition, "action", ""), from_state_id[:24], to_state_id[:24],
        )
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
        logger.info("[Neo4j] upsert Zone id=%s type=%s", zone.id, getattr(zone, "zone_type", ""))
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
        logger.info("[Neo4j] upsert Zones for app=%s count=%d", app_id, len(zones))
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
        logger.info("[Neo4j] upsert Frame id=%s", frame.id)
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

"""GraphManager provides high-level graph operations backed by Neo4j."""

from __future__ import annotations

import json
from typing import Any, LiteralString, cast
from graph_agent.neo4j_client.driver import Neo4jDriver
from graph_agent.neo4j_client.repository import GraphRepository
from graph_agent.models import (
    App,
    CoverageSnapshot,
    GraphRelease,
    IngestionRun,
    Intent,
    Session,
    State,
    Transition,
    TransitionEntity,
    TransitionRevision,
    Evidence,
)


class GraphManager:
    """High-level manager for Neo4j graph operations.
    
    Provides a simplified interface for creating and managing
    apps, sessions, states, transitions, and their relationships.
    """
    
    def __init__(self) -> None:
        self._driver = Neo4jDriver()
        self._repo: GraphRepository | None = None
    
    async def __aenter__(self) -> GraphManager:
        await self._driver.connect()
        await self._driver.ensure_schema()
        self._repo = GraphRepository(self._driver.driver)
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._driver.close()
    
    async def _run_write(self, query: str, **params) -> None:
        """Execute a write query."""
        async with self._driver.driver.session() as session:
            await session.run(cast(LiteralString, query), **params)

    async def _run_read(self, query: str, **params) -> list[Any]:
        """Execute a read query and return records."""
        async with self._driver.driver.session() as session:
            result = await session.run(cast(LiteralString, query), **params)
            return await result.data()

    async def run_read(self, query: str, **params) -> list[Any]:
        """Public readonly query API for business modules."""
        return await self._run_read(query, **params)

    def get_driver(self):
        """Expose connected Neo4j driver through a stable API."""
        return self._driver.driver

    async def run_write_transaction(
        self,
        operations: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Run batched write operations in one transaction."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.execute_batch(operations)
    
    # App operations
    async def add_app(self, app: App) -> None:
        """Add or update an app."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_app(app)
    
    # Session operations
    async def add_session(self, session: Session) -> None:
        """Add or update a session."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_session(session)

    async def add_ingestion_run(self, run: IngestionRun) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_ingestion_run(run)

    async def link_session_ingestion_run(self, session_id: str, ingest_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_session_ingestion_run(session_id, ingest_id)
    
    async def link_session_discovered(self, session_id: str, state_id: str) -> None:
        """Link a session to a discovered state."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_session_discovered_state(session_id, state_id)
    
    async def link_session_transition(self, session_id: str, transition_id: str) -> None:
        """Link a session to a transition."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_session_discovered_transition(session_id, transition_id)

    async def link_ingestion_emits_state(self, ingest_id: str, state_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_state(ingest_id, state_id)

    async def link_ingestion_emits_transition(self, ingest_id: str, transition_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_transition(ingest_id, transition_id)

    async def link_ingestion_emits_evidence(self, ingest_id: str, evidence_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_evidence(ingest_id, evidence_id)

    async def link_ingestion_emits_menu(self, ingest_id: str, menu_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_menu(ingest_id, menu_id)

    async def link_ingestion_emits_zone(self, ingest_id: str, zone_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_zone(ingest_id, zone_id)
    
    async def link_app_state(self, app_id: str, state_id: str) -> None:
        """Link an app to a state."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_app_state(app_id, state_id)

    async def link_app_session(self, app_id: str, session_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_app_session(app_id, session_id)

    async def touch_app_last_session(
        self,
        app_id: str,
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.touch_app_last_session(app_id)

    async def update_app_stats(self, app_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.update_app_stats(app_id)
    
    async def set_session_inventory(self, session_id: str, inventory: dict) -> None:
        """Store inventory data with a session."""
        from graph_agent.neo4j_client.queries import CypherQueries

        query = CypherQueries.SET_SESSION_INVENTORY
        await self._run_write(
            query, 
            session_id=session_id, 
            inventory=inventory
        )
    
    async def update_session_stats(
        self, 
        session_id: str, 
        stats: dict[str, Any]
    ) -> None:
        """Update session with various stats."""
        raw_tasks = stats.get("intervention_tasks", [])
        task_ids: list[str] = []
        if isinstance(raw_tasks, list):
            for task in raw_tasks:
                if isinstance(task, dict):
                    tid = str(task.get("task_id") or "").strip()
                    if tid:
                        task_ids.append(tid)
                elif task is not None:
                    text = str(task).strip()
                    if text:
                        task_ids.append(text)
        tasks_json = json.dumps(raw_tasks, ensure_ascii=False)
        dynamic_prefixes = ("layout_", "knowledge_", "skip_", "semantic_")
        core_keys = {
            "visited_urls",
            "mapping_stopped",
            "stop_reason",
            "start_url",
            "states_added",
            "transitions_added",
            "filtered_non_ui_edges",
            "semantic_mismatch_warnings",
            "url_discontinuity_warnings",
            "frame_context_transition_warnings",
            "manual_transition_count",
            "auto_transition_count",
            "intervention_task_count",
            "intervention_tasks",
            "current_release_id",
            "latest_ingest_version_id",
            "current_coverage_snapshot_id",
        }
        dynamic_stats: dict[str, Any] = {}
        for key, value in stats.items():
            if key in core_keys:
                continue
            if not key.startswith(dynamic_prefixes):
                continue
            if isinstance(value, (str, int, float, bool, list)) or value is None:
                dynamic_stats[key] = value
            elif isinstance(value, dict):
                dynamic_stats[key] = json.dumps(value, ensure_ascii=False)
            else:
                dynamic_stats[key] = str(value)
        dynamic_metrics_json = json.dumps(dynamic_stats, ensure_ascii=False)

        from graph_agent.neo4j_client.queries import CypherQueries

        await self._run_write(
            CypherQueries.UPDATE_SESSION_STATS,
            session_id=session_id,
            visited_urls=stats.get("visited_urls", []),
            mapping_stopped=stats.get("mapping_stopped", False),
            stop_reason=stats.get("stop_reason"),
            start_url=stats.get("start_url", ""),
            states_added=stats.get("states_added", 0),
            transitions_added=stats.get("transitions_added", 0),
            filtered_non_ui_edges=stats.get("filtered_non_ui_edges", 0),
            semantic_mismatch_warnings=stats.get("semantic_mismatch_warnings", 0),
            url_discontinuity_warnings=stats.get("url_discontinuity_warnings", 0),
            frame_context_transition_warnings=stats.get("frame_context_transition_warnings", 0),
            manual_transition_count=stats.get("manual_transition_count", 0),
            auto_transition_count=stats.get("auto_transition_count", 0),
            intervention_task_count=stats.get("intervention_task_count", 0),
            intervention_tasks=task_ids,
            intervention_tasks_json=tasks_json,
            current_release_id=stats.get("current_release_id", ""),
            latest_ingest_version_id=stats.get("latest_ingest_version_id", ""),
            current_coverage_snapshot_id=stats.get("current_coverage_snapshot_id", ""),
            dynamic_metrics_json=dynamic_metrics_json,
            dynamic_stats=dynamic_stats,
        )
    
    # State operations
    async def add_state(self, state: State) -> None:
        """Add or update a state."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_state(state)
    
    # Transition operations
    async def add_transition(self, transition: Transition) -> None:
        """Add or update a transition."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_transition(
            transition,
            transition.from_state_id or "",
            transition.to_state_id or "",
        )

    async def add_transition_entity(self, entity: TransitionEntity) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_transition_entity(entity)

    async def add_transition_entity_with_session(
        self,
        entity: TransitionEntity,
        session_id: str,
    ) -> None:
        """累积 confirmed_session_count（按 session 去重）。

        SkipAdvisor 在 _query_coverage 里读 ent.confirmed_session_count，
        作为"该 transition 已被多少个独立 session 复现"的强信号。
        """
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_transition_entity_with_session(entity, session_id)

    async def add_coverage_snapshot(self, snapshot: CoverageSnapshot) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_coverage_snapshot(snapshot)

    async def link_session_coverage(
        self, session_id: str, coverage_id: str
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_session_achieved_coverage(session_id, coverage_id)

    async def link_release_coverage(
        self, release_id: str, coverage_id: str
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_release_has_coverage(release_id, coverage_id)

    async def add_intent(self, intent: Intent) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_intent(intent)

    async def link_transition_intent(self, transition_id: str, intent_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_transition_intent(transition_id, intent_id)

    async def link_zone_covers_intent(
        self,
        *,
        from_state_id: str,
        selector: str,
        intent_id: str,
        confidence: float,
        session_id: str,
    ) -> None:
        """建立 (:Zone)-[:COVERS_INTENT {observed_count, confidence}]->(:Intent)。

        匹配规则：与 from_state 相连的 Zone 中，selector 与 transition.selector
        互为子串关系的 zone 视为命中。每次命中 observed_count +1。
        """
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_zone_covers_intent(
            from_state_id=from_state_id,
            selector=selector,
            intent_id=intent_id,
            confidence=confidence,
            session_id=session_id,
        )

    async def add_transition_revision(self, revision: TransitionRevision) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_transition_revision(revision)

    async def get_active_transition_revision(
        self, stable_key: str
    ) -> TransitionRevision | None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_active_transition_revision(stable_key)

    async def activate_transition_revision(
        self,
        stable_key: str,
        revision_id: str,
        supersedes_revision_ids: list[str] | None = None,
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.activate_transition_revision(
            stable_key,
            revision_id,
            supersedes_revision_ids,
        )

    async def attach_transition_revision(self, stable_key: str, revision_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.attach_transition_revision(stable_key, revision_id)

    async def link_ingestion_emits_revision(self, ingest_id: str, revision_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_ingestion_emits_revision(ingest_id, revision_id)

    async def add_evidence(self, evidence: Evidence) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_evidence(evidence)

    async def link_transition_evidence(self, transition_id: str, evidence_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_transition_evidence(transition_id, evidence_id)

    async def link_session_evidence(self, session_id: str, evidence_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_session_evidence(session_id, evidence_id)

    async def get_transition_evidence(self, transition_id: str) -> list[Evidence]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_transition_evidence(transition_id)

    async def link_transition_navigated_via(
        self, transition_id: str, menu_id: str
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_transition_navigated_via(transition_id, menu_id)
    
    # Query operations
    async def get_all_transitions(self, app_id: str) -> list[dict]:
        """Get all transitions for an app."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_all_transitions(app_id)

    async def get_all_transitions_for_release(
        self, app_id: str, release_id: str | None = None
    ) -> list[dict]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        if release_id:
            rows = await self._repo.get_all_transitions_by_release(app_id, release_id)
            if rows:
                return rows
        return await self._repo.get_all_transitions(app_id)
    
    async def get_states_with_intents(self, app_id: str) -> list[dict]:
        """Get all states and their intents for an app."""
        query = """
        MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
        OPTIONAL MATCH (s)<-[:FROM]-(t:Transition)-[:REALIZES]->(i:Intent)
        RETURN s.id as state_id, s.url as url, s.title as title,
               collect(DISTINCT i{.*}) as intents
        """
        result = await self._run_read(query, app_id=app_id)
        return [record for record in result]

    async def add_graph_release(self, release: GraphRelease) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_graph_release(release)

    async def link_release_revision(self, release_id: str, revision_id: str) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_release_revision(release_id, revision_id)

    async def deactivate_other_active_releases(
        self,
        app_id: str,
        keep_release_id: str,
    ) -> None:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.deactivate_other_active_releases(
            app_id=app_id,
            keep_release_id=keep_release_id,
        )
    
    # Zone operations
    async def add_zones(
        self,
        app_id: str,
        zones: list[dict],
        state_id: str | None = None,
    ) -> None:
        """Add zones to an app with proper PRD relationships.

        Creates:
        - (:State)-[:HAS_ZONE]->(:Zone) when state_id is provided
        - (:App)-[:HAS_ZONE]->(:Zone) as aggregate link
        """
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.upsert_zones_for_app(
            app_id=app_id,
            zones=zones,
            state_id=state_id,
        )

    async def get_skip_advisor_coverage(
        self,
        *,
        url_clean: str,
        app_id: str = "",
        app_name: str = "",
    ) -> dict[str, Any]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_skip_advisor_coverage(
            url_clean=url_clean,
            app_id=app_id,
            app_name=app_name,
        )

    async def get_knowledge_release_rows(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        release_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_knowledge_release_rows(
            app_id=app_id,
            app_name=app_name,
            release_id=release_id,
            limit=limit,
        )

    async def get_knowledge_legacy_rows(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        limit: int,
    ) -> list[dict[str, Any]]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_knowledge_legacy_rows(
            app_id=app_id,
            app_name=app_name,
            limit=limit,
        )

    async def get_runner_warm_start_candidates(
        self,
        *,
        app_id: str = "",
        app_name: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_runner_warm_start_candidates(
            app_id=app_id,
            app_name=app_name,
            limit=limit,
        )

    async def get_state_url(self, sid: str) -> str:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_state_url(sid)

    async def get_latest_active_release_by_app_name(self, app_name: str) -> str:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_latest_active_release_by_app_name(app_name)

    async def get_latest_active_release_by_app_id(self, app_id: str) -> str:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_latest_active_release_by_app_id(app_id)

    async def get_latest_semantic_baseline(
        self,
        *,
        app_id: str,
        exclude_session_id: str,
    ) -> dict[str, Any]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_latest_semantic_baseline(
            app_id=app_id,
            exclude_session_id=exclude_session_id,
        )

    async def get_semantic_stability_trend(
        self,
        *,
        app_id: str,
        limit: int = 10,
        default_threshold: float = 90.0,
    ) -> list[dict[str, Any]]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.get_semantic_stability_trend(
            app_id=app_id,
            limit=limit,
            default_threshold=default_threshold,
        )

    async def evaluate_semantic_stability_gate(
        self,
        *,
        app_id: str,
        window: int = 10,
        min_score: float = 90.0,
        avg_score: float = 92.0,
        consecutive_pass_required: int = 3,
        default_threshold: float = 90.0,
    ) -> dict[str, Any]:
        rows = await self.get_semantic_stability_trend(
            app_id=app_id,
            limit=window,
            default_threshold=default_threshold,
        )
        scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
        sample_size = len(rows)
        min_observed = min(scores) if scores else 0.0
        avg_observed = (sum(scores) / sample_size) if sample_size else 0.0
        consecutive_passed = 0
        for row in reversed(rows):
            if bool(row.get("passed")):
                consecutive_passed += 1
                continue
            break

        has_window = sample_size >= max(1, int(window))
        min_ok = has_window and min_observed >= float(min_score)
        avg_ok = has_window and avg_observed >= float(avg_score)
        consecutive_ok = consecutive_passed >= max(1, int(consecutive_pass_required))
        passed = has_window and min_ok and avg_ok and consecutive_ok

        failure_reasons: list[str] = []
        if not has_window:
            failure_reasons.append(
                f"insufficient_samples:{sample_size}<{max(1, int(window))}"
            )
        if has_window and not min_ok:
            failure_reasons.append(
                f"min_score:{round(min_observed, 2)}<{float(min_score):.2f}"
            )
        if has_window and not avg_ok:
            failure_reasons.append(
                f"avg_score:{round(avg_observed, 2)}<{float(avg_score):.2f}"
            )
        if not consecutive_ok:
            failure_reasons.append(
                f"consecutive_passed:{consecutive_passed}<{max(1, int(consecutive_pass_required))}"
            )

        return {
            "app_id": app_id,
            "window": max(1, int(window)),
            "sample_size": sample_size,
            "passed": passed,
            "min_score_threshold": float(min_score),
            "avg_score_threshold": float(avg_score),
            "consecutive_pass_required": max(1, int(consecutive_pass_required)),
            "min_score_observed": round(min_observed, 2),
            "avg_score_observed": round(avg_observed, 2),
            "consecutive_pass_observed": consecutive_passed,
            "failure_reasons": failure_reasons,
            "rows": rows,
        }

    async def evaluate_retention_plan(
        self,
        *,
        app_id: str,
        keep_releases: int = 5,
        min_age_days: int = 14,
    ) -> dict[str, Any]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.plan_retention(
            app_id=app_id,
            keep_releases=keep_releases,
            min_age_days=min_age_days,
        )

    async def run_retention(
        self,
        *,
        app_id: str,
        keep_releases: int = 5,
        min_age_days: int = 14,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        return await self._repo.execute_retention(
            app_id=app_id,
            keep_releases=keep_releases,
            min_age_days=min_age_days,
            batch_size=batch_size,
        )

    async def get_zones(self, app_id: str, zone_type: str | None = None) -> list[dict]:
        """Get zones for an app.

        Args:
            app_id: The app ID
            zone_type: Optional filter by zone type

        Returns:
            List of zone records
        """
        if zone_type:
            query = """
            MATCH (a:App {id: $app_id})-[:HAS_ZONE]->(z:Zone {type: $zone_type})
            RETURN z.id as id, z.type as type, z.selector as selector,
                   z.element_count as element_count, z.bounds as bounds,
                   z.text_sample as text_sample
            """
            result = await self._run_read(query, app_id=app_id, zone_type=zone_type)
        else:
            query = """
            MATCH (a:App {id: $app_id})-[:HAS_ZONE]->(z:Zone)
            RETURN z.id as id, z.type as type, z.selector as selector,
                   z.element_count as element_count, z.bounds as bounds,
                   z.text_sample as text_sample
            """
            result = await self._run_read(query, app_id=app_id)
        return [record for record in result]
    
    # Menu operations
    async def add_menus(
        self,
        app_id: str,
        menus: list[dict],
        page_url: str = "",
        session_id: str | None = None,
    ) -> None:
        """Add menu items to an app with proper PRD relationships.

        Creates:
        - (:App)-[:HAS_MENU]->(:Menu)
        - (:Menu)-[:CHILD_OF]->(:Menu)
        - (:Menu)-[:LEADS_TO]->(:State) for menus with matching href
        - (:Session)-[:DISCOVERED]->(:Menu) when session_id is provided
        """
        if page_url:
            await self._clear_page_menus(app_id, page_url)

        flattened = self._flatten_menus(menus)

        query = """
        UNWIND $menus as menu
        MERGE (m:Menu {id: menu.id})
        SET m.text = menu.text,
            m.href = menu.href,
            m.level = menu.level,
            m.order = menu.order,
            m.is_active = coalesce(menu.is_active, true),
            m.ingest_version_id = menu.ingest_version_id,
            m.page_url = $page_url,
            m.name = coalesce(menu.text, menu.label, menu.menu_key, menu.id),
            m.updated_at = datetime()
        WITH m, menu
        MATCH (a:App {id: $app_id})
        MERGE (a)-[:HAS_MENU]->(m)
        WITH m, menu
        OPTIONAL MATCH (parent:Menu {id: menu.parent_id})
        WHERE menu.parent_id IS NOT NULL
        FOREACH (ignore IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
            MERGE (m)-[:CHILD_OF {order_index: coalesce(menu.order, 0)}]->(parent)
        )
        WITH m, menu
        OPTIONAL MATCH (s:State)
        WHERE s.url = menu.href OR s.url ENDS WITH menu.href
        FOREACH (ignore IN CASE WHEN s IS NOT NULL AND menu.href <> '' THEN [1] ELSE [] END |
            MERGE (m)-[:LEADS_TO {first_seen: datetime(), session_id: $session_id}]->(s)
        )
        """
        await self._run_write(
            query,
            app_id=app_id,
            menus=flattened,
            page_url=page_url,
            session_id=session_id or "",
        )

        if session_id:
            disc_query = """
            UNWIND $menus as menu
            MATCH (sess:Session {id: $session_id}), (m:Menu {id: menu.id})
            MERGE (sess)-[:DISCOVERED]->(m)
            """
            await self._run_write(
                disc_query,
                menus=flattened,
                session_id=session_id,
            )

    async def _clear_page_menus(self, app_id: str, page_url: str) -> None:
        """Mark previous page menus inactive to preserve history."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.mark_page_menus_inactive(app_id=app_id, page_url=page_url)
    
    def _flatten_menus(self, menus: list[dict], parent_id: str | None = None) -> list[dict]:
        """Flatten menu hierarchy for Neo4j storage."""
        result = []
        for menu in menus:
            menu_copy = {**menu, "parent_id": parent_id}
            children = menu_copy.pop("children", [])
            result.append(menu_copy)
            if children:
                result.extend(self._flatten_menus(children, menu["id"]))
        return result
    
    async def get_menus(self, app_id: str, level: int | None = None) -> list[dict]:
        """Get menu items for an app.
        
        Args:
            app_id: The app ID
            level: Optional filter by hierarchy level
            
        Returns:
            List of menu records with parent-child relationships
        """
        if level is not None:
            query = """
            MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu {level: $level})
            OPTIONAL MATCH (m)-[:CHILD_OF]->(parent:Menu)
            RETURN m.id as id, m.text as text, m.href as href,
                   m.level as level, m.order as order, m.is_active as is_active,
                   parent.id as parent_id
            ORDER BY m.level, m.order
            """
            result = await self._run_read(query, app_id=app_id, level=level)
        else:
            query = """
            MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
            OPTIONAL MATCH (m)-[:CHILD_OF]->(parent:Menu)
            RETURN m.id as id, m.text as text, m.href as href,
                   m.level as level, m.order as order, m.is_active as is_active,
                   parent.id as parent_id
            ORDER BY m.level, m.order
            """
            result = await self._run_read(query, app_id=app_id)
        return [record for record in result]

    async def get_menu_hrefs(self, app_id: str) -> list[str]:
        """Get all unique hrefs from menu items.

        Returns:
            Sorted list of unique hrefs
        """
        query = """
        MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
        WHERE m.href IS NOT NULL AND m.href <> ''
        RETURN DISTINCT m.href as href
        ORDER BY href
        """
        result = await self._run_read(query, app_id=app_id)
        return [record["href"] for record in result]

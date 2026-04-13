from __future__ import annotations
import logging
from datetime import datetime
from typing import Any
from neo4j import AsyncDriver
from graph_agent.models import (
    App, State, Transition, Zone, FrameNode, Entity, EntityInstance,
    Intent, Checkpoint, FieldConstraint, TestCase, Session, Menu,
)
from graph_agent.neo4j.queries import CypherQueries

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
                first_seen=datetime.utcnow().isoformat(),
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

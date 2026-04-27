"""GraphManager provides high-level graph operations backed by Neo4j."""

from __future__ import annotations

from typing import Any
from graph_agent.neo4j_client.driver import Neo4jDriver
from graph_agent.neo4j_client.repository import GraphRepository
from graph_agent.models import (
    App,
    Session,
    State,
    Transition,
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
            await session.run(query, **params)
    
    async def _run_read(self, query: str, **params) -> list[Any]:
        """Execute a read query and return records."""
        async with self._driver.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()
    
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
    
    async def link_app_state(self, app_id: str, state_id: str) -> None:
        """Link an app to a state."""
        if self._repo is None:
            raise RuntimeError("GraphManager not initialized")
        await self._repo.link_app_state(app_id, state_id)
    
    async def set_session_inventory(self, session_id: str, inventory: dict) -> None:
        """Store inventory data with a session."""
        query = """
        MATCH (s:Session {id: $session_id})
        SET s.inventory = $inventory
        """
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
        # Convert stats to a format suitable for Neo4j properties
        query = """
        MATCH (s:Session {id: $session_id})
        SET s.visited_urls = $visited_urls,
            s.mapping_stopped = $mapping_stopped,
            s.stop_reason = $stop_reason,
            s.start_url = $start_url,
            s.states_added = $states_added,
            s.transitions_added = $transitions_added,
            s.filtered_non_ui_edges = $filtered_non_ui_edges,
            s.semantic_mismatch_warnings = $semantic_mismatch_warnings,
            s.url_discontinuity_warnings = $url_discontinuity_warnings,
            s.frame_context_transition_warnings = $frame_context_transition_warnings,
            s.name = coalesce(s.focus, s.id)
        """
        await self._run_write(
            query,
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
            transition.from_state_id,
            transition.to_state_id,
        )

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
        query = """
        UNWIND $zones as zone
        MERGE (z:Zone {id: zone.id})
        SET z.type = zone.type,
            z.selector = zone.selector,
            z.element_count = zone.element_count,
            z.bounds = zone.bounds,
            z.text_sample = zone.text_sample,
            z.name = coalesce(zone.summary, zone.type, zone.id),
            z.updated_at = datetime()
        WITH z, zone
        MATCH (a:App {id: $app_id})
        MERGE (a)-[:HAS_ZONE]->(z)
        """
        await self._run_write(query, app_id=app_id, zones=zones)

        if state_id:
            link_query = """
            UNWIND $zones as zone
            MATCH (s:State {id: $state_id}), (z:Zone {id: zone.id})
            MERGE (s)-[:HAS_ZONE]->(z)
            """
            await self._run_write(link_query, zones=zones, state_id=state_id)

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
            m.is_active = menu.is_active,
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
        """Clear existing menus for a page to avoid duplicates."""
        query = """
        MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu {page_url: $page_url})
        DETACH DELETE m
        """
        await self._run_write(query, app_id=app_id, page_url=page_url)
    
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

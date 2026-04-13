"""GraphManager provides high-level graph operations backed by Neo4j."""

from __future__ import annotations

from typing import Any
from graph_agent.neo4j.driver import Neo4jDriver
from graph_agent.neo4j.repository import GraphRepository
from graph_agent.models import (
    App,
    Session,
    State,
    Transition,
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
        async with self._driver.session() as session:
            await session.run(query, **params)
    
    async def _run_read(self, query: str, **params) -> list[Any]:
        """Execute a read query and return records."""
        async with self._driver.session() as session:
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
            s.frame_context_transition_warnings = $frame_context_transition_warnings
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
        OPTIONAL MATCH (s)-[:HAS_INTENT]->(i:Intent)
        RETURN s.id as state_id, s.url as url, s.title as title,
               collect(i{.*}) as intents
        """
        result = await self._run_read(query, app_id=app_id)
        return [record for record in result]

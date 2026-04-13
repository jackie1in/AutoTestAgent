"""Neo4j-based graph manager - replaces networkx operations."""

from __future__ import annotations

import logging
from pathlib import Path

from graph_agent.neo4j import Neo4jDriver, GraphRepository
from graph_agent.models import (
    App, State, Transition, Zone, Intent, Checkpoint, Session,
)

logger = logging.getLogger(__name__)


class Neo4jGraphManager:
    """Manages graph operations using Neo4j instead of networkx."""
    
    def __init__(self, driver: Neo4jDriver | None = None):
        self._driver = driver or Neo4jDriver()
        self._repo: GraphRepository | None = None
    
    async def connect(self) -> GraphRepository:
        """Connect to Neo4j and return repository."""
        await self._driver.connect()
        await self._driver.ensure_schema()
        self._repo = GraphRepository(self._driver.driver)
        return self._repo
    
    async def close(self):
        """Close Neo4j connection."""
        await self._driver.close()
        self._repo = None
    
    @property
    def repo(self) -> GraphRepository:
        """Get repository (must call connect() first)."""
        if self._repo is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._repo
    
    async def add_app(self, app: App) -> None:
        """Add or update an app."""
        await self.repo.upsert_app(app)
    
    async def add_state(self, state: State) -> None:
        """Add or update a state node."""
        await self.repo.upsert_state(state)
    
    async def add_transition(self, transition: Transition) -> None:
        """Add or update a transition edge."""
        await self.repo.upsert_transition(transition)
    
    async def link_app_state(self, app_id: str, state_id: str) -> None:
        """Link app to state."""
        await self.repo.link_app_state(app_id, state_id)
    
    async def get_state(self, state_id: str) -> State | None:
        """Get state by ID."""
        return await self.repo.get_state(state_id)
    
    async def get_states_by_url(self, url: str) -> list[State]:
        """Get states by URL."""
        return await self.repo.get_states_by_url(url)
    
    async def get_transitions_from_state(self, state_id: str) -> list[tuple[Transition, State]]:
        """Get transitions from a state."""
        return await self.repo.get_transitions_from_state(state_id)
    
    async def shortest_path(self, from_state_id: str, to_state_id: str) -> list[State] | None:
        """Find shortest path between two states."""
        return await self.repo.get_shortest_path(from_state_id, to_state_id)
    
    async def has_path(self, from_state_id: str, to_state_id: str) -> bool:
        """Check if path exists between two states."""
        path = await self.shortest_path(from_state_id, to_state_id)
        return path is not None and len(path) > 0
    
    async def get_all_states(self) -> list[State]:
        """Get all states."""
        return await self.repo.get_all_states()
    
    async def get_state_count(self) -> int:
        """Get total state count."""
        return await self.repo.get_state_count()
    
    async def get_transition_count(self) -> int:
        """Get total transition count."""
        return await self.repo.get_transition_count()
    
    async def add_zone(self, zone: Zone) -> None:
        """Add or update a zone."""
        await self.repo.upsert_zone(zone)
    
    async def link_state_zone(self, state_id: str, zone_id: str) -> None:
        """Link state to zone."""
        await self.repo.link_state_zone(state_id, zone_id)
    
    async def add_intent(self, intent: Intent) -> None:
        """Add or update an intent."""
        await self.repo.upsert_intent(intent)
    
    async def link_transition_intent(self, transition_id: str, intent_id: str) -> None:
        """Link transition to intent."""
        await self.repo.link_transition_intent(transition_id, intent_id)
    
    async def add_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Add or update a checkpoint."""
        await self.repo.upsert_checkpoint(checkpoint)
    
    async def link_transition_checkpoint(self, transition_id: str, checkpoint_id: str, timing: str) -> None:
        """Link transition to checkpoint."""
        if timing == "before":
            await self.repo.link_transition_check_before(transition_id, checkpoint_id)
        else:
            await self.repo.link_transition_check_after(transition_id, checkpoint_id)
    
    async def add_session(self, session: Session) -> None:
        """Add or update a session."""
        await self.repo.upsert_session(session)
    
    async def link_session_discovered(self, session_id: str, state_id: str) -> None:
        """Link session to discovered state."""
        await self.repo.link_session_discovered_state(session_id, state_id)
    
    async def link_session_transition(self, session_id: str, transition_id: str) -> None:
        """Link session to discovered transition."""
        await self.repo.link_session_discovered_transition(session_id, transition_id)
    
    async def merge_graph(self, other_app_id: str, into_app_id: str) -> None:
        """Merge one app's graph into another."""
        # Neo4j handles this via relationships - just link states to new app
        states = await self.repo.get_app_states(other_app_id)
        for state in states:
            await self.repo.link_app_state(into_app_id, state.id)
    
    async def export_to_json(self, app_id: str, path: str | Path) -> None:
        """Export graph to JSON (for compatibility)."""
        import json
        
        app = await self.repo.get_app(app_id)
        states = await self.repo.get_app_states(app_id)
        
        # Get all transitions between these states
        transitions = []
        for state in states:
            trans = await self.repo.get_transitions_from_state(state.id)
            for t, target in trans:
                transitions.append({
                    "from": state.id,
                    "to": target.id,
                    "transition": t.model_dump(),
                })
        
        data = {
            "app": app.model_dump() if app else None,
            "states": [s.model_dump() for s in states],
            "transitions": transitions,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    async def import_from_json(self, path: str | Path, app_id: str | None = None) -> App:
        """Import graph from JSON (for compatibility)."""
        import json
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Import app
        app_data = data.get("app") or {"id": app_id or "imported", "name": "Imported"}
        app = App(**app_data)
        await self.repo.upsert_app(app)
        
        # Import states
        for state_data in data.get("states", []):
            state = State(**state_data)
            await self.repo.upsert_state(state)
            await self.repo.link_app_state(app.id, state.id)
        
        # Import transitions
        for trans_data in data.get("transitions", []):
            trans = Transition(**trans_data["transition"])
            await self.repo.upsert_transition(trans)
        
        return app
    
    async def clear_graph(self, app_id: str | None = None) -> None:
        """Clear graph data."""
        if app_id:
            # Delete only this app's data
            await self.repo.delete_app_data(app_id)
        else:
            # Clear all (use with caution)
            await self.repo.clear_all()


# Compatibility aliases for networkx migration
async def save_graph(manager: Neo4jGraphManager, path: str | Path, app_id: str) -> None:
    """Export graph to JSON (compatibility function)."""
    await manager.export_to_json(app_id, path)


async def load_graph(manager: Neo4jGraphManager, path: str | Path, app_id: str | None = None) -> None:
    """Import graph from JSON (compatibility function)."""
    await manager.import_from_json(path, app_id)

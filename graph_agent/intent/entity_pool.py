from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver

from graph_agent.models import Entity, EntityInstance

logger = logging.getLogger(__name__)


class EntityPool:
    """Manages test data entity instances in Neo4j."""

    def __init__(self, neo4j_driver: AsyncDriver):
        self._neo4j = neo4j_driver

    async def get_available(self, entity_id: str) -> EntityInstance | None:
        """Get an available instance of the specified entity type."""
        async with self._neo4j.session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $eid})-[:HAS_INSTANCE]->(ei:EntityInstance) "
                "WHERE ei.status = 'available' "
                "RETURN ei.id AS id, ei.data AS data, ei.status AS status, "
                "ei.session_id AS session_id, ei.created_at AS created_at "
                "LIMIT 1",
                eid=entity_id,
            )
            rec = await result.single()
            if rec:
                return EntityInstance(
                    id=rec["id"],
                    data=rec.get("data", "{}"),
                    status=rec.get("status", "available"),
                    session_id=rec.get("session_id"),
                )
        return None

    async def create_instance(
        self, entity: Entity, data: dict[str, Any], session_id: str
    ) -> EntityInstance:
        """Create a new entity instance and persist to Neo4j."""
        now = datetime.utcnow().isoformat()
        instance = EntityInstance(
            id=f"instance:{entity.id}:{now}",
            data=json.dumps(data),
            session_id=session_id,
        )
        async with self._neo4j.session() as session:
            await session.run(
                "CREATE (ei:EntityInstance {id: $id, data: $data, status: 'available', "
                "session_id: $sid, created_at: $now})",
                id=instance.id, data=instance.data, sid=session_id, now=now,
            )
            await session.run(
                "MATCH (e:Entity {id: $eid}), (ei:EntityInstance {id: $eiid}) "
                "MERGE (e)-[:HAS_INSTANCE]->(ei)",
                eid=entity.id, eiid=instance.id,
            )
        return instance

    async def consume(self, instance_id: str) -> None:
        """Mark an instance as consumed."""
        async with self._neo4j.session() as session:
            await session.run(
                "MATCH (ei:EntityInstance {id: $id}) SET ei.status = 'consumed'",
                id=instance_id,
            )

    async def expire(self, instance_id: str) -> None:
        """Mark an instance as expired."""
        async with self._neo4j.session() as session:
            await session.run(
                "MATCH (ei:EntityInstance {id: $id}) SET ei.status = 'expired'",
                id=instance_id,
            )

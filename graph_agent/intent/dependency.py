from __future__ import annotations

import json
import logging
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyzes data flow dependencies between intents."""

    async def analyze(self, neo4j_driver: AsyncDriver) -> list[dict[str, Any]]:
        """Analyze intent dependencies and create DATA_FLOW relationships.

        Rules:
        - If Intent A produces Entity X, and Intent B requires Entity X as precondition,
          then B depends on A (A)-[:DATA_FLOW]->(B)
        - Detected via Entity Checkpoints (pre/post conditions) linked to Intents
        """
        dependencies: list[dict[str, Any]] = []

        async with neo4j_driver.session() as session:
            # Find producer intents (those with entity postcondition checkpoints)
            producers = await session.run(
                "MATCH (i:Intent)<-[:MAPPED_TO]-(t:Transition)-[:CHECK_AFTER]->(c:Checkpoint) "
                "WHERE c.layer = 'entity' AND c.rule_type = 'entity_created' "
                "RETURN i.id AS intent_id, i.key AS intent_key, c.rule AS rule"
            )
            producer_map: dict[str, str] = {}
            for rec in await producers.data():
                try:
                    rule = json.loads(rec.get("rule", "{}"))
                    entity_id = rule.get("entity_id", "")
                    if entity_id:
                        producer_map[entity_id] = rec["intent_id"]
                except Exception:
                    continue

            # Find consumer intents (those with entity precondition checkpoints)
            consumers = await session.run(
                "MATCH (i:Intent)<-[:MAPPED_TO]-(t:Transition)-[:CHECK_BEFORE]->(c:Checkpoint) "
                "WHERE c.layer = 'entity' AND c.rule_type = 'entity_exists' "
                "RETURN i.id AS intent_id, i.key AS intent_key, c.rule AS rule"
            )
            for rec in await consumers.data():
                try:
                    rule = json.loads(rec.get("rule", "{}"))
                    entity_id = rule.get("entity_id", "")
                    if entity_id and entity_id in producer_map:
                        producer_id = producer_map[entity_id]
                        consumer_id = rec["intent_id"]

                        await session.run(
                            "MATCH (a:Intent {id: $pid}), (b:Intent {id: $cid}) "
                            "MERGE (a)-[:DATA_FLOW {entity_id: $eid}]->(b)",
                            pid=producer_id, cid=consumer_id, eid=entity_id,
                        )
                        dependencies.append({
                            "from": producer_id,
                            "to": consumer_id,
                            "entity_id": entity_id,
                        })
                except Exception:
                    continue

        logger.info("Found %d intent dependencies", len(dependencies))
        return dependencies

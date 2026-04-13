from __future__ import annotations

import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class IntentPlanner:
    """Plans test execution order based on intent dependencies."""

    async def plan(self, target_intent_id: str, neo4j_driver: AsyncDriver) -> list[str]:
        """Generate an ordered list of intent IDs needed to reach the target.

        Uses topological sort on the intent dependency (DATA_FLOW) subgraph.
        """
        async with neo4j_driver.session() as session:
            # Get all ancestors of target intent via DATA_FLOW
            result = await session.run(
                "MATCH path = (ancestor:Intent)-[:DATA_FLOW*]->(target:Intent {id: $tid}) "
                "WITH ancestor, length(path) AS depth "
                "ORDER BY depth DESC "
                "RETURN DISTINCT ancestor.id AS id",
                tid=target_intent_id,
            )
            ancestors = [rec["id"] for rec in await result.data()]

        # Topological order: deepest ancestors first, target last
        ordered = ancestors + [target_intent_id]
        # Remove duplicates preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for iid in ordered:
            if iid not in seen:
                seen.add(iid)
                unique.append(iid)

        logger.info("Plan for %s: %d steps", target_intent_id, len(unique))
        return unique

from __future__ import annotations

import logging

from neo4j import AsyncDriver

from graph_agent.models import CoverageReport, TransitionConfidenceDistribution

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Computes test coverage metrics from the Neo4j graph."""

    def __init__(self, neo4j_driver: AsyncDriver):
        self._driver = neo4j_driver

    async def compute(self) -> CoverageReport:
        """Compute current coverage report from graph data."""
        async with self._driver.session() as session:
            # Menu coverage: states with menu_path vs total leaf menu states
            r = await session.run(
                "OPTIONAL MATCH (s:State) "
                "WITH count(s) AS total "
                "OPTIONAL MATCH (s2:State) WHERE size(s2.menu_path) > 0 "
                "RETURN total, count(s2) AS discovered"
            )
            rec = await r.single()
            total_states = rec["total"] if rec else 0
            discovered_states = rec["discovered"] if rec else 0
            menu_cov = discovered_states / max(1, total_states)

            # Zone coverage
            r2 = await session.run(
                "OPTIONAL MATCH (z:Zone) "
                "WITH count(z) AS total "
                "OPTIONAL MATCH (z2:Zone) WHERE z2.exploration_status IN ['explored', 'validated'] "
                "RETURN total, count(z2) AS explored"
            )
            rec2 = await r2.single()
            total_zones = rec2["total"] if rec2 else 0
            explored_zones = rec2["explored"] if rec2 else 0
            zone_cov = explored_zones / max(1, total_zones)

            # Interaction coverage: transitions with confidence > 0 vs total
            r3 = await session.run(
                "OPTIONAL MATCH (t:Transition) "
                "WITH count(t) AS total "
                "OPTIONAL MATCH (t2:Transition) WHERE t2.validation_count > 0 "
                "RETURN total, count(t2) AS validated"
            )
            rec3 = await r3.single()
            total_trans = rec3["total"] if rec3 else 0
            validated_trans = rec3["validated"] if rec3 else 0
            interaction_cov = validated_trans / max(1, total_trans)

            # Confidence distribution
            r4 = await session.run(
                "OPTIONAL MATCH (t:Transition) "
                "RETURN "
                "sum(CASE WHEN t.confidence >= 0.8 THEN 1 ELSE 0 END) AS high, "
                "sum(CASE WHEN t.confidence >= 0.4 AND t.confidence < 0.8 THEN 1 ELSE 0 END) AS medium, "
                "sum(CASE WHEN t.confidence < 0.4 THEN 1 ELSE 0 END) AS low"
            )
            rec4 = await r4.single()
            dist = TransitionConfidenceDistribution(
                high=rec4["high"] or 0 if rec4 else 0,
                medium=rec4["medium"] or 0 if rec4 else 0,
                low=rec4["low"] or 0 if rec4 else 0,
            )

            overall = menu_cov * 0.3 + zone_cov * 0.4 + interaction_cov * 0.3

            if overall >= 0.9:
                recommendation = "complete"
            elif dist.low > dist.high:
                recommendation = "needs_validation"
            else:
                recommendation = "needs_more"

            report = CoverageReport(
                menu_coverage=round(menu_cov, 4),
                zone_coverage=round(zone_cov, 4),
                interaction_coverage=round(interaction_cov, 4),
                transition_confidence=dist,
                overall_completeness=round(overall, 4),
                recommendation=recommendation,
            )
            logger.info(
                "Coverage: menu=%.1f%% zone=%.1f%% interaction=%.1f%% overall=%.1f%% → %s",
                menu_cov * 100,
                zone_cov * 100,
                interaction_cov * 100,
                overall * 100,
                recommendation,
            )
            return report

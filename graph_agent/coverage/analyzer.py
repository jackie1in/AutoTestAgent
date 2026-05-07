from __future__ import annotations

import logging

from neo4j import AsyncDriver

from graph_agent.models import CoverageReport, TransitionConfidenceDistribution

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Computes test coverage metrics from the Neo4j graph."""

    def __init__(self, neo4j_driver: AsyncDriver):
        self._driver = neo4j_driver

    async def compute(self, app_id: str | None = None) -> CoverageReport:
        """Compute current coverage report from graph data.

        Args:
            app_id: Optional app ID to scope the analysis. If None, computes globally.
        """
        async with self._driver.session() as session:
            app_clause = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State) " if app_id else "OPTIONAL MATCH (s:State) "
            app_clause2 = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s2:State) WHERE size(s2.menu_path) > 0 " if app_id else "OPTIONAL MATCH (s2:State) WHERE size(s2.menu_path) > 0 "
            app_clause_zone = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(:State)-[:HAS_ZONE]->(z:Zone) " if app_id else "OPTIONAL MATCH (z:Zone) "
            app_clause_zone2 = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(:State)-[:HAS_ZONE]->(z2:Zone) WHERE z2.exploration_status IN ['explored', 'validated'] " if app_id else "OPTIONAL MATCH (z2:Zone) WHERE z2.exploration_status IN ['explored', 'validated'] "
            app_clause_trans = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(:State)<-[:FROM]-(t:Transition) " if app_id else "OPTIONAL MATCH (t:Transition) "
            app_clause_trans2 = "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(:State)<-[:FROM]-(t2:Transition) WHERE t2.validation_count > 0 " if app_id else "OPTIONAL MATCH (t2:Transition) WHERE t2.validation_count > 0 "

            params = {"app_id": app_id} if app_id else {}

            # Menu coverage: states with menu_path vs total leaf menu states
            r = await session.run(
                app_clause +
                "WITH count(s) AS total "
                + app_clause2 +
                "RETURN total, count(s2) AS discovered",
                **params
            )
            rec = await r.single()
            total_states = rec["total"] if rec else 0
            discovered_states = rec["discovered"] if rec else 0
            menu_cov = discovered_states / max(1, total_states)

            # Zone coverage
            r2 = await session.run(
                app_clause_zone +
                "WITH count(z) AS total "
                + app_clause_zone2 +
                "RETURN total, count(z2) AS explored",
                **params
            )
            rec2 = await r2.single()
            total_zones = rec2["total"] if rec2 else 0
            explored_zones = rec2["explored"] if rec2 else 0
            zone_cov = explored_zones / max(1, total_zones)

            # Interaction coverage: transitions with confidence > 0 vs total
            r3 = await session.run(
                app_clause_trans +
                "WITH count(t) AS total "
                + app_clause_trans2 +
                "RETURN total, count(t2) AS validated",
                **params
            )
            rec3 = await r3.single()
            total_trans = rec3["total"] if rec3 else 0
            validated_trans = rec3["validated"] if rec3 else 0
            interaction_cov = validated_trans / max(1, total_trans)

            # State coverage: unique spa_routes vs total states (SPA diversity)
            app_clause_spa = (
                "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State) "
                if app_id
                else "OPTIONAL MATCH (s:State) "
            )
            r_spa = await session.run(
                app_clause_spa
                + "RETURN count(s) AS total_states, count(DISTINCT s.spa_route) AS unique_routes",
                **params,
            )
            rec_spa = await r_spa.single()
            total_states = rec_spa["total_states"] or 0 if rec_spa else 0
            unique_routes = rec_spa["unique_routes"] or 0 if rec_spa else 0
            # State coverage: reward having multiple distinct routes; cap at 1.0
            state_cov = min(1.0, unique_routes / max(1, total_states * 0.5))

            # Confidence distribution
            r4 = await session.run(
                app_clause_trans +
                "RETURN "
                "sum(CASE WHEN t.confidence >= 0.8 THEN 1 ELSE 0 END) AS high, "
                "sum(CASE WHEN t.confidence >= 0.4 AND t.confidence < 0.8 THEN 1 ELSE 0 END) AS medium, "
                "sum(CASE WHEN t.confidence < 0.4 THEN 1 ELSE 0 END) AS low",
                **params
            )
            rec4 = await r4.single()
            dist = TransitionConfidenceDistribution(
                high=rec4["high"] or 0 if rec4 else 0,
                medium=rec4["medium"] or 0 if rec4 else 0,
                low=rec4["low"] or 0 if rec4 else 0,
            )

            # Weighted overall: lower menu weight for SPAs, add state coverage
            overall = menu_cov * 0.2 + zone_cov * 0.3 + interaction_cov * 0.2 + state_cov * 0.3

            # Minimum states threshold: never report complete with < 3 states
            min_states_required = 3
            if total_states < min_states_required:
                recommendation = "needs_more"
            elif overall >= 0.85:
                recommendation = "complete"
            elif dist.low > dist.high:
                recommendation = "needs_validation"
            else:
                recommendation = "needs_more"

            report = CoverageReport(
                menu_coverage=round(menu_cov, 4),
                zone_coverage=round(zone_cov, 4),
                interaction_coverage=round(interaction_cov, 4),
                state_coverage=round(state_cov, 4),
                transition_confidence=dist,
                overall_completeness=round(overall, 4),
                recommendation=recommendation,
            )
            logger.info(
                "Coverage: menu=%.1f%% zone=%.1f%% interaction=%.1f%% state=%.1f%% overall=%.1f%% → %s",
                menu_cov * 100,
                zone_cov * 100,
                interaction_cov * 100,
                state_cov * 100,
                overall * 100,
                recommendation,
            )
            return report

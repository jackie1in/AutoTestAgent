from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Zones explored more than this long ago are eligible for re-exploration
_ZONE_STALE_HOURS = 168


@dataclass
class ScheduledTask:
    type: str
    priority: int
    target_id: str
    context: dict[str, Any] = field(default_factory=dict)


class ExplorationScheduler:
    """Generates prioritized exploration tasks based on coverage gaps."""

    async def schedule(
        self,
        neo4j_driver: AsyncDriver,
        focus: Literal["breadth", "depth", "validation"] = "breadth",
        max_tasks: int = 500,
        app_id: str | None = None,
    ) -> list[ScheduledTask]:
        tasks: list[ScheduledTask] = []

        async with neo4j_driver.session() as session:
            if focus in ("breadth", "depth"):
                # P100: States with no zones (undiscovered pages)
                if app_id:
                    q1 = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State) "
                        "WHERE NOT (s)-[:HAS_ZONE]->() "
                        "RETURN s.id AS id, s.url AS url, s.title AS title "
                        "ORDER BY s.first_discovered DESC LIMIT $limit"
                    )
                else:
                    q1 = (
                        "MATCH (s:State) WHERE NOT (s)-[:HAS_ZONE]->() "
                        "RETURN s.id AS id, s.url AS url, s.title AS title "
                        "ORDER BY s.first_discovered DESC LIMIT $limit"
                    )
                r1 = await session.run(q1, limit=max_tasks, app_id=app_id)
                for rec in await r1.data():
                    tasks.append(
                        ScheduledTask(
                            type="discover_page",
                            priority=100,
                            target_id=rec["id"],
                            context={"url": rec["url"], "title": rec.get("title", "")},
                        )
                    )

                # P80: Unexplored zones
                if app_id:
                    q2 = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status IN ['undiscovered', 'discovered'] "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt, z.summary AS summary "
                        "ORDER BY z.exploration_status LIMIT $limit"
                    )
                else:
                    q2 = (
                        "MATCH (s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status IN ['undiscovered', 'discovered'] "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt, z.summary AS summary "
                        "ORDER BY z.exploration_status LIMIT $limit"
                    )
                r2 = await session.run(q2, limit=max_tasks, app_id=app_id)
                for rec in await r2.data():
                    tasks.append(
                        ScheduledTask(
                            type="explore_zone",
                            priority=80,
                            target_id=rec["zid"],
                            context={
                                "state_id": rec["sid"],
                                "zone_type": rec["zt"],
                                "summary": rec.get("summary", ""),
                            },
                        )
                    )

                # P60: Partially explored zones
                if app_id:
                    q3 = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status = 'partial' "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt "
                        "LIMIT $limit"
                    )
                else:
                    q3 = (
                        "MATCH (s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status = 'partial' "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt "
                        "LIMIT $limit"
                    )
                r3 = await session.run(q3, limit=max_tasks, app_id=app_id)
                for rec in await r3.data():
                    tasks.append(
                        ScheduledTask(
                            type="explore_zone",
                            priority=60,
                            target_id=rec["zid"],
                            context={"state_id": rec["sid"], "zone_type": rec["zt"]},
                        )
                    )

                # P30: Stale zones — previously explored but older than threshold
                stale_cutoff = (
                    datetime.utcnow() - timedelta(hours=_ZONE_STALE_HOURS)
                ).isoformat()
                if app_id:
                    qs = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status = 'explored' "
                        "AND z.last_explored < $cutoff "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt, "
                        "z.last_explored AS le "
                        "ORDER BY z.last_explored ASC LIMIT $limit"
                    )
                else:
                    qs = (
                        "MATCH (s:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status = 'explored' "
                        "AND z.last_explored < $cutoff "
                        "RETURN z.id AS zid, s.id AS sid, z.zone_type AS zt, "
                        "z.last_explored AS le "
                        "ORDER BY z.last_explored ASC LIMIT $limit"
                    )
                r_stale = await session.run(qs, cutoff=stale_cutoff, limit=max_tasks, app_id=app_id)
                for rec in await r_stale.data():
                    tasks.append(
                        ScheduledTask(
                            type="explore_zone",
                            priority=30,
                            target_id=rec["zid"],
                            context={
                                "state_id": rec["sid"],
                                "zone_type": rec["zt"],
                                "reason": "stale_re_explore",
                            },
                        )
                    )

            if focus in ("validation", "depth"):
                # P20: Low-confidence transitions
                if app_id:
                    q4 = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State) "
                        "WHERE t.confidence < 0.4 "
                        "RETURN t.id AS tid, s1.id AS from_id, s2.id AS to_id, "
                        "t.confidence AS conf, t.action AS action "
                        "ORDER BY t.confidence ASC LIMIT $limit"
                    )
                else:
                    q4 = (
                        "MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State) "
                        "WHERE t.confidence < 0.4 "
                        "RETURN t.id AS tid, s1.id AS from_id, s2.id AS to_id, "
                        "t.confidence AS conf, t.action AS action "
                        "ORDER BY t.confidence ASC LIMIT $limit"
                    )
                r4 = await session.run(q4, limit=max_tasks, app_id=app_id)
                for rec in await r4.data():
                    tasks.append(
                        ScheduledTask(
                            type="validate_transition",
                            priority=20,
                            target_id=rec["tid"],
                            context={
                                "from_state_id": rec["from_id"],
                                "to_state_id": rec["to_id"],
                                "confidence": rec["conf"],
                            },
                        )
                    )

            # P40: Zones that need constraint probing (depth focus)
            if focus == "depth":
                if app_id:
                    q5 = (
                        "MATCH (a:App {id: $app_id})-[:HAS_STATE]->(:State)-[:HAS_ZONE]->(z:Zone) "
                        "WHERE z.exploration_status = 'explored' "
                        "AND NOT (z)-[:HAS_FIELD]->() "
                        "RETURN z.id AS zid LIMIT $limit"
                    )
                else:
                    q5 = (
                        "MATCH (z:Zone) WHERE z.exploration_status = 'explored' "
                        "AND NOT (z)-[:HAS_FIELD]->() "
                        "RETURN z.id AS zid LIMIT $limit"
                    )
                r5 = await session.run(q5, limit=max_tasks, app_id=app_id)
                for rec in await r5.data():
                    tasks.append(
                        ScheduledTask(
                            type="probe_constraints",
                            priority=40,
                            target_id=rec["zid"],
                            context={},
                        )
                    )

        tasks.sort(key=lambda t: t.priority, reverse=True)
        return tasks[:max_tasks]

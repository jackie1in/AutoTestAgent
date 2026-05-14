from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from neo4j import AsyncDriver

from graph_agent.models import Checkpoint, State, Transition, Zone

logger = logging.getLogger(__name__)

# Fingerprint changes within this window (seconds) don't degrade confidence,
# protecting against pages with dynamic content (timestamps, counters, etc.).
_FP_GRACE_PERIOD_SECONDS = 300


def _source_priority(source_type: str | None) -> int:
    value = (source_type or "auto").strip().lower()
    if value == "manual_graph_assisted":
        return 3
    if value == "manual_raw":
        return 2
    return 1


@dataclass
class CartographyResult:
    """Output of a single exploration task."""

    states: list[State] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    zone_state_map: dict[str, str] = field(default_factory=dict)
    checkpoint_transition_map: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, object]] = field(default_factory=list)
    menus: list[dict[str, object]] = field(default_factory=list)
    zone_hints: list[dict[str, object]] = field(default_factory=list)
    layout_evidence: list[dict[str, object]] = field(default_factory=list)
    layout_metrics: dict[str, object] = field(default_factory=dict)
    intervention_tasks: list[dict[str, object]] = field(default_factory=list)
    semantic_conflict_count: int = 0


@dataclass
class MergeReport:
    """Summary of a merge operation."""

    states_created: int = 0
    states_updated: int = 0
    transitions_created: int = 0
    transitions_updated: int = 0
    transitions_confidence_boosted: int = 0
    zones_created: int = 0
    checkpoints_created: int = 0


class GraphMerger:
    """Merges exploration findings into the Neo4j graph.

    All writes within a single ``merge()`` call share one Neo4j transaction
    so that partial failures are rolled back automatically.
    """

    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def merge(
        self,
        findings: CartographyResult,
        session_id: str,
        app_id: str | None = None,
    ) -> MergeReport:
        report = MergeReport()

        async with self._driver.session() as neo_session:
            tx = await neo_session.begin_transaction()
            try:
                for state in findings.states:
                    created = await self._merge_state(tx, state, session_id)
                    if created:
                        report.states_created += 1
                    else:
                        report.states_updated += 1
                    if app_id:
                        await tx.run(
                            "MATCH (a:App {id: $app_id}), (s:State {id: $state_id}) "
                            "MERGE (a)-[:HAS_STATE]->(s)",
                            app_id=app_id, state_id=state.id,
                        )

                for transition in findings.transitions:
                    result = await self._merge_transition(tx, transition, session_id)
                    if result == "created":
                        report.transitions_created += 1
                    elif result == "boosted":
                        report.transitions_confidence_boosted += 1
                    else:
                        report.transitions_updated += 1

                for zone in findings.zones:
                    state_id = findings.zone_state_map.get(zone.id)
                    created = await self._merge_zone(tx, zone, state_id)
                    if created:
                        report.zones_created += 1

                for cp in findings.checkpoints:
                    transition_id = findings.checkpoint_transition_map.get(cp.id)
                    await self._merge_checkpoint(tx, cp, session_id, transition_id)
                    report.checkpoints_created += 1

                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

        logger.info(
            "Merge complete: %d states (+%d new), %d transitions (+%d new, %d boosted), %d zones, %d checkpoints",
            report.states_created + report.states_updated,
            report.states_created,
            report.transitions_created
            + report.transitions_updated
            + report.transitions_confidence_boosted,
            report.transitions_created,
            report.transitions_confidence_boosted,
            report.zones_created,
            report.checkpoints_created,
        )
        return report

    # ── State ────────────────────────────────────────────────

    async def _merge_state(self, tx, state: State, session_id: str) -> bool:
        result = await tx.run(
            "MATCH (s:State {id: $id}) RETURN s.id AS id, s.fingerprint AS fp, "
            "s.last_visited AS lv",
            id=state.id,
        )
        existing = await result.single()
        now = datetime.now(UTC).isoformat()
        if existing is None:
            await tx.run(
                "CREATE (s:State {id: $id, url: $url, title: $title, "
                "fingerprint: $fp, spa_route: $spa_route, first_discovered: $now, last_visited: $now, "
                "visit_count: 1, menu_path: $menu_path, is_modal: $is_modal, "
                "name: coalesce($title, $url, $id)})",
                id=state.id,
                url=state.url,
                title=state.title,
                fp=state.fingerprint,
                spa_route=state.spa_route or "",
                now=now,
                menu_path=state.menu_path,
                is_modal=state.is_modal,
            )
            await tx.run(
                "MATCH (sess:Session {id: $sid}), (s:State {id: $state_id}) "
                "MERGE (sess)-[:DISCOVERED]->(s)",
                sid=session_id,
                state_id=state.id,
            )
            return True

        # Existing state — decide whether fingerprint change is significant
        fp_changed = (
            existing["fp"] != state.fingerprint
            and state.fingerprint
            and existing["fp"]
        )
        if fp_changed:
            last_visited = existing.get("lv") or ""
            should_degrade = True
            if last_visited:
                try:
                    lv_dt = datetime.fromisoformat(last_visited)
                    now_dt = datetime.fromisoformat(now)
                    elapsed = (now_dt - lv_dt).total_seconds()
                    if elapsed < _FP_GRACE_PERIOD_SECONDS:
                        should_degrade = False
                except (ValueError, TypeError):
                    pass

            if should_degrade:
                await tx.run(
                    "MATCH (s:State {id: $id})<-[:FROM]-(t:Transition) "
                    "SET t.confidence = CASE WHEN t.confidence - 0.1 < 0 THEN 0.0 "
                    "ELSE t.confidence - 0.1 END",
                    id=state.id,
                )

        update_parts = [
            "s.last_visited = $now",
            "s.visit_count = s.visit_count + 1",
        ]
        if state.fingerprint:
            update_parts.append("s.fingerprint = $fp")
        if state.spa_route:
            update_parts.append("s.spa_route = $spa_route")

        await tx.run(
            "MATCH (s:State {id: $id}) SET " + ", ".join(update_parts),
            id=state.id,
            now=now,
            fp=state.fingerprint,
            spa_route=state.spa_route or "",
        )
        return False

    # ── Transition ───────────────────────────────────────────

    def _transition_stable_key(self, transition: Transition) -> str:
        from_id = str(transition.from_state_id or "")
        to_id = str(transition.to_state_id or "")
        action = (
            transition.action.value
            if hasattr(transition.action, "value")
            else str(transition.action)
        )
        semantic = str(transition.semantic_action_key or transition.selector or "")
        return f"{from_id}|{to_id}|{action.lower()}|{semantic}"

    async def _write_transition_revision(
        self,
        tx,
        *,
        stable_key: str,
        transition: Transition,
        transition_id: str,
        session_id: str,
        active: bool,
        supersedes_revision_ids: list[str],
    ) -> str:
        revision_id = (
            f"trev:{hashlib.md5(f'{stable_key}|{transition_id}|{session_id}|{datetime.now(UTC).isoformat()}'.encode()).hexdigest()[:16]}"
        )
        action_val = (
            transition.action.value
            if hasattr(transition.action, "value")
            else str(transition.action)
        )
        source_value = (
            transition.source_type.value
            if hasattr(transition.source_type, "value")
            else str(transition.source_type)
        )
        intent_key = ""
        if transition.intent:
            intent_key = str(getattr(transition.intent, "key", "") or "")
        if not intent_key:
            intent_key = str(transition.semantic_action_key or "")
        await tx.run(
            "MERGE (ent:TransitionEntity {stable_key: $stable_key}) "
            "SET ent.app_id = coalesce(ent.app_id, $app_id), "
            "    ent.from_state_id = coalesce(ent.from_state_id, $from_state_id), "
            "    ent.to_state_id = coalesce(ent.to_state_id, $to_state_id), "
            "    ent.action = coalesce(ent.action, $action), "
            "    ent.semantic_action_key = coalesce(ent.semantic_action_key, $semantic_action_key), "
            "    ent.created_at = coalesce(ent.created_at, $now)",
            stable_key=stable_key,
            app_id=None,
            from_state_id=str(transition.from_state_id or ""),
            to_state_id=str(transition.to_state_id or ""),
            action=action_val,
            semantic_action_key=str(transition.semantic_action_key or ""),
            now=datetime.now(UTC).isoformat(),
        )
        await tx.run(
            "MERGE (rev:TransitionRevision {revision_id: $revision_id}) "
            "SET rev.stable_key = $stable_key, "
            "    rev.transition_id = $transition_id, "
            "    rev.confidence = $confidence, "
            "    rev.intent_key = $intent_key, "
            "    rev.source_type = $source_type, "
            "    rev.operator_id = $operator_id, "
            "    rev.selector = $selector, "
            "    rev.action = $action, "
            "    rev.from_state_id = $from_state_id, "
            "    rev.to_state_id = $to_state_id, "
            "    rev.session_id = $session_id, "
            "    rev.ingest_version_id = $ingest_version_id, "
            "    rev.created_at = $now, "
            "    rev.is_active = $is_active",
            revision_id=revision_id,
            stable_key=stable_key,
            transition_id=transition_id,
            confidence=max(0.0, min(1.0, float(transition.confidence))),
            intent_key=intent_key or None,
            source_type=source_value or "auto",
            operator_id=str(getattr(transition, "operator_id", "agent") or "agent"),
            selector=transition.selector,
            action=action_val,
            from_state_id=str(transition.from_state_id or ""),
            to_state_id=str(transition.to_state_id or ""),
            session_id=session_id,
            ingest_version_id=str(getattr(transition, "ingest_version_id", "") or ""),
            now=datetime.now(UTC).isoformat(),
            is_active=active,
        )
        await tx.run(
            "MATCH (ent:TransitionEntity {stable_key: $stable_key}), "
            "(rev:TransitionRevision {revision_id: $revision_id}) "
            "MERGE (ent)-[:HAS_REVISION]->(rev)",
            stable_key=stable_key,
            revision_id=revision_id,
        )
        for old_revision_id in supersedes_revision_ids:
            await tx.run(
                "MATCH (new:TransitionRevision {revision_id: $new_revision_id}), "
                "(old:TransitionRevision {revision_id: $old_revision_id}) "
                "MERGE (new)-[:SUPERSEDES]->(old)",
                new_revision_id=revision_id,
                old_revision_id=old_revision_id,
            )
        return revision_id

    async def _merge_transition(self, tx, transition: Transition, session_id: str) -> str:
        action_val = (
            transition.action.value
            if hasattr(transition.action, "value")
            else str(transition.action)
        )
        result = await tx.run(
            "MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State) "
            "WHERE s1.id = $from_id AND s2.id = $to_id "
            "AND t.action = $action AND t.selector = $selector "
            "RETURN t.id AS id, t.confidence AS confidence, "
            "coalesce(t.source_type, 'auto') AS source_type",
            from_id=transition.from_state_id,
            to_id=transition.to_state_id,
            action=action_val,
            selector=transition.selector,
        )
        existing = await result.single()
        now = datetime.now(UTC).isoformat()
        stable_key = self._transition_stable_key(transition)
        active_revision_result = await tx.run(
            "MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision {is_active: true}) "
            "RETURN rev.revision_id AS revision_id, rev.source_type AS source_type, rev.confidence AS confidence "
            "ORDER BY rev.created_at DESC LIMIT 1",
            stable_key=stable_key,
        )
        active_revision = await active_revision_result.single()
        incoming_source = (
            transition.source_type.value
            if hasattr(transition.source_type, "value")
            else str(transition.source_type)
        )
        incoming_priority = _source_priority(incoming_source)
        incoming_conf = max(0.0, min(1.0, float(transition.confidence)))
        should_activate_revision = True
        supersedes_revision_ids: list[str] = []
        if active_revision is not None:
            current_priority = _source_priority(str(active_revision.get("source_type") or "auto"))
            current_conf = float(active_revision.get("confidence") or 0.0)
            if incoming_priority > current_priority:
                should_activate_revision = True
            elif incoming_priority < current_priority:
                should_activate_revision = False
            else:
                should_activate_revision = incoming_conf >= current_conf
            if should_activate_revision:
                old_revision_id = str(active_revision.get("revision_id") or "")
                if old_revision_id:
                    supersedes_revision_ids.append(old_revision_id)

        if existing is None:
            initial_conf = max(0.5, transition.confidence)
            failed_reqs_json = None
            if transition.failed_requests:
                try:
                    normalized: list[dict[str, object]] = []
                    for fr in transition.failed_requests:
                        if isinstance(fr, dict):
                            normalized.append(fr)
                        elif hasattr(fr, "model_dump"):
                            normalized.append(fr.model_dump(mode="json"))
                    failed_reqs_json = json.dumps(normalized, ensure_ascii=False)
                except Exception:
                    pass

            # Serialize steps to JSON for storage
            steps_json = None
            if transition.steps:
                try:
                    steps_json = json.dumps(
                        [s.model_dump(mode="json") for s in transition.steps],
                        ensure_ascii=False,
                    )
                except Exception:
                    pass

            logger.info(
                "[Neo4j] CREATE Transition id=%s action=%s %s→%s steps=%d",
                transition.id, action_val,
                str(transition.from_state_id or "")[:24],
                str(transition.to_state_id or "")[:24],
                len(transition.steps) if transition.steps else 0,
            )

            await tx.run(
                "CREATE (t:Transition {id: $id, selector: $selector, action: $action, "
                "action_value: $action_value, param_name: $param_name, thought: $thought, "
                "confidence: $conf, first_discovered: $now, session_id: $sid, "
                "step_index: $step_index, element_snapshot: $elem, frame_path: $fp, "
                "tab_id: $tab_id, target_tab_id: $target_tab_id, tab_action: $tab_action, "
                "name: $name, failed_requests: $failed_requests, validation_count: 0, "
                "semantic_action_key: $sak, steps: $steps})",
                id=transition.id,
                selector=transition.selector,
                action=action_val,
                action_value=transition.action_value,
                param_name=transition.param_name,
                thought=transition.thought,
                conf=initial_conf,
                now=now,
                sid=session_id,
                step_index=transition.step_index,
                elem=transition.element_snapshot,
                fp=transition.frame_path,
                tab_id=transition.tab_id,
                target_tab_id=transition.target_tab_id,
                tab_action=str(transition.tab_action) if transition.tab_action else None,
                name=transition.selector or str(transition.action) or transition.id,
                failed_requests=failed_reqs_json,
                sak=getattr(transition, "semantic_action_key", None),
                steps=steps_json,
            )
            if transition.from_state_id:
                await tx.run(
                    "MATCH (t:Transition {id: $tid}), (s:State {id: $sid}) "
                    "MERGE (t)-[:FROM]->(s)",
                    tid=transition.id,
                    sid=transition.from_state_id,
                )
            if transition.to_state_id:
                await tx.run(
                    "MATCH (t:Transition {id: $tid}), (s:State {id: $sid}) "
                    "MERGE (t)-[:TO]->(s)",
                    tid=transition.id,
                    sid=transition.to_state_id,
                )
            await tx.run(
                "MATCH (sess:Session {id: $sess_id}), (t:Transition {id: $tid}) "
                "MERGE (sess)-[:DISCOVERED]->(t)",
                sess_id=session_id,
                tid=transition.id,
            )

            # Create Intent node and REALIZES relationship if intent is present
            if transition.intent:
                intent = transition.intent
                intent_id = intent.id or intent.key or f"intent:{intent.summary or session_id}"
                logger.info(
                    "[Neo4j] MERGE Intent id=%s key=%s transition=%s",
                    intent_id, getattr(intent, "key", ""), transition.id,
                )
                await tx.run(
                    "MERGE (i:Intent {id: $intent_id}) "
                    "SET i.name = coalesce(i.name, $name, $summary, $key, $raw), "
                    "    i.summary = coalesce(i.summary, $summary, $name, $raw), "
                    "    i.key = coalesce(i.key, $key, $raw), "
                    "    i.verb = coalesce(i.verb, $verb), "
                    "    i.object = coalesce(i.object, $object), "
                    "    i.raw = coalesce(i.raw, $raw), "
                    "    i.confidence = CASE WHEN i.confidence IS NULL OR $confidence > i.confidence "
                    "                        THEN $confidence ELSE i.confidence END",
                    intent_id=intent_id,
                    name=intent.name or "",
                    summary=intent.summary or "",
                    key=intent.key or "",
                    verb=intent.verb or "",
                    object=intent.object or "",
                    raw=intent.raw or "",
                    confidence=intent.confidence if intent.confidence is not None else 0.5,
                )
                await tx.run(
                    "MATCH (t:Transition {id: $tid}), (i:Intent {id: $intent_id}) "
                    "MERGE (t)-[:REALIZES]->(i)",
                    tid=transition.id,
                    intent_id=intent_id,
                )

            # Conflict resolution: prefer shorter selector among same (from, to) pair
            conflict_result = await tx.run(
                "MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State) "
                "WHERE s1.id = $from_id AND s2.id = $to_id "
                "AND t.id <> $tid "
                "RETURN t.id AS id, t.selector AS sel, t.confidence AS conf, "
                "coalesce(t.source_type, 'auto') AS source_type",
                from_id=transition.from_state_id,
                to_id=transition.to_state_id,
                tid=transition.id,
            )
            conflicts = [record async for record in conflict_result]
            new_sel_len = len(transition.selector or "")
            source_type_val = getattr(transition, "source_type", None)
            if hasattr(source_type_val, "value"):
                new_priority = _source_priority(source_type_val.value)  # type: ignore[union-attr]
            else:
                new_priority = _source_priority(str(source_type_val or "auto"))
            for rec in conflicts:
                old_priority = _source_priority(str(rec.get("source_type") or "auto"))
                if new_priority > old_priority:
                    await tx.run(
                        "MATCH (t:Transition {id: $tid}) SET t.confidence = CASE "
                        "WHEN t.confidence + 0.15 > 1.0 THEN 1.0 ELSE t.confidence + 0.15 END",
                        tid=transition.id,
                    )
                    await tx.run(
                        "MATCH (t:Transition {id: $oid}) SET t.confidence = CASE "
                        "WHEN t.confidence - 0.15 < 0 THEN 0.0 ELSE t.confidence - 0.15 END",
                        oid=rec["id"],
                    )
                    continue
                if new_priority < old_priority:
                    await tx.run(
                        "MATCH (t:Transition {id: $tid}) SET t.confidence = CASE "
                        "WHEN t.confidence - 0.15 < 0 THEN 0.0 ELSE t.confidence - 0.15 END",
                        tid=transition.id,
                    )
                    continue
                old_sel_len = len(rec["sel"] or "")
                if new_sel_len < old_sel_len:
                    # New transition has shorter selector: boost new, degrade old
                    await tx.run(
                        "MATCH (t:Transition {id: $tid}) SET t.confidence = CASE "
                        "WHEN t.confidence + 0.1 > 1.0 THEN 1.0 ELSE t.confidence + 0.1 END",
                        tid=transition.id,
                    )
                    await tx.run(
                        "MATCH (t:Transition {id: $oid}) SET t.confidence = CASE "
                        "WHEN t.confidence - 0.1 < 0 THEN 0.0 ELSE t.confidence - 0.1 END",
                        oid=rec["id"],
                    )
                elif new_sel_len > old_sel_len:
                    # Old transition has shorter selector: degrade new
                    await tx.run(
                        "MATCH (t:Transition {id: $tid}) SET t.confidence = CASE "
                        "WHEN t.confidence - 0.1 < 0 THEN 0.0 ELSE t.confidence - 0.1 END",
                        tid=transition.id,
                    )

            if should_activate_revision:
                await tx.run(
                    "MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision {is_active: true}) "
                    "SET rev.is_active = false",
                    stable_key=stable_key,
                )
            await self._write_transition_revision(
                tx,
                stable_key=stable_key,
                transition=transition,
                transition_id=transition.id,
                session_id=session_id,
                active=should_activate_revision,
                supersedes_revision_ids=supersedes_revision_ids,
            )

            return "created"

        existing_priority = _source_priority(str(existing.get("source_type") or "auto"))
        incoming_source = (
            transition.source_type.value
            if hasattr(transition.source_type, "value")
            else str(transition.source_type)
        )
        incoming_priority = _source_priority(incoming_source)
        if incoming_priority > existing_priority:
            await tx.run(
                "MATCH (t:Transition {id: $id}) "
                "SET t.source_type = $source_type, t.operator_id = $operator_id, "
                "    t.confidence = CASE WHEN t.confidence < $incoming_conf THEN $incoming_conf ELSE t.confidence END, "
                "    t.last_validated = $now, "
                "    t.validation_count = t.validation_count + 1",
                id=existing["id"],
                source_type=incoming_source,
                operator_id=getattr(transition, "operator_id", "agent"),
                incoming_conf=max(0.1, min(1.0, float(transition.confidence))),
                now=now,
            )
            target_transition_id = str(existing["id"])
            if should_activate_revision:
                await tx.run(
                    "MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision {is_active: true}) "
                    "SET rev.is_active = false",
                    stable_key=stable_key,
                )
            await self._write_transition_revision(
                tx,
                stable_key=stable_key,
                transition=transition,
                transition_id=target_transition_id,
                session_id=session_id,
                active=should_activate_revision,
                supersedes_revision_ids=supersedes_revision_ids,
            )
            return "updated"

        new_conf = min(1.0, existing["confidence"] + 0.2)
        await tx.run(
            "MATCH (t:Transition {id: $id}) "
            "SET t.confidence = $conf, t.last_validated = $now, "
            "    t.validation_count = t.validation_count + 1",
            id=existing["id"],
            conf=new_conf,
            now=now,
        )
        await tx.run(
            "MATCH (sess:Session {id: $sess_id}), (t:Transition {id: $tid}) "
            "MERGE (sess)-[:VALIDATED]->(t)",
            sess_id=session_id,
            tid=existing["id"],
        )
        if should_activate_revision:
            await tx.run(
                "MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision {is_active: true}) "
                "SET rev.is_active = false",
                stable_key=stable_key,
            )
        await self._write_transition_revision(
            tx,
            stable_key=stable_key,
            transition=transition,
            transition_id=str(existing["id"]),
            session_id=session_id,
            active=should_activate_revision,
            supersedes_revision_ids=supersedes_revision_ids,
        )
        return "boosted"

    # ── Zone ─────────────────────────────────────────────────

    async def _merge_zone(self, tx, zone: Zone, state_id: str | None = None) -> bool:
        result = await tx.run(
            "MATCH (z:Zone {id: $id}) RETURN z.id",
            id=zone.id,
        )
        existing = await result.single()
        if existing is None:
            zone_type = (
                zone.zone_type.value
                if hasattr(zone.zone_type, "value")
                else str(zone.zone_type)
            )
            await tx.run(
                "CREATE (z:Zone {id: $id, zone_type: $zt, root_selector: $rs, "
                "summary: $summary, interactive_count: $ic, "
                "exploration_status: 'discovered'})",
                id=zone.id,
                zt=zone_type,
                rs=zone.root_selector,
                summary=zone.summary,
                ic=zone.interactive_count,
            )
            if state_id:
                await tx.run(
                    "MATCH (s:State {id: $sid}), (z:Zone {id: $zid}) "
                    "MERGE (s)-[:HAS_ZONE]->(z)",
                    sid=state_id,
                    zid=zone.id,
                )
            return True
        return False

    # ── Checkpoint ───────────────────────────────────────────

    async def _merge_checkpoint(
        self, tx, cp: Checkpoint, session_id: str, transition_id: str | None = None
    ) -> None:
        layer = cp.layer.value if hasattr(cp.layer, "value") else str(cp.layer)
        timing = cp.timing.value if hasattr(cp.timing, "value") else str(cp.timing)
        expect = cp.expect.value if hasattr(cp.expect, "value") else str(cp.expect)
        severity = cp.severity.value if hasattr(cp.severity, "value") else str(cp.severity)
        origin = (
            cp.origin_type.value if hasattr(cp.origin_type, "value") else str(cp.origin_type)
        )
        await tx.run(
            "MERGE (c:Checkpoint {id: $id}) "
            "SET c.layer = $layer, c.timing = $timing, c.expect = $expect, "
            "c.severity = $severity, c.rule_type = $rt, c.rule = $rule, "
            "c.description = $desc, c.origin_type = $origin, c.session_id = $sid",
            id=cp.id,
            layer=layer,
            timing=timing,
            expect=expect,
            severity=severity,
            rt=cp.rule_type,
            rule=cp.rule,
            desc=cp.description,
            origin=origin,
            sid=session_id,
        )
        await tx.run(
            "MATCH (sess:Session {id: $sess_id}), (c:Checkpoint {id: $cid}) "
            "MERGE (sess)-[:GENERATED]->(c)",
            sess_id=session_id,
            cid=cp.id,
        )
        if transition_id:
            rel_type = "CHECK_AFTER" if timing == "after" else "CHECK_BEFORE"
            await tx.run(
                f"MATCH (t:Transition {{id: $tid}}), (c:Checkpoint {{id: $cid}}) "
                f"MERGE (t)-[:{rel_type}]->(c)",
                tid=transition_id,
                cid=cp.id,
            )

    # ── Validation (standalone, outside merge tx) ────────────

    async def update_transition_confidence(
        self, transition_id: str, passed: bool, session_id: str
    ) -> None:
        delta = 0.2 if passed else -0.3
        async with self._driver.session() as session:
            await session.run(
                "MATCH (t:Transition {id: $id}) "
                "SET t.confidence = CASE "
                "  WHEN t.confidence + $delta > 1.0 THEN 1.0 "
                "  WHEN t.confidence + $delta < 0.0 THEN 0.0 "
                "  ELSE t.confidence + $delta END, "
                "t.last_validated = $now, "
                "t.validation_count = t.validation_count + 1",
                id=transition_id,
                delta=delta,
                now=datetime.now(UTC).isoformat(),
            )
            rel = "VALIDATED" if passed else "INVALIDATED"
            await session.run(
                f"MATCH (sess:Session {{id: $sid}}), (t:Transition {{id: $tid}}) "
                f"MERGE (sess)-[:{rel}]->(t)",
                sid=session_id,
                tid=transition_id,
            )

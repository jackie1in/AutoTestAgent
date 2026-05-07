class CypherQueries:
    """Centralized Cypher query templates."""

    # ===== App =====
    UPSERT_APP = """
    MERGE (a:App {id: $id})
    SET a += $props
    SET a.name = coalesce(a.name, a.entry_url, a.id)
    RETURN a
    """

    GET_APP_BY_ID = "MATCH (a:App {id: $id}) RETURN a"
    GET_ALL_APPS = "MATCH (a:App) RETURN a ORDER BY a.last_session_at DESC"

    LINK_APP_STATE = """
    MATCH (a:App {id: $app_id}), (s:State {id: $state_id})
    MERGE (a)-[:HAS_STATE]->(s)
    """

    LINK_APP_SESSION = """
    MATCH (a:App {id: $app_id}), (sess:Session {id: $session_id})
    MERGE (a)-[:HAS_SESSION]->(sess)
    """

    TOUCH_APP_LAST_SESSION = """
    MATCH (a:App {id: $app_id})
    SET a.last_session_at = datetime($last_session_at)
    """

    LINK_SESSION_INGESTION_RUN = """
    MATCH (sess:Session {id: $session_id}), (run:IngestionRun {id: $ingest_id})
    MERGE (sess)-[:GENERATES]->(run)
    """

    SET_SESSION_INVENTORY = """
    MATCH (s:Session {id: $session_id})
    SET s.inventory = $inventory
    """

    UPDATE_SESSION_STATS = """
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
        s.manual_transition_count = $manual_transition_count,
        s.auto_transition_count = $auto_transition_count,
        s.intervention_task_count = $intervention_task_count,
        s.intervention_tasks = $intervention_tasks,
        s.intervention_tasks_json = $intervention_tasks_json,
        s.current_release_id = $current_release_id,
        s.latest_ingest_version_id = $latest_ingest_version_id,
        s.current_coverage_snapshot_id = $current_coverage_snapshot_id,
        s.dynamic_metrics_json = $dynamic_metrics_json,
        s.name = coalesce(s.focus, s.id)
    SET s += $dynamic_stats
    """

    GET_APP_STATES = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
    RETURN s ORDER BY s.last_visited DESC
    """

    GET_STATES_WITH_INTENTS = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
    OPTIONAL MATCH (s)<-[:FROM]-(t:Transition)-[:REALIZES]->(i:Intent)
    RETURN s.id as state_id, s.url as url, s.title as title,
           collect(DISTINCT i{.*}) as intents
    """

    GET_APP_SESSIONS = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(sess:Session)
    RETURN sess ORDER BY sess.timestamp DESC
    """

    UPDATE_APP_STATS = """
    MATCH (a:App {id: $app_id})
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WITH a, count(s) AS sc
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s2:State)<-[:FROM]-(t:Transition)
    WITH a, sc, count(DISTINCT t) AS tc
    OPTIONAL MATCH (a)-[:HAS_SESSION]->(sess:Session)
    WITH a, sc, tc, count(sess) AS sessc, max(sess.timestamp) AS last_sess
    SET a.total_states = sc,
        a.total_transitions = tc,
        a.total_sessions = sessc,
        a.last_session_at = coalesce(last_sess, a.last_session_at)
    """

    # ===== State =====
    UPSERT_STATE = """
    MERGE (s:State {id: $id})
    SET s += $props
    SET s.name = coalesce(s.title, s.url, s.id)
    RETURN s
    """

    GET_STATE_BY_ID = "MATCH (s:State {id: $id}) RETURN s"
    GET_STATE_BY_URL = "MATCH (s:State {url: $url}) RETURN s"
    GET_ALL_STATES = "MATCH (s:State) RETURN s ORDER BY s.last_visited DESC"

    # ===== Transition =====
    UPSERT_TRANSITION = """
    MERGE (t:Transition {id: $id})
    SET t += $props
    SET t.name = coalesce(t.selector, t.action, t.id)
    WITH t
    MATCH (from_state:State {id: $from_state_id})
    MERGE (t)-[:FROM]->(from_state)
    WITH t
    MATCH (to_state:State {id: $to_state_id})
    MERGE (t)-[:TO]->(to_state)
    RETURN t
    """

    GET_TRANSITIONS_FROM_STATE = """
    MATCH (s:State {id: $state_id})<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    RETURN t, target
    """

    GET_ALL_TRANSITIONS = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    RETURN t.id AS id,
           t.selector AS selector,
           t.action AS action,
           t.action_value AS action_value,
           t.param_name AS param_name,
           t.element_snapshot AS element_snapshot,
           t.frame_path AS frame_path,
           t.tab_id AS tab_id,
           t.target_tab_id AS target_tab_id,
           t.tab_action AS tab_action,
           t.thought AS thought,
           t.confidence AS confidence,
           t.step_index AS step_index,
           t.session_id AS session_id,
           t.intent AS intent,
           t.intent_failure_reason AS intent_failure_reason,
           t.selector_chain AS selector_chain,
           t.semantic_action_key AS semantic_action_key,
           t.evidence_ids AS evidence_ids,
           s.id AS from_state_id,
           s.url AS source_url,
           target.id AS to_state_id,
           target.url AS target_url
    ORDER BY t.confidence DESC
    """

    GET_ALL_TRANSITIONS_BY_RELEASE = """
    MATCH (r:GraphRelease {id: $release_id, app_id: $app_id, status: 'active'})
          <-[:IN_RELEASE]-(rev:TransitionRevision {is_active: true})
          <-[:HAS_REVISION]-(ent:TransitionEntity)
    MATCH (t:Transition {id: rev.transition_id})
    OPTIONAL MATCH (s:State {id: rev.from_state_id})
    OPTIONAL MATCH (target:State {id: rev.to_state_id})
    RETURN t.id AS id,
           t.selector AS selector,
           t.action AS action,
           t.action_value AS action_value,
           t.param_name AS param_name,
           t.element_snapshot AS element_snapshot,
           t.frame_path AS frame_path,
           t.tab_id AS tab_id,
           t.target_tab_id AS target_tab_id,
           t.tab_action AS tab_action,
           t.thought AS thought,
           rev.confidence AS confidence,
           t.step_index AS step_index,
           t.session_id AS session_id,
           t.intent AS intent,
           t.intent_failure_reason AS intent_failure_reason,
           t.selector_chain AS selector_chain,
           t.semantic_action_key AS semantic_action_key,
           t.evidence_ids AS evidence_ids,
           rev.source_type AS source_type,
           rev.operator_id AS operator_id,
           rev.ingest_version_id AS ingest_version_id,
           s.id AS from_state_id,
           s.url AS source_url,
           target.id AS to_state_id,
           target.url AS target_url
    ORDER BY rev.confidence DESC
    """

    GET_LOW_CONFIDENCE_TRANSITIONS = """
    MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    WHERE t.confidence < $threshold
    RETURN s1, t, s2 ORDER BY t.confidence ASC
    """

    # ===== Zone =====
    UPSERT_ZONE = """
    MERGE (z:Zone {id: $id})
    SET z += $props
    SET z.name = coalesce(z.summary, z.zone_type, z.id)
    RETURN z
    """

    LINK_STATE_ZONE = """
    MATCH (s:State {id: $state_id}), (z:Zone {id: $zone_id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    UPSERT_ZONES_FOR_APP = """
    UNWIND $zones as zone
    MERGE (z:Zone {id: zone.id})
    SET z.type = zone.type,
        z.selector = zone.selector,
        z.element_count = zone.element_count,
        z.bounds = zone.bounds,
        z.text_sample = zone.text_sample,
        z.ingest_version_id = zone.ingest_version_id,
        z.name = coalesce(zone.summary, zone.type, zone.id),
        z.updated_at = datetime()
    WITH z, zone,
         coalesce(z.exploration_status, 'undiscovered') AS prev_status,
         coalesce(zone.exploration_status, 'discovered') AS new_status
    WITH z, zone, prev_status, new_status,
         CASE prev_status
             WHEN 'validated' THEN 4
             WHEN 'explored' THEN 3
             WHEN 'partial' THEN 2
             WHEN 'discovered' THEN 1
             ELSE 0
         END AS prev_p,
         CASE new_status
             WHEN 'validated' THEN 4
             WHEN 'explored' THEN 3
             WHEN 'partial' THEN 2
             WHEN 'discovered' THEN 1
             ELSE 0
         END AS new_p
    SET z.exploration_status = CASE WHEN new_p >= prev_p THEN new_status ELSE prev_status END,
        z.last_explored = CASE
            WHEN zone.last_explored IS NOT NULL
              THEN datetime(zone.last_explored)
            ELSE z.last_explored
        END
    WITH z
    MATCH (a:App {id: $app_id})
    MERGE (a)-[:HAS_ZONE]->(z)
    """

    LINK_STATE_ZONES_DIRECT = """
    UNWIND $zones as zone
    MATCH (s:State {id: $state_id}), (z:Zone {id: zone.id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    LINK_STATE_ZONES_BY_IDS = """
    UNWIND $zones as zone
    UNWIND coalesce(zone.state_ids, []) as sid
    MATCH (s:State {id: sid}), (z:Zone {id: zone.id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    LINK_TRANSITION_ZONE = """
    MATCH (t:Transition {id: $transition_id}), (z:Zone {id: $zone_id})
    MERGE (t)-[:IN_ZONE]->(z)
    """

    GET_UNEXPLORED_ZONES = """
    MATCH (s:State)-[:HAS_ZONE]->(z:Zone)
    WHERE z.exploration_status IN ['undiscovered', 'discovered']
    RETURN s, z ORDER BY z.exploration_status
    """

    # ===== Frame =====
    UPSERT_FRAME = """
    MERGE (f:Frame {id: $id})
    SET f += $props
    SET f.name = coalesce(f.name, f.src, f.selector, f.id)
    RETURN f
    """

    LINK_STATE_FRAME = """
    MATCH (s:State {id: $state_id}), (f:Frame {id: $frame_id})
    MERGE (s)-[:HAS_FRAME {depth: $depth}]->(f)
    """

    LINK_FRAME_PARENT = """
    MATCH (parent:Frame {id: $parent_id}), (child:Frame {id: $child_id})
    MERGE (parent)-[:CONTAINS_FRAME]->(child)
    """

    LINK_TRANSITION_FRAME = """
    MATCH (t:Transition {id: $transition_id}), (f:Frame {id: $frame_id})
    MERGE (t)-[:IN_FRAME]->(f)
    """

    # ===== Intent =====
    UPSERT_INTENT = """
    MERGE (i:Intent {id: $id})
    SET i += $props
    SET i.name = coalesce(i.name, i.summary, i.key, i.id)
    RETURN i
    """

    LINK_TRANSITION_INTENT = """
    MATCH (t:Transition {id: $transition_id}), (i:Intent {id: $intent_id})
    MERGE (t)-[:REALIZES]->(i)
    """

    LINK_INTENT_REQUIRES_ENTITY = """
    MATCH (i:Intent {id: $intent_id}), (e:Entity {id: $entity_id})
    MERGE (i)-[:REQUIRES_ENTITY {field_mapping: $field_mapping}]->(e)
    """

    LINK_INTENT_PRODUCES_ENTITY = """
    MATCH (i:Intent {id: $intent_id}), (e:Entity {id: $entity_id})
    MERGE (i)-[:PRODUCES_ENTITY {field_mapping: $field_mapping}]->(e)
    """

    LINK_INTENT_PRECONDITION = """
    MATCH (i:Intent {id: $intent_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (i)-[:HAS_PRECONDITION]->(c)
    """

    LINK_INTENT_POSTCONDITION = """
    MATCH (i:Intent {id: $intent_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (i)-[:HAS_POSTCONDITION]->(c)
    """

    LINK_INTENT_DATA_FLOW = """
    MATCH (a:Intent {id: $from_intent_id}), (b:Intent {id: $to_intent_id})
    MERGE (a)-[:DATA_FLOW {from_slot: $from_slot, to_slot: $to_slot}]->(b)
    """

    LINK_INTENT_ALTERNATIVE = """
    MATCH (a:Intent {id: $intent_id_a}), (b:Intent {id: $intent_id_b})
    MERGE (a)-[:ALTERNATIVE_OF]->(b)
    """

    LINK_INTENT_MUST_PRECEDE = """
    MATCH (a:Intent {id: $before_intent_id}), (b:Intent {id: $after_intent_id})
    MERGE (a)-[:MUST_PRECEDE]->(b)
    """

    # ===== Entity =====
    UPSERT_ENTITY = """
    MERGE (e:Entity {id: $id})
    SET e += $props
    SET e.name = coalesce(e.name, e.description, e.id)
    RETURN e
    """

    UPSERT_ENTITY_INSTANCE = """
    MERGE (ei:EntityInstance {id: $id})
    SET ei += $props
    SET ei.name = coalesce(ei.status, ei.id)
    WITH ei
    MATCH (e:Entity {id: $entity_id})
    MERGE (e)-[:HAS_INSTANCE]->(ei)
    RETURN ei
    """

    LINK_ENTITY_INSTANCE_INTENT = """
    MATCH (ei:EntityInstance {id: $instance_id}), (i:Intent {id: $intent_id})
    MERGE (ei)-[:CREATED_BY_INTENT]->(i)
    """

    # ===== Checkpoint =====
    UPSERT_CHECKPOINT = """
    MERGE (c:Checkpoint {id: $id})
    SET c += $props
    SET c.name = coalesce(c.description, c.rule_type, c.layer, c.id)
    RETURN c
    """

    LINK_TRANSITION_CHECK_BEFORE = """
    MATCH (t:Transition {id: $transition_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (t)-[:CHECK_BEFORE]->(c)
    """

    LINK_TRANSITION_CHECK_AFTER = """
    MATCH (t:Transition {id: $transition_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (t)-[:CHECK_AFTER]->(c)
    """

    LINK_STATE_CHECK = """
    MATCH (s:State {id: $state_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (s)-[:CHECK_STATE]->(c)
    """

    LINK_CHECKPOINT_ENTITY = """
    MATCH (c:Checkpoint {id: $checkpoint_id}), (e:Entity {id: $entity_id})
    MERGE (c)-[:CHECKS_ENTITY]->(e)
    """

    # ===== FieldConstraint =====
    UPSERT_FIELD_CONSTRAINT = """
    MERGE (fc:FieldConstraint {id: $id})
    SET fc += $props
    SET fc.name = coalesce(fc.field_name, fc.selector, fc.input_type, fc.id)
    RETURN fc
    """

    LINK_ZONE_FIELD = """
    MATCH (z:Zone {id: $zone_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (z)-[:HAS_FIELD]->(fc)
    """

    LINK_TRANSITION_OPERATES_ON = """
    MATCH (t:Transition {id: $transition_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (t)-[:OPERATES_ON]->(fc)
    """

    # ===== TestCase =====
    UPSERT_TEST_CASE = """
    MERGE (tc:TestCase {id: $id})
    SET tc += $props
    SET tc.name = coalesce(tc.name, tc.description, tc.category, tc.id)
    RETURN tc
    """

    LINK_TEST_CASE_TRANSITION = """
    MATCH (tc:TestCase {id: $test_case_id}), (t:Transition {id: $transition_id})
    MERGE (tc)-[:DERIVED_FROM]->(t)
    """

    LINK_TEST_CASE_CHECKPOINT = """
    MATCH (tc:TestCase {id: $test_case_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (tc)-[:EXPECTS]->(c)
    """

    LINK_TEST_CASE_COVERS = """
    MATCH (tc:TestCase {id: $test_case_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (tc)-[:COVERS]->(fc)
    """

    # ===== Evidence =====
    UPSERT_EVIDENCE = """
    MERGE (e:Evidence {id: $id})
    SET e += $props
    SET e.name = coalesce(e.summary, e.evidence_type, e.id)
    RETURN e
    """

    LINK_TRANSITION_EVIDENCE = """
    MATCH (t:Transition {id: $transition_id}), (e:Evidence {id: $evidence_id})
    MERGE (t)-[:SUPPORTED_BY]->(e)
    """

    LINK_SESSION_EVIDENCE = """
    MATCH (sess:Session {id: $session_id}), (e:Evidence {id: $evidence_id})
    MERGE (e)-[:OBSERVED_IN]->(sess)
    """

    GET_TRANSITION_EVIDENCE = """
    MATCH (t:Transition {id: $transition_id})-[:SUPPORTED_BY]->(e:Evidence)
    RETURN e
    ORDER BY coalesce(e.created_at, '') DESC
    """

    # ===== Session =====
    UPSERT_SESSION = """
    MERGE (s:Session {id: $id})
    SET s += $props
    SET s.name = coalesce(s.focus, s.id)
    RETURN s
    """

    LINK_SESSION_DISCOVERED_STATE = """
    MATCH (sess:Session {id: $session_id}), (s:State {id: $state_id})
    MERGE (sess)-[:DISCOVERED]->(s)
    """

    LINK_SESSION_DISCOVERED_TRANSITION = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:DISCOVERED]->(t)
    """

    LINK_INGESTION_EMITS_STATE = """
    MATCH (run:IngestionRun {id: $ingest_id}), (s:State {id: $state_id})
    MERGE (run)-[:EMITS]->(s)
    """

    LINK_INGESTION_EMITS_TRANSITION = """
    MATCH (run:IngestionRun {id: $ingest_id}), (t:Transition {id: $transition_id})
    MERGE (run)-[:EMITS]->(t)
    """

    LINK_INGESTION_EMITS_EVIDENCE = """
    MATCH (run:IngestionRun {id: $ingest_id}), (e:Evidence {id: $evidence_id})
    MERGE (run)-[:EMITS]->(e)
    """

    LINK_INGESTION_EMITS_MENU = """
    MATCH (run:IngestionRun {id: $ingest_id}), (m:Menu {id: $menu_id})
    MERGE (run)-[:EMITS]->(m)
    """

    LINK_INGESTION_EMITS_ZONE = """
    MATCH (run:IngestionRun {id: $ingest_id}), (z:Zone {id: $zone_id})
    MERGE (run)-[:EMITS]->(z)
    """

    UPSERT_INGESTION_RUN = """
    MERGE (run:IngestionRun {id: $id})
    SET run += $props
    RETURN run
    """

    UPSERT_TRANSITION_ENTITY = """
    MERGE (ent:TransitionEntity {stable_key: $stable_key})
    SET ent += $props
    RETURN ent
    """

    UPSERT_TRANSITION_REVISION = """
    MERGE (rev:TransitionRevision {revision_id: $revision_id})
    SET rev += $props
    RETURN rev
    """

    LINK_ENTITY_HAS_REVISION = """
    MATCH (ent:TransitionEntity {stable_key: $stable_key}),
          (rev:TransitionRevision {revision_id: $revision_id})
    MERGE (ent)-[:HAS_REVISION]->(rev)
    """

    DEACTIVATE_ACTIVE_REVISIONS = """
    MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision {is_active: true})
    SET rev.is_active = false
    RETURN collect(rev.revision_id) AS revision_ids
    """

    LINK_REVISION_SUPERSEDES = """
    MATCH (new:TransitionRevision {revision_id: $new_revision_id}),
          (old:TransitionRevision {revision_id: $old_revision_id})
    MERGE (new)-[:SUPERSEDES]->(old)
    """

    LINK_INGESTION_EMITS_REVISION = """
    MATCH (run:IngestionRun {id: $ingest_id}),
          (rev:TransitionRevision {revision_id: $revision_id})
    MERGE (run)-[:EMITS]->(rev)
    """

    GET_ACTIVE_REVISION_BY_STABLE_KEY = """
    MATCH (:TransitionEntity {stable_key: $stable_key})-[:HAS_REVISION]->(rev:TransitionRevision)
    WHERE rev.is_active = true
    RETURN rev
    ORDER BY rev.created_at DESC
    LIMIT 1
    """

    UPSERT_GRAPH_RELEASE = """
    MERGE (r:GraphRelease {id: $id})
    SET r += $props
    RETURN r
    """

    DEACTIVATE_OTHER_ACTIVE_RELEASES = """
    MATCH (r:GraphRelease {app_id: $app_id, status: 'active'})
    WHERE r.id <> $keep_release_id
    SET r.status = 'inactive',
        r.deactivated_at = datetime($now)
    """

    LINK_RELEASE_INCLUDES_REVISION = """
    MATCH (r:GraphRelease {id: $release_id}),
          (rev:TransitionRevision {revision_id: $revision_id})
    MERGE (rev)-[:IN_RELEASE]->(r)
    """

    UPSERT_COVERAGE_SNAPSHOT = """
    MERGE (cov:CoverageSnapshot {id: $id})
    SET cov += $props
    SET cov.name = coalesce(cov.session_id, cov.id)
    RETURN cov
    """

    LINK_SESSION_ACHIEVED_COVERAGE = """
    MATCH (sess:Session {id: $session_id}),
          (cov:CoverageSnapshot {id: $coverage_id})
    MERGE (sess)-[:ACHIEVED]->(cov)
    """

    LINK_RELEASE_HAS_COVERAGE = """
    MATCH (r:GraphRelease {id: $release_id}),
          (cov:CoverageSnapshot {id: $coverage_id})
    MERGE (r)-[:HAS_COVERAGE]->(cov)
    SET r.coverage_overall = cov.overall_completeness,
        r.coverage_zone = cov.zone_coverage,
        r.coverage_interaction = cov.interaction_coverage,
        r.coverage_state = cov.state_coverage,
        r.coverage_menu = cov.menu_coverage,
        r.coverage_recommendation = cov.recommendation
    """

    UPSERT_TRANSITION_ENTITY_WITH_SESSION = """
    MERGE (ent:TransitionEntity {stable_key: $stable_key})
    ON CREATE SET ent += $props,
                  ent.confirmed_session_count = 1,
                  ent.first_seen_session = $session_id,
                  ent.last_confirm_session = $session_id,
                  ent.last_confirmed_at = $now
    ON MATCH SET ent.app_id = coalesce(ent.app_id, $props.app_id),
                 ent.from_state_id = coalesce(ent.from_state_id, $props.from_state_id),
                 ent.to_state_id = coalesce(ent.to_state_id, $props.to_state_id),
                 ent.action = coalesce(ent.action, $props.action),
                 ent.semantic_action_key = coalesce(ent.semantic_action_key, $props.semantic_action_key),
                 ent.confirmed_session_count = CASE
                     WHEN ent.last_confirm_session = $session_id
                       THEN coalesce(ent.confirmed_session_count, 1)
                     ELSE coalesce(ent.confirmed_session_count, 0) + 1
                 END,
                 ent.last_confirm_session = $session_id,
                 ent.last_confirmed_at = $now,
                 ent.first_seen_session = coalesce(ent.first_seen_session, $session_id)
    RETURN ent
    """

    LINK_ZONE_COVERS_INTENT = """
    OPTIONAL MATCH (s:State {id: $from_state_id})-[:HAS_ZONE]->(z:Zone)
    WHERE z.selector IS NOT NULL
      AND z.selector <> ''
      AND ( $selector = z.selector
            OR $selector CONTAINS z.selector
            OR z.selector CONTAINS $selector )
    WITH collect(DISTINCT z) AS matched_zones
    WITH CASE WHEN size(matched_zones) > 0 THEN matched_zones ELSE [] END AS zones
    UNWIND zones AS z
    MATCH (i:Intent {id: $intent_id})
    MERGE (z)-[r:COVERS_INTENT]->(i)
    ON CREATE SET r.observed_count = 1,
                  r.first_confirmed_at = $now,
                  r.last_confirmed_at = $now,
                  r.confidence = $confidence,
                  r.last_session_id = $session_id
    ON MATCH SET r.observed_count = coalesce(r.observed_count, 0) + 1,
                 r.last_confirmed_at = $now,
                 r.last_session_id = $session_id,
                 r.confidence = CASE
                     WHEN $confidence > coalesce(r.confidence, 0.0)
                       THEN $confidence
                     ELSE r.confidence
                 END
    """

    LINK_SESSION_GENERATED_CHECKPOINT = """
    MATCH (sess:Session {id: $session_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (sess)-[:GENERATED]->(c)
    """

    LINK_SESSION_VALIDATED = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:VALIDATED]->(t)
    """

    LINK_SESSION_INVALIDATED = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:INVALIDATED]->(t)
    """

    LINK_SESSION_CREATED_ENTITY_INSTANCE = """
    MATCH (sess:Session {id: $session_id}), (ei:EntityInstance {id: $instance_id})
    MERGE (sess)-[:CREATED]->(ei)
    """

    # ===== Menu Navigation (New Schema: Menu as separate node) =====

    UPSERT_MENU = """
    MERGE (m:Menu {id: $id})
    SET m += $props
    RETURN m
    """

    LINK_APP_MENU = """
    MATCH (a:App {id: $app_id}), (m:Menu {id: $menu_id})
    MERGE (a)-[:HAS_MENU]->(m)
    """

    LINK_MENU_CHILD_OF = """
    MATCH (child:Menu {id: $child_id}), (parent:Menu {id: $parent_id})
    MERGE (child)-[:CHILD_OF {order_index: $order_index}]->(parent)
    """

    LINK_MENU_LEADS_TO = """
    MATCH (m:Menu {id: $menu_id}), (s:State {id: $state_id})
    MERGE (m)-[:LEADS_TO {first_seen: datetime($first_seen), session_id: $session_id}]->(s)
    """

    LINK_TRANSITION_NAVIGATED_VIA = """
    MATCH (t:Transition {id: $transition_id}), (m:Menu {id: $menu_id})
    MERGE (t)-[:NAVIGATED_VIA]->(m)
    """

    LINK_SESSION_DISCOVERED_MENU = """
    MATCH (sess:Session {id: $session_id}), (m:Menu {id: $menu_id})
    MERGE (sess)-[:DISCOVERED]->(m)
    """

    GET_MENU_BY_ID = "MATCH (m:Menu {id: $id}) RETURN m"

    GET_APP_MENUS = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
    RETURN m ORDER BY m.level, m.order_index
    """

    GET_MENU_TREE = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
    OPTIONAL MATCH (m)-[:CHILD_OF]->(parent:Menu)
    OPTIONAL MATCH (m)-[:LEADS_TO]->(s:State)
    RETURN m, parent.id AS parent_id, s.id AS state_id
    ORDER BY m.level, m.order_index
    """

    GET_MENU_LEADS_TO_STATE = """
    MATCH (m:Menu {id: $menu_id})-[:LEADS_TO]->(s:State)
    RETURN s
    """

    MARK_PAGE_MENUS_INACTIVE = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu {page_url: $page_url})
    SET m.is_active = false,
        m.inactive_at = datetime($inactive_at),
        m.updated_at = datetime($inactive_at)
    """

    # ===== Legacy Menu Navigation (Deprecated: State-based menu links) =====

    LINK_MENU_PARENT_LEGACY = """
    MATCH (child:State {id: $child_id}), (parent:State {id: $parent_id})
    MERGE (child)-[:MENU_PARENT]->(parent)
    """

    LINK_MENU_NEXT_LEGACY = """
    MATCH (s:State {id: $state_id}), (next:State {id: $next_id})
    MERGE (s)-[:MENU_NEXT]->(next)
    """

    # ===== Graph Queries =====
    SHORTEST_PATH = """
    MATCH path = shortestPath(
        (s1:State {id: $from_id})-[:FROM|TO*]-(s2:State {id: $to_id})
    )
    RETURN path
    """

    COVERAGE_STATS = """
    OPTIONAL MATCH (s:State) WITH count(s) AS total_states
    OPTIONAL MATCH (z:Zone) WITH total_states, count(z) AS total_zones
    OPTIONAL MATCH (z2:Zone) WHERE z2.exploration_status IN ['explored', 'validated']
    WITH total_states, total_zones, count(z2) AS explored_zones
    OPTIONAL MATCH (t:Transition) WITH total_states, total_zones, explored_zones, count(t) AS total_transitions
    OPTIONAL MATCH (t2:Transition) WHERE t2.confidence >= 0.8
    WITH total_states, total_zones, explored_zones, total_transitions, count(t2) AS high_conf
    OPTIONAL MATCH (t3:Transition) WHERE t3.confidence >= 0.4 AND t3.confidence < 0.8
    WITH total_states, total_zones, explored_zones, total_transitions, high_conf, count(t3) AS med_conf
    OPTIONAL MATCH (t4:Transition) WHERE t4.confidence < 0.4
    RETURN total_states, total_zones, explored_zones, total_transitions, high_conf, med_conf, count(t4) AS low_conf
    """

    COVERAGE_STATS_BY_APP = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
    WITH count(s) AS total_states, collect(s) AS states
    UNWIND states AS st
    OPTIONAL MATCH (st)-[:HAS_ZONE]->(z:Zone)
    WITH total_states, states, collect(DISTINCT z) AS all_zones
    WITH total_states, states, size(all_zones) AS total_zones,
         size([z IN all_zones WHERE z.exploration_status IN ['explored', 'validated']]) AS explored_zones
    UNWIND states AS st2
    OPTIONAL MATCH (st2)<-[:FROM]-(t:Transition)
    WITH total_states, total_zones, explored_zones, collect(DISTINCT t) AS all_trans
    WITH total_states, total_zones, explored_zones,
         size(all_trans) AS total_transitions,
         size([t IN all_trans WHERE t.confidence >= 0.8]) AS high_conf,
         size([t IN all_trans WHERE t.confidence >= 0.4 AND t.confidence < 0.8]) AS med_conf,
         size([t IN all_trans WHERE t.confidence < 0.4]) AS low_conf
    RETURN total_states, total_zones, explored_zones, total_transitions, high_conf, med_conf, low_conf
    """

    FULL_GRAPH_EXPORT = """
    MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    OPTIONAL MATCH (t)-[:IN_ZONE]->(z:Zone)
    RETURN s1, t, s2, i, z
    """

    FULL_GRAPH_EXPORT_BY_APP = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    OPTIONAL MATCH (t)-[:IN_ZONE]->(z:Zone)
    RETURN s1, t, s2, i, z
    """

    SKIP_ADVISOR_BASE_BY_APP_ID = """
    MATCH (a:App {id: $app_id})
    OPTIONAL MATCH (rel:GraphRelease {app_id: a.id, status: 'active'})
    WITH a, rel
    ORDER BY rel.created_at DESC
    WITH a, head(collect(rel)) AS active_release
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WHERE split(split(coalesce(s.url, ''), '?')[0], '#')[0] = $url_clean
    OPTIONAL MATCH (s)-[:HAS_ZONE]->(z:Zone)
    OPTIONAL MATCH (z)-[ci:COVERS_INTENT]->(:Intent)
    WITH active_release, s, z, sum(coalesce(ci.observed_count, 0)) AS zone_intent_total
    RETURN
        count(DISTINCT s) AS state_count,
        max(s.last_visited) AS last_visited,
        coalesce(active_release.coverage_overall, 0.0) AS release_coverage,
        collect({
            selector: coalesce(z.selector, ''),
            status: coalesce(z.exploration_status, 'undiscovered'),
            last_explored: z.last_explored,
            intent_confirm: zone_intent_total
        }) AS zones
    """

    SKIP_ADVISOR_BASE_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a
    ORDER BY coalesce(a.last_session_at, a.created_at) DESC
    LIMIT 1
    OPTIONAL MATCH (rel:GraphRelease {app_id: a.id, status: 'active'})
    WITH a, rel
    ORDER BY rel.created_at DESC
    WITH a, head(collect(rel)) AS active_release
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WHERE split(split(coalesce(s.url, ''), '?')[0], '#')[0] = $url_clean
    OPTIONAL MATCH (s)-[:HAS_ZONE]->(z:Zone)
    OPTIONAL MATCH (z)-[ci:COVERS_INTENT]->(:Intent)
    WITH active_release, s, z, sum(coalesce(ci.observed_count, 0)) AS zone_intent_total
    RETURN
        count(DISTINCT s) AS state_count,
        max(s.last_visited) AS last_visited,
        coalesce(active_release.coverage_overall, 0.0) AS release_coverage,
        collect({
            selector: coalesce(z.selector, ''),
            status: coalesce(z.exploration_status, 'undiscovered'),
            last_explored: z.last_explored,
            intent_confirm: zone_intent_total
        }) AS zones
    """

    SKIP_ADVISOR_ENTITY_BY_APP_ID = """
    MATCH (a:App {id: $app_id})
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WHERE split(split(coalesce(s.url, ''), '?')[0], '#')[0] = $url_clean
    OPTIONAL MATCH (ent:TransitionEntity)
    WHERE s IS NOT NULL AND ent.from_state_id = s.id
    WITH coalesce(ent.confirmed_session_count, 0) AS n
    RETURN sum(CASE WHEN n > 1 THEN n - 1 ELSE 0 END) AS entity_confirm_total
    """

    SKIP_ADVISOR_ENTITY_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a
    ORDER BY coalesce(a.last_session_at, a.created_at) DESC
    LIMIT 1
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WHERE split(split(coalesce(s.url, ''), '?')[0], '#')[0] = $url_clean
    OPTIONAL MATCH (ent:TransitionEntity)
    WHERE s IS NOT NULL AND ent.from_state_id = s.id
    WITH coalesce(ent.confirmed_session_count, 0) AS n
    RETURN sum(CASE WHEN n > 1 THEN n - 1 ELSE 0 END) AS entity_confirm_total
    """

    KNOWLEDGE_RELEASE_ROWS_BY_APP_ID = """
    MATCH (a:App {id: $app_id})
    MATCH (r:GraphRelease {id: $release_id, app_id: a.id, status: 'active'})
          <-[:IN_RELEASE]-(rev:TransitionRevision {is_active: true})
    MATCH (t:Transition {id: rev.transition_id})
    OPTIONAL MATCH (s:State {id: rev.from_state_id})
    OPTIONAL MATCH (target:State {id: rev.to_state_id})
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    RETURN t.id AS id,
           rev.confidence AS confidence,
           t.selector AS selector,
           t.action AS action,
           s.url AS source_url,
           target.url AS target_url,
           i{.*} AS intent
    ORDER BY rev.confidence DESC
    LIMIT $limit
    """

    KNOWLEDGE_RELEASE_ROWS_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
    MATCH (r:GraphRelease {id: $release_id, app_id: a.id, status: 'active'})
          <-[:IN_RELEASE]-(rev:TransitionRevision {is_active: true})
    MATCH (t:Transition {id: rev.transition_id})
    OPTIONAL MATCH (s:State {id: rev.from_state_id})
    OPTIONAL MATCH (target:State {id: rev.to_state_id})
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    RETURN t.id AS id,
           rev.confidence AS confidence,
           t.selector AS selector,
           t.action AS action,
           s.url AS source_url,
           target.url AS target_url,
           i{.*} AS intent
    ORDER BY rev.confidence DESC
    LIMIT $limit
    """

    KNOWLEDGE_LEGACY_ROWS_BY_APP_ID = """
    MATCH (a:App {id: $app_id})
    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    RETURN t.id AS id,
           coalesce(t.confidence, 0.0) AS confidence,
           t.selector AS selector,
           t.action AS action,
           s.url AS source_url,
           target.url AS target_url,
           i{.*} AS intent
    ORDER BY confidence DESC
    LIMIT $limit
    """

    KNOWLEDGE_LEGACY_ROWS_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    RETURN t.id AS id,
           coalesce(t.confidence, 0.0) AS confidence,
           t.selector AS selector,
           t.action AS action,
           s.url AS source_url,
           target.url AS target_url,
           i{.*} AS intent
    ORDER BY confidence DESC
    LIMIT $limit
    """

    RUNNER_WARM_START_CANDIDATES_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    OPTIONAL MATCH (target)-[:HAS_ZONE]->(z:Zone)
    RETURN t.id AS transition_id,
           coalesce(t.confidence, 0.0) AS confidence,
           target.url AS target_url,
           count(z) = 0 AS zone_unexplored
    ORDER BY confidence DESC
    LIMIT $limit
    """

    RUNNER_WARM_START_CANDIDATES_BY_APP_ID = """
    MATCH (a:App {id: $app_id})
    MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    OPTIONAL MATCH (target)-[:HAS_ZONE]->(z:Zone)
    RETURN t.id AS transition_id,
           coalesce(t.confidence, 0.0) AS confidence,
           target.url AS target_url,
           count(z) = 0 AS zone_unexplored
    ORDER BY confidence DESC
    LIMIT $limit
    """

    RUNNER_STATE_URL_BY_ID = """
    MATCH (s:State {id: $sid})
    RETURN s.url AS url
    """

    RUNNER_ACTIVE_RELEASE_BY_APP_NAME = """
    MATCH (a:App {name: $app_name})
    WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
    MATCH (rel:GraphRelease {app_id: a.id, status: 'active'})
    RETURN rel.id AS id, rel.created_at AS created_at
    ORDER BY rel.created_at DESC
    LIMIT 1
    """

    RUNNER_ACTIVE_RELEASE_BY_APP_ID = """
    MATCH (rel:GraphRelease {app_id: $app_id, status: 'active'})
    RETURN rel.id AS id, rel.created_at AS created_at
    ORDER BY rel.created_at DESC
    LIMIT 1
    """

    SESSION_LATEST_SEMANTIC_BASELINE_BY_APP_ID = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(s:Session)
    WHERE s.id <> $exclude_session_id
      AND s.semantic_state_keys_json IS NOT NULL
      AND s.semantic_transition_keys_json IS NOT NULL
      AND s.semantic_intent_keys_json IS NOT NULL
    RETURN s.id AS session_id,
           s.semantic_state_keys_json AS semantic_state_keys_json,
           s.semantic_transition_keys_json AS semantic_transition_keys_json,
           s.semantic_intent_keys_json AS semantic_intent_keys_json
    ORDER BY s.timestamp DESC
    LIMIT 1
    """

    SESSION_SEMANTIC_STABILITY_TREND_BY_APP_ID = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(s:Session)
    WHERE s.semantic_stability_score IS NOT NULL
    WITH s
    ORDER BY s.timestamp DESC
    LIMIT $limit
    RETURN s.id AS session_id,
           s.timestamp AS timestamp,
           s.semantic_stability_mode AS mode,
           s.semantic_baseline_session_id AS baseline_session_id,
           coalesce(s.semantic_stability_score, 0.0) AS score,
           coalesce(s.semantic_stability_threshold, $default_threshold) AS threshold,
           coalesce(s.semantic_stability_passed, false) AS passed
    ORDER BY timestamp ASC
    """

    RETENTION_KEEP_RELEASE_IDS_BY_APP = """
    MATCH (r:GraphRelease {app_id: $app_id})
    WITH r
    ORDER BY coalesce(r.created_at, r.deactivated_at) DESC
    LIMIT $keep_releases
    RETURN collect(r.id) AS keep_release_ids
    """

    RETENTION_CANDIDATE_RELEASE_IDS_BY_APP = """
    MATCH (r:GraphRelease {app_id: $app_id, status: 'inactive'})
    WHERE coalesce(r.deactivated_at, r.created_at) < datetime($before_ts)
      AND NOT r.id IN $keep_release_ids
    RETURN collect(r.id) AS candidate_release_ids
    """

    RETENTION_COUNT_REVISIONS_FOR_RELEASES = """
    MATCH (rev:TransitionRevision)-[:IN_RELEASE]->(r:GraphRelease)
    WHERE r.id IN $release_ids
      AND coalesce(rev.is_active, false) = false
    RETURN count(DISTINCT rev) AS count
    """

    RETENTION_COUNT_SESSIONS_FOR_RELEASES = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(s:Session)
    WHERE s.current_release_id IN $release_ids
      AND coalesce(s.timestamp, datetime($before_ts)) < datetime($before_ts)
    RETURN count(DISTINCT s) AS count
    """

    RETENTION_COUNT_INGEST_RUNS_FOR_RELEASE_SESSIONS = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(s:Session)-[:GENERATES]->(run:IngestionRun)
    WHERE s.current_release_id IN $release_ids
      AND coalesce(s.timestamp, datetime($before_ts)) < datetime($before_ts)
    RETURN count(DISTINCT run) AS count
    """

    RETENTION_DELETE_RELEASES_BY_IDS = """
    MATCH (r:GraphRelease)
    WHERE r.id IN $release_ids
    WITH r LIMIT $batch_size
    DETACH DELETE r
    RETURN count(r) AS deleted_count
    """

    RETENTION_DELETE_SESSIONS_BY_RELEASE_IDS = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(s:Session)
    WHERE s.current_release_id IN $release_ids
      AND coalesce(s.timestamp, datetime($before_ts)) < datetime($before_ts)
    WITH s LIMIT $batch_size
    DETACH DELETE s
    RETURN count(s) AS deleted_count
    """

    RETENTION_DELETE_ORPHAN_INACTIVE_REVISIONS = """
    MATCH (rev:TransitionRevision)
    WHERE coalesce(rev.is_active, false) = false
      AND NOT (rev)-[:IN_RELEASE]->(:GraphRelease)
    WITH rev LIMIT $batch_size
    DETACH DELETE rev
    RETURN count(rev) AS deleted_count
    """

    RETENTION_DELETE_ORPHAN_COVERAGE_SNAPSHOTS = """
    MATCH (cov:CoverageSnapshot)
    WHERE NOT ()-[:HAS_COVERAGE]->(cov)
      AND coalesce(cov.captured_at, datetime($before_ts)) < datetime($before_ts)
    WITH cov LIMIT $batch_size
    DETACH DELETE cov
    RETURN count(cov) AS deleted_count
    """

    RETENTION_DELETE_ORPHAN_INGEST_RUNS = """
    MATCH (run:IngestionRun)
    WHERE NOT (:Session)-[:GENERATES]->(run)
      AND coalesce(run.created_at, datetime($before_ts)) < datetime($before_ts)
    WITH run LIMIT $batch_size
    DETACH DELETE run
    RETURN count(run) AS deleted_count
    """

    RETENTION_DELETE_ORPHAN_EVIDENCE = """
    MATCH (e:Evidence)
    WHERE NOT (:Session)-[:PROVIDED]->(e)
      AND NOT (:Transition)-[:SUPPORTED_BY]->(e)
      AND coalesce(e.created_at, datetime($before_ts)) < datetime($before_ts)
    WITH e LIMIT $batch_size
    DETACH DELETE e
    RETURN count(e) AS deleted_count
    """

    RETENTION_COUNT_ACTIVE_RELEASES_BY_APP = """
    MATCH (r:GraphRelease {app_id: $app_id, status: 'active'})
    RETURN count(r) AS active_count
    """

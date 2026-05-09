class SessionQueriesMixin:
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

class RetentionQueriesMixin:
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

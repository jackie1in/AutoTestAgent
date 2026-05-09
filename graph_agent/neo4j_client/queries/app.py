class AppQueriesMixin:
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

class RunnerQueriesMixin:
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

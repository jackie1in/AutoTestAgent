class SkipAdvisorQueriesMixin:
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

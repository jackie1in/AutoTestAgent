class SemanticQueriesMixin:
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

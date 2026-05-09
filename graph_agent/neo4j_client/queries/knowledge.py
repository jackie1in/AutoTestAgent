class KnowledgeQueriesMixin:
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

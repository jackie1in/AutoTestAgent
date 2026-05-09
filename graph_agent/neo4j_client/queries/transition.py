class TransitionQueriesMixin:
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
           t.confidence AS confidence,
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

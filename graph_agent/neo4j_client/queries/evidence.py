class EvidenceQueriesMixin:
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

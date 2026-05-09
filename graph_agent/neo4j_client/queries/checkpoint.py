class CheckpointQueriesMixin:
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

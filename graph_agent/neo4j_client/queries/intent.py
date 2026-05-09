class IntentQueriesMixin:
    UPSERT_INTENT = """
    MERGE (i:Intent {id: $id})
    SET i += $props
    SET i.name = coalesce(i.name, i.summary, i.key, i.id)
    RETURN i
    """

    LINK_TRANSITION_INTENT = """
    MATCH (t:Transition {id: $transition_id}), (i:Intent {id: $intent_id})
    MERGE (t)-[:REALIZES]->(i)
    """

    LINK_INTENT_REQUIRES_ENTITY = """
    MATCH (i:Intent {id: $intent_id}), (e:Entity {id: $entity_id})
    MERGE (i)-[:REQUIRES_ENTITY {field_mapping: $field_mapping}]->(e)
    """

    LINK_INTENT_PRODUCES_ENTITY = """
    MATCH (i:Intent {id: $intent_id}), (e:Entity {id: $entity_id})
    MERGE (i)-[:PRODUCES_ENTITY {field_mapping: $field_mapping}]->(e)
    """

    LINK_INTENT_PRECONDITION = """
    MATCH (i:Intent {id: $intent_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (i)-[:HAS_PRECONDITION]->(c)
    """

    LINK_INTENT_POSTCONDITION = """
    MATCH (i:Intent {id: $intent_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (i)-[:HAS_POSTCONDITION]->(c)
    """

    LINK_INTENT_DATA_FLOW = """
    MATCH (a:Intent {id: $from_intent_id}), (b:Intent {id: $to_intent_id})
    MERGE (a)-[:DATA_FLOW {from_slot: $from_slot, to_slot: $to_slot}]->(b)
    """

    LINK_INTENT_ALTERNATIVE = """
    MATCH (a:Intent {id: $intent_id_a}), (b:Intent {id: $intent_id_b})
    MERGE (a)-[:ALTERNATIVE_OF]->(b)
    """

    LINK_INTENT_MUST_PRECEDE = """
    MATCH (a:Intent {id: $before_intent_id}), (b:Intent {id: $after_intent_id})
    MERGE (a)-[:MUST_PRECEDE]->(b)
    """

class EntityQueriesMixin:
    UPSERT_ENTITY = """
    MERGE (e:Entity {id: $id})
    SET e += $props
    SET e.name = coalesce(e.name, e.description, e.id)
    RETURN e
    """

    UPSERT_ENTITY_INSTANCE = """
    MERGE (ei:EntityInstance {id: $id})
    SET ei += $props
    SET ei.name = coalesce(ei.status, ei.id)
    WITH ei
    MATCH (e:Entity {id: $entity_id})
    MERGE (e)-[:HAS_INSTANCE]->(ei)
    RETURN ei
    """

    LINK_ENTITY_INSTANCE_INTENT = """
    MATCH (ei:EntityInstance {id: $instance_id}), (i:Intent {id: $intent_id})
    MERGE (ei)-[:CREATED_BY_INTENT]->(i)
    """

class FieldConstraintQueriesMixin:
    UPSERT_FIELD_CONSTRAINT = """
    MERGE (fc:FieldConstraint {id: $id})
    SET fc += $props
    SET fc.name = coalesce(fc.field_name, fc.selector, fc.input_type, fc.id)
    RETURN fc
    """

    LINK_ZONE_FIELD = """
    MATCH (z:Zone {id: $zone_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (z)-[:HAS_FIELD]->(fc)
    """

    LINK_TRANSITION_OPERATES_ON = """
    MATCH (t:Transition {id: $transition_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (t)-[:OPERATES_ON]->(fc)
    """

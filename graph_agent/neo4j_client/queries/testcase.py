class TestCaseQueriesMixin:
    UPSERT_TEST_CASE = """
    MERGE (tc:TestCase {id: $id})
    SET tc += $props
    SET tc.name = coalesce(tc.name, tc.description, tc.category, tc.id)
    RETURN tc
    """

    LINK_TEST_CASE_TRANSITION = """
    MATCH (tc:TestCase {id: $test_case_id}), (t:Transition {id: $transition_id})
    MERGE (tc)-[:DERIVED_FROM]->(t)
    """

    LINK_TEST_CASE_CHECKPOINT = """
    MATCH (tc:TestCase {id: $test_case_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (tc)-[:EXPECTS]->(c)
    """

    LINK_TEST_CASE_COVERS = """
    MATCH (tc:TestCase {id: $test_case_id}), (fc:FieldConstraint {id: $field_id})
    MERGE (tc)-[:COVERS]->(fc)
    """

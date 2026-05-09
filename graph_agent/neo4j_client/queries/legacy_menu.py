class LegacyMenuQueriesMixin:
    LINK_MENU_PARENT_LEGACY = """
    MATCH (child:State {id: $child_id}), (parent:State {id: $parent_id})
    MERGE (child)-[:MENU_PARENT]->(parent)
    """

    LINK_MENU_NEXT_LEGACY = """
    MATCH (s:State {id: $state_id}), (next:State {id: $next_id})
    MERGE (s)-[:MENU_NEXT]->(next)
    """

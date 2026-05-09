class FrameQueriesMixin:
    UPSERT_FRAME = """
    MERGE (f:Frame {id: $id})
    SET f += $props
    SET f.name = coalesce(f.name, f.src, f.selector, f.id)
    RETURN f
    """

    LINK_STATE_FRAME = """
    MATCH (s:State {id: $state_id}), (f:Frame {id: $frame_id})
    MERGE (s)-[:HAS_FRAME {depth: $depth}]->(f)
    """

    LINK_FRAME_PARENT = """
    MATCH (parent:Frame {id: $parent_id}), (child:Frame {id: $child_id})
    MERGE (parent)-[:CONTAINS_FRAME]->(child)
    """

    LINK_TRANSITION_FRAME = """
    MATCH (t:Transition {id: $transition_id}), (f:Frame {id: $frame_id})
    MERGE (t)-[:IN_FRAME]->(f)
    """

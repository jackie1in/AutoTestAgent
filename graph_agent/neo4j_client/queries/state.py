class StateQueriesMixin:
    UPSERT_STATE = """
    MERGE (s:State {id: $id})
    SET s += $props
    SET s.name = coalesce(s.title, s.url, s.id)
    RETURN s
    """

    GET_STATE_BY_ID = "MATCH (s:State {id: $id}) RETURN s"
    GET_STATE_BY_URL = "MATCH (s:State {url: $url}) RETURN s"
    GET_ALL_STATES = "MATCH (s:State) RETURN s ORDER BY s.last_visited DESC"

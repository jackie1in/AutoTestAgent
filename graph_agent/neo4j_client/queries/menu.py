class MenuQueriesMixin:
    UPSERT_MENU = """
    MERGE (m:Menu {id: $id})
    SET m += $props
    RETURN m
    """

    LINK_APP_MENU = """
    MATCH (a:App {id: $app_id}), (m:Menu {id: $menu_id})
    MERGE (a)-[:HAS_MENU]->(m)
    """

    LINK_MENU_CHILD_OF = """
    MATCH (child:Menu {id: $child_id}), (parent:Menu {id: $parent_id})
    MERGE (child)-[:CHILD_OF {order_index: $order_index}]->(parent)
    """

    LINK_MENU_LEADS_TO = """
    MATCH (m:Menu {id: $menu_id}), (s:State {id: $state_id})
    MERGE (m)-[:LEADS_TO {first_seen: datetime($first_seen), session_id: $session_id}]->(s)
    """

    LINK_TRANSITION_NAVIGATED_VIA = """
    MATCH (t:Transition {id: $transition_id}), (m:Menu {id: $menu_id})
    MERGE (t)-[:NAVIGATED_VIA]->(m)
    """

    LINK_SESSION_DISCOVERED_MENU = """
    MATCH (sess:Session {id: $session_id}), (m:Menu {id: $menu_id})
    MERGE (sess)-[:DISCOVERED]->(m)
    """

    GET_MENU_BY_ID = "MATCH (m:Menu {id: $id}) RETURN m"

    GET_APP_MENUS = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
    RETURN m ORDER BY m.level, m.order_index
    """

    GET_MENU_TREE = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
    OPTIONAL MATCH (m)-[:CHILD_OF]->(parent:Menu)
    OPTIONAL MATCH (m)-[:LEADS_TO]->(s:State)
    RETURN m, parent.id AS parent_id, s.id AS state_id
    ORDER BY m.level, m.order_index
    """

    GET_MENU_LEADS_TO_STATE = """
    MATCH (m:Menu {id: $menu_id})-[:LEADS_TO]->(s:State)
    RETURN s
    """

    MARK_PAGE_MENUS_INACTIVE = """
    MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu {page_url: $page_url})
    SET m.is_active = false,
        m.inactive_at = datetime($inactive_at),
        m.updated_at = datetime($inactive_at)
    """

class CypherQueries:
    """Centralized Cypher query templates."""

    # ===== App =====
    UPSERT_APP = """
    MERGE (a:App {id: $id})
    SET a += $props
    RETURN a
    """

    GET_APP_BY_ID = "MATCH (a:App {id: $id}) RETURN a"
    GET_ALL_APPS = "MATCH (a:App) RETURN a ORDER BY a.last_session_at DESC"

    LINK_APP_STATE = """
    MATCH (a:App {id: $app_id}), (s:State {id: $state_id})
    MERGE (a)-[:HAS_STATE]->(s)
    """

    LINK_APP_SESSION = """
    MATCH (a:App {id: $app_id}), (sess:Session {id: $session_id})
    MERGE (a)-[:HAS_SESSION]->(sess)
    """

    GET_APP_STATES = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
    RETURN s ORDER BY s.last_visited DESC
    """

    GET_APP_SESSIONS = """
    MATCH (a:App {id: $app_id})-[:HAS_SESSION]->(sess:Session)
    RETURN sess ORDER BY sess.timestamp DESC
    """

    UPDATE_APP_STATS = """
    MATCH (a:App {id: $app_id})
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
    WITH a, count(s) AS sc
    OPTIONAL MATCH (a)-[:HAS_STATE]->(s2:State)<-[:FROM]-(t:Transition)
    WITH a, sc, count(DISTINCT t) AS tc
    OPTIONAL MATCH (a)-[:HAS_SESSION]->(sess:Session)
    WITH a, sc, tc, count(sess) AS sessc
    SET a.total_states = sc, a.total_transitions = tc, a.total_sessions = sessc
    """

    # ===== State =====
    UPSERT_STATE = """
    MERGE (s:State {id: $id})
    SET s += $props
    RETURN s
    """

    GET_STATE_BY_ID = "MATCH (s:State {id: $id}) RETURN s"
    GET_STATE_BY_URL = "MATCH (s:State {url: $url}) RETURN s"
    GET_ALL_STATES = "MATCH (s:State) RETURN s ORDER BY s.last_visited DESC"

    # ===== Transition =====
    UPSERT_TRANSITION = """
    MERGE (t:Transition {id: $id})
    SET t += $props
    WITH t
    MATCH (from_state:State {id: $from_state_id})
    MERGE (t)-[:FROM]->(from_state)
    WITH t
    MATCH (to_state:State {id: $to_state_id})
    MERGE (t)-[:TO]->(to_state)
    RETURN t
    """

    GET_TRANSITIONS_FROM_STATE = """
    MATCH (s:State {id: $state_id})<-[:FROM]-(t:Transition)-[:TO]->(target:State)
    RETURN t, target
    """

    GET_LOW_CONFIDENCE_TRANSITIONS = """
    MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    WHERE t.confidence < $threshold
    RETURN s1, t, s2 ORDER BY t.confidence ASC
    """

    # ===== Zone =====
    UPSERT_ZONE = """
    MERGE (z:Zone {id: $id})
    SET z += $props
    RETURN z
    """

    LINK_STATE_ZONE = """
    MATCH (s:State {id: $state_id}), (z:Zone {id: $zone_id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    LINK_TRANSITION_ZONE = """
    MATCH (t:Transition {id: $transition_id}), (z:Zone {id: $zone_id})
    MERGE (t)-[:IN_ZONE]->(z)
    """

    GET_UNEXPLORED_ZONES = """
    MATCH (s:State)-[:HAS_ZONE]->(z:Zone)
    WHERE z.exploration_status IN ['undiscovered', 'discovered']
    RETURN s, z ORDER BY z.exploration_status
    """

    # ===== Frame =====
    UPSERT_FRAME = """
    MERGE (f:Frame {id: $id})
    SET f += $props
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

    # ===== Intent =====
    UPSERT_INTENT = """
    MERGE (i:Intent {id: $id})
    SET i += $props
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

    # ===== Entity =====
    UPSERT_ENTITY = """
    MERGE (e:Entity {id: $id})
    SET e += $props
    RETURN e
    """

    UPSERT_ENTITY_INSTANCE = """
    MERGE (ei:EntityInstance {id: $id})
    SET ei += $props
    WITH ei
    MATCH (e:Entity {id: $entity_id})
    MERGE (e)-[:HAS_INSTANCE]->(ei)
    RETURN ei
    """

    LINK_ENTITY_INSTANCE_INTENT = """
    MATCH (ei:EntityInstance {id: $instance_id}), (i:Intent {id: $intent_id})
    MERGE (ei)-[:CREATED_BY_INTENT]->(i)
    """

    # ===== Checkpoint =====
    UPSERT_CHECKPOINT = """
    MERGE (c:Checkpoint {id: $id})
    SET c += $props
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

    # ===== FieldConstraint =====
    UPSERT_FIELD_CONSTRAINT = """
    MERGE (fc:FieldConstraint {id: $id})
    SET fc += $props
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

    # ===== TestCase =====
    UPSERT_TEST_CASE = """
    MERGE (tc:TestCase {id: $id})
    SET tc += $props
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

    # ===== Session =====
    UPSERT_SESSION = """
    MERGE (s:Session {id: $id})
    SET s += $props
    RETURN s
    """

    LINK_SESSION_DISCOVERED_STATE = """
    MATCH (sess:Session {id: $session_id}), (s:State {id: $state_id})
    MERGE (sess)-[:DISCOVERED]->(s)
    """

    LINK_SESSION_DISCOVERED_TRANSITION = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:DISCOVERED]->(t)
    """

    LINK_SESSION_GENERATED_CHECKPOINT = """
    MATCH (sess:Session {id: $session_id}), (c:Checkpoint {id: $checkpoint_id})
    MERGE (sess)-[:GENERATED]->(c)
    """

    LINK_SESSION_VALIDATED = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:VALIDATED]->(t)
    """

    LINK_SESSION_INVALIDATED = """
    MATCH (sess:Session {id: $session_id}), (t:Transition {id: $transition_id})
    MERGE (sess)-[:INVALIDATED]->(t)
    """

    LINK_SESSION_CREATED_ENTITY_INSTANCE = """
    MATCH (sess:Session {id: $session_id}), (ei:EntityInstance {id: $instance_id})
    MERGE (sess)-[:CREATED]->(ei)
    """

    # ===== Menu Navigation (New Schema: Menu as separate node) =====

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

    # ===== Legacy Menu Navigation (Deprecated: State-based menu links) =====

    LINK_MENU_PARENT_LEGACY = """
    MATCH (child:State {id: $child_id}), (parent:State {id: $parent_id})
    MERGE (child)-[:MENU_PARENT]->(parent)
    """

    LINK_MENU_NEXT_LEGACY = """
    MATCH (s:State {id: $state_id}), (next:State {id: $next_id})
    MERGE (s)-[:MENU_NEXT]->(next)
    """

    # ===== Graph Queries =====
    SHORTEST_PATH = """
    MATCH path = shortestPath(
        (s1:State {id: $from_id})-[:FROM|TO*]-(s2:State {id: $to_id})
    )
    RETURN path
    """

    COVERAGE_STATS = """
    OPTIONAL MATCH (s:State) WITH count(s) AS total_states
    OPTIONAL MATCH (z:Zone) WITH total_states, count(z) AS total_zones
    OPTIONAL MATCH (z2:Zone) WHERE z2.exploration_status IN ['explored', 'validated']
    WITH total_states, total_zones, count(z2) AS explored_zones
    OPTIONAL MATCH (t:Transition) WITH total_states, total_zones, explored_zones, count(t) AS total_transitions
    OPTIONAL MATCH (t2:Transition) WHERE t2.confidence >= 0.8
    WITH total_states, total_zones, explored_zones, total_transitions, count(t2) AS high_conf
    OPTIONAL MATCH (t3:Transition) WHERE t3.confidence >= 0.4 AND t3.confidence < 0.8
    WITH total_states, total_zones, explored_zones, total_transitions, high_conf, count(t3) AS med_conf
    OPTIONAL MATCH (t4:Transition) WHERE t4.confidence < 0.4
    RETURN total_states, total_zones, explored_zones, total_transitions, high_conf, med_conf, count(t4) AS low_conf
    """

    COVERAGE_STATS_BY_APP = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s:State)
    WITH count(s) AS total_states, collect(s) AS states
    UNWIND states AS st
    OPTIONAL MATCH (st)-[:HAS_ZONE]->(z:Zone)
    WITH total_states, states, collect(DISTINCT z) AS all_zones
    WITH total_states, states, size(all_zones) AS total_zones,
         size([z IN all_zones WHERE z.exploration_status IN ['explored', 'validated']]) AS explored_zones
    UNWIND states AS st2
    OPTIONAL MATCH (st2)<-[:FROM]-(t:Transition)
    WITH total_states, total_zones, explored_zones, collect(DISTINCT t) AS all_trans
    WITH total_states, total_zones, explored_zones,
         size(all_trans) AS total_transitions,
         size([t IN all_trans WHERE t.confidence >= 0.8]) AS high_conf,
         size([t IN all_trans WHERE t.confidence >= 0.4 AND t.confidence < 0.8]) AS med_conf,
         size([t IN all_trans WHERE t.confidence < 0.4]) AS low_conf
    RETURN total_states, total_zones, explored_zones, total_transitions, high_conf, med_conf, low_conf
    """

    FULL_GRAPH_EXPORT = """
    MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    OPTIONAL MATCH (t)-[:IN_ZONE]->(z:Zone)
    RETURN s1, t, s2, i, z
    """

    FULL_GRAPH_EXPORT_BY_APP = """
    MATCH (a:App {id: $app_id})-[:HAS_STATE]->(s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
    OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
    OPTIONAL MATCH (t)-[:IN_ZONE]->(z:Zone)
    RETURN s1, t, s2, i, z
    """

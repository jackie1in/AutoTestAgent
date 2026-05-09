class GraphQueriesMixin:
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

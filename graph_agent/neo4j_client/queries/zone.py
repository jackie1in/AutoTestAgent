class ZoneQueriesMixin:
    UPSERT_ZONE = """
    MERGE (z:Zone {id: $id})
    SET z += $props
    SET z.name = coalesce(z.summary, z.zone_type, z.id)
    RETURN z
    """

    LINK_STATE_ZONE = """
    MATCH (s:State {id: $state_id}), (z:Zone {id: $zone_id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    UPSERT_ZONES_FOR_APP = """
    UNWIND $zones as zone
    MERGE (z:Zone {id: zone.id})
    SET z.type = zone.type,
        z.selector = zone.selector,
        z.element_count = zone.element_count,
        z.bounds = zone.bounds,
        z.text_sample = zone.text_sample,
        z.ingest_version_id = zone.ingest_version_id,
        z.name = coalesce(zone.summary, zone.type, zone.id),
        z.updated_at = datetime()
    WITH z, zone,
         coalesce(z.exploration_status, 'undiscovered') AS prev_status,
         coalesce(zone.exploration_status, 'discovered') AS new_status
    WITH z, zone, prev_status, new_status,
         CASE prev_status
             WHEN 'validated' THEN 4
             WHEN 'explored' THEN 3
             WHEN 'partial' THEN 2
             WHEN 'discovered' THEN 1
             ELSE 0
         END AS prev_p,
         CASE new_status
             WHEN 'validated' THEN 4
             WHEN 'explored' THEN 3
             WHEN 'partial' THEN 2
             WHEN 'discovered' THEN 1
             ELSE 0
         END AS new_p
    SET z.exploration_status = CASE WHEN new_p >= prev_p THEN new_status ELSE prev_status END,
        z.last_explored = CASE
            WHEN zone.last_explored IS NOT NULL
              THEN datetime(zone.last_explored)
            ELSE z.last_explored
        END
    WITH z
    MATCH (a:App {id: $app_id})
    MERGE (a)-[:HAS_ZONE]->(z)
    """

    LINK_STATE_ZONES_DIRECT = """
    UNWIND $zones as zone
    MATCH (s:State {id: $state_id}), (z:Zone {id: zone.id})
    MERGE (s)-[:HAS_ZONE]->(z)
    """

    LINK_STATE_ZONES_BY_IDS = """
    UNWIND $zones as zone
    UNWIND coalesce(zone.state_ids, []) as sid
    MATCH (s:State {id: sid}), (z:Zone {id: zone.id})
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

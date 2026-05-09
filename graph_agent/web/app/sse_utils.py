import asyncio
import json
from typing import Any

from graph_agent.models import GraphEdge
from graph_agent.playback.engine import run_playback
from graph_agent.web.app.config import _resolve_app_name


def _log_entry_to_sse(entry: dict) -> dict:
    step_index = entry.get("step_index", 0)
    selector = entry.get("selector", "")
    action = entry.get("action", "")
    success = entry.get("success", False)
    error = entry.get("error")
    intent_summary = entry.get("intent", "")

    message = f"Step {step_index}: {action} {selector}"
    if intent_summary:
        message += f" ({intent_summary})"
    if error:
        message += f" — {error}"
    level = "error" if not success else "info"
    out = {
        "step_index": step_index,
        "message": message,
        "level": level,
        "selector": selector,
        "action": action,
        "success": success,
        "intent": intent_summary,
    }
    if error is not None:
        out["error"] = error
    return out


async def _apply_playback_feedback(
    edge_list: list[GraphEdge],
    playback_result: dict[str, Any],
) -> None:
    from graph_agent.web.app.graphrag_setup import _get_driver
    try:
        driver = await _get_driver()
    except Exception:
        return
    success = bool(playback_result.get("success"))
    failed_step_index = playback_result.get("failed_step_index")
    failed_edge_id = playback_result.get("failed_edge_id")

    transition_ids = [edge.edge_id for edge in edge_list if edge.edge_id]
    if not transition_ids:
        return
    async with driver.session() as session:
        for idx, edge in enumerate(edge_list):
            if not edge.edge_id:
                continue
            passed = success or (
                failed_step_index is not None and idx < int(failed_step_index)
            )
            if failed_edge_id and edge.edge_id == failed_edge_id:
                passed = False
            delta = 0.1 if passed else -0.2
            await session.run(
                """
                MATCH (t:Transition {id: $transition_id})
                SET t.confidence = CASE
                    WHEN coalesce(t.confidence, 0.5) + $delta > 1.0 THEN 1.0
                    WHEN coalesce(t.confidence, 0.5) + $delta < 0.0 THEN 0.0
                    ELSE coalesce(t.confidence, 0.5) + $delta
                END,
                t.last_validated = datetime(),
                t.validation_count = coalesce(t.validation_count, 0) + 1
                """,
                transition_id=edge.edge_id,
                delta=delta,
            )

        app_name = _resolve_app_name()
        stats_result = await session.run(
            """
            MATCH (a:App)
            WHERE $app_name = '' OR a.name = $app_name
            WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
            OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
            WITH a, count(s) AS total_states
            OPTIONAL MATCH (a)-[:HAS_STATE]->(:State)<-[:FROM]-(t:Transition)
            WITH a, total_states, count(t) AS total_transitions,
                 count(CASE WHEN coalesce(t.validation_count, 0) > 0 THEN 1 END) AS validated_transitions
            SET a.last_playback_success = $success,
                a.last_playback_at = datetime(),
                a.interaction_coverage = CASE
                    WHEN total_transitions = 0 THEN 0.0
                    ELSE toFloat(validated_transitions) / toFloat(total_transitions)
                END
            RETURN a.id AS app_id
            """,
            app_name=app_name,
            success=success,
        )
        await stats_result.single()


async def _sse_generator(
    edge_list: list[GraphEdge],
    test_data: dict,
    start_url: str,
    expected_end_url: str | None,
    wait_for_network: bool,
):
    q: asyncio.Queue = asyncio.Queue()

    async def log_callback(entry: dict) -> None:
        await q.put(_log_entry_to_sse(entry))

    async def worker():
        try:
            result = await run_playback(
                edge_list,
                test_data,
                start_url,
                expected_end_url,
                log_callback=log_callback,
                wait_for_network=wait_for_network,
            )
            await _apply_playback_feedback(edge_list, result)
            if result["success"]:
                await q.put({"level": "success", "actual_url": result["actual_url"]})
            else:
                await q.put(
                    {
                        "level": "error",
                        "error": result.get("error") or "Playback failed",
                        "actual_url": result.get("actual_url", ""),
                    }
                )
        except Exception as e:
            await q.put({"level": "error", "error": str(e), "actual_url": ""})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    while True:
        item = await q.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"
        if item.get("level") in ("success", "error"):
            break
    await task


_graph_stream_subscribers: list[asyncio.Queue] = []


def notify_intent_update(edge_id: str, intent_dict: dict | None, status: str) -> None:
    event = {"edge_id": edge_id, "intent": intent_dict, "status": status}
    for q in list(_graph_stream_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

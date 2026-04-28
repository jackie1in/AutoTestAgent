from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from graph_agent.cartography.config import is_http_url
from graph_agent.cartography.types import LayoutEvidenceItem, LayoutMetrics

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult


async def persist_mapping_result(
    *,
    app_id: str,
    app_name: str,
    session_id: str,
    resolved_url: str,
    current_url: str,
    inventory: list[dict],
    initial_actions_log: list[dict[str, object]],
    result: "CartographyResult",
) -> None:
    from graph_agent.models import (
        ActionType,
        App,
        Evidence,
        EvidenceType,
        Session,
        State,
        Transition,
    )
    from graph_agent.neo4j_client.manager import GraphManager

    async with GraphManager() as manager:
        app = App(id=app_id, name=app_name, entry_url=resolved_url)
        await manager.add_app(app)
        session = Session(id=session_id, app_id=app_id)
        await manager.add_session(session)
        if inventory:
            await manager.set_session_inventory(session_id, {"elements": inventory})

        stats = {
            "states_added": 0,
            "transitions_added": 0,
            "filtered_non_ui_edges": 0,
            "semantic_mismatch_warnings": 0,
            "url_discontinuity_warnings": 0,
            "frame_context_transition_warnings": 0,
        }

        login_url = resolved_url
        post_login_url = current_url or resolved_url
        if login_url != post_login_url and initial_actions_log:
            login_fp = hashlib.md5(login_url.encode()).hexdigest()[:12]
            post_fp = hashlib.md5(post_login_url.encode()).hexdigest()[:12]
            from_state = State(id=f"state:prelogin:{login_fp}", url=login_url, title="Login page")
            to_state = State(id=f"state:prelogin:{post_fp}", url=post_login_url, title="Post-login page")
            prelogin_transition = Transition(
                id=f"{session_id}:prelogin",
                selector="[pre-login]",
                action=ActionType.CLICK,
                from_state_id=from_state.id,
                to_state_id=to_state.id,
                thought="Auto-login: filled credentials and submitted",
                confidence=0.9,
            )
            await manager.add_state(from_state)
            await manager.link_app_state(app_id, from_state.id)
            await manager.link_session_discovered(session_id, from_state.id)
            stats["states_added"] += 1
            await manager.add_state(to_state)
            await manager.link_app_state(app_id, to_state.id)
            await manager.link_session_discovered(session_id, to_state.id)
            stats["states_added"] += 1
            await manager.add_transition(prelogin_transition)
            await manager.link_session_transition(session_id, prelogin_transition.id)
            stats["transitions_added"] += 1

        seen_state_ids: set[str] = set()
        for state in result.states:
            if state.id in seen_state_ids:
                continue
            await manager.add_state(state)
            await manager.link_app_state(app_id, state.id)
            await manager.link_session_discovered(session_id, state.id)
            seen_state_ids.add(state.id)
            stats["states_added"] += 1

        for transition in result.transitions:
            await manager.add_transition(transition)
            await manager.link_session_transition(session_id, transition.id)
            stats["transitions_added"] += 1
            for evidence_id in transition.evidence_ids:
                evidence_type = (
                    EvidenceType.URL_CHANGE if evidence_id.endswith("url_change") else EvidenceType.DOM_DIFF
                )
                evidence = Evidence(
                    id=evidence_id,
                    transition_id=transition.id,
                    session_id=session_id,
                    evidence_type=evidence_type,
                    summary=f"{evidence_type.value} evidence for {transition.selector}",
                    payload=json.dumps(
                        {
                            "from_state_id": transition.from_state_id,
                            "to_state_id": transition.to_state_id,
                            "selector": transition.selector,
                            "step_index": transition.step_index,
                        },
                        ensure_ascii=False,
                    ),
                    confidence=min(1.0, max(0.1, transition.confidence)),
                )
                await manager.add_evidence(evidence)
                await manager.link_transition_evidence(transition.id, evidence.id)
                await manager.link_session_evidence(session_id, evidence.id)

        # Persist layout evidence emitted by mapping pipeline (session-level).
        if getattr(result, "layout_evidence", None):
            for idx, ev in enumerate(result.layout_evidence):
                url = str(ev.get("url") or "")
                layout_fp = str(ev.get("layout_fingerprint") or "")
                if not layout_fp:
                    continue
                evidence_id = f"evidence:layout:{session_id}:{idx}:{layout_fp[:8]}"
                confidence = float(ev.get("layout_confidence") or 0.5)
                layout_evidence = Evidence(
                    id=evidence_id,
                    transition_id="",
                    session_id=session_id,
                    evidence_type=EvidenceType.LAYOUT,
                    summary=f"layout evidence at step {ev.get('step', idx)}",
                    payload=json.dumps(
                        {
                            "url": url,
                            "step": int(ev.get("step") or idx),
                            "layout_fingerprint": layout_fp,
                            "layout_summary": str(ev.get("layout_summary") or ""),
                        },
                        ensure_ascii=False,
                    ),
                    confidence=max(0.1, min(1.0, confidence)),
                )
                await manager.add_evidence(layout_evidence)
                await manager.link_session_evidence(session_id, evidence_id)

        menu_rows: list[dict[str, object]] = []
        if getattr(result, "menus", None):
            for i, menu in enumerate(result.menus):
                text = (menu.get("text") or "").strip()
                href = (menu.get("href") or "").strip()
                level = int(menu.get("level") or 0)
                source_url = (menu.get("source_url") or "").strip()
                if not text and not href:
                    continue
                menu_id_src = f"{app_id}|{source_url}|{level}|{text}|{href}"
                menu_rows.append(
                    {
                        "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
                        "text": text or href,
                        "href": href,
                        "level": level,
                        "order": i,
                        "is_active": False,
                    }
                )
            if menu_rows:
                await manager.add_menus(
                    app_id=app_id,
                    menus=menu_rows,
                    page_url=current_url or resolved_url,
                    session_id=session_id,
                )

        if getattr(result, "zone_hints", None):
            zone_rows: list[dict[str, object]] = []
            for zone in result.zone_hints:
                selector = (zone.get("selector") or "").strip()
                z_type = (zone.get("zone_type") or "content").strip()
                summary = (zone.get("description") or "").strip()
                if not selector:
                    continue
                zid_src = f"{app_id}|{z_type}|{selector}"
                zone_rows.append(
                    {
                        "id": f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}",
                        "type": z_type,
                        "selector": selector,
                        "element_count": 0,
                        "bounds": "",
                        "text_sample": summary,
                    }
                )
            if zone_rows:
                await manager.add_zones(app_id=app_id, zones=zone_rows)

        if menu_rows:
            for transition in result.transitions:
                signal = f"{transition.selector} {transition.thought or ''}".lower()
                for menu in menu_rows:
                    text = str(menu.get("text") or "").strip().lower()
                    if text and text in signal:
                        await manager.link_transition_navigated_via(transition.id, str(menu["id"]))
                        break

        visited_urls: list[str] = []
        if result.history:
            for h in result.history:
                url = h.get("url", "")
                if url and is_http_url(url):
                    visited_urls.append(url)
        visited_urls = list(dict.fromkeys(visited_urls))

        final_text = result.history[-1].get("result", "") if result.history else ""
        mapping_stopped = False
        stop_reason = None
        for marker in ("Stopped:", "停止："):
            if marker in final_text:
                idx = final_text.find(marker)
                stop_reason = final_text[idx + len(marker) :].strip()
                if len(stop_reason) > 200:
                    stop_reason = stop_reason[:200] + "..."
                mapping_stopped = True
                break

        await manager.update_session_stats(
            session_id=session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": resolved_url,
                **stats,
                **(result.layout_metrics if isinstance(result.layout_metrics, dict) else {}),
            },
        )
        print(
            f"Graph stored in Neo4j: app_id={app_id} "
            f"(states={stats.get('states_added', 0)}, transitions={stats.get('transitions_added', 0)})"
        )

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from graph_agent.cartography.config import clean_url, is_http_url
from graph_agent.lib.observability import observe
from graph_agent.cartography.persistence.semantic_stability import (
    SEMANTIC_STABILITY_THRESHOLD,
    _as_int,
    _as_str,
    _build_semantic_metrics,
    _calculate_semantic_stability,
)

if TYPE_CHECKING:
    from graph_agent.graph.merger import CartographyResult
    from graph_agent.neo4j_client.manager import GraphManager

logger = logging.getLogger(__name__)


class _FinalizeMixin:
    app_id: str
    session_id: str
    resolved_url: str
    current_url: str
    _initialized: bool
    _finalized: bool
    _manager: "GraphManager | None"
    _accumulated_result: "CartographyResult | None"
    _menu_rows: list[dict[str, object]]
    _ingest_version_id: str
    _stats: dict[str, int]
    _active_revision_ids: list[str]
    _state_ids_by_url: dict[str, set[str]]

    @observe(
        name="cartography.finalize_persistence_session",
        metadata={"component": "cartography", "stage": "persistence"},
    )
    async def finalize(self, final_result: "CartographyResult") -> None:
        """Phase 3: Write menus, zones, release, coverage snapshot, session stats."""
        if not self._initialized or self._manager is None:
            logger.warning("[PERSIST] finalize called before init, skipping")
            return
        if self._finalized:
            return

        from graph_agent.models import (
            CoverageSnapshot,
            GraphRelease,
        )
        from graph_agent.coverage.analyzer import CoverageAnalyzer

        manager = self._manager

        # Merge final_result into accumulated for complete metrics
        if self._accumulated_result is not None and final_result is not None:
            seen_sids = {s.id for s in self._accumulated_result.states}
            for state in final_result.states:
                if state.id not in seen_sids:
                    self._accumulated_result.states.append(state)
                    seen_sids.add(state.id)
            seen_tids = {t.id for t in self._accumulated_result.transitions}
            for transition in final_result.transitions:
                if transition.id not in seen_tids:
                    self._accumulated_result.transitions.append(transition)
                    seen_tids.add(transition.id)
            if final_result.history:
                self._accumulated_result.history.extend(final_result.history)
            if final_result.layout_evidence:
                self._accumulated_result.layout_evidence.extend(final_result.layout_evidence)
            if final_result.menus:
                self._accumulated_result.menus = final_result.menus
            if final_result.zone_hints:
                self._accumulated_result.zone_hints = final_result.zone_hints
            if final_result.layout_metrics:
                self._accumulated_result.layout_metrics = final_result.layout_metrics
            if final_result.intervention_tasks:
                self._accumulated_result.intervention_tasks = final_result.intervention_tasks
            self._accumulated_result.semantic_conflict_count = final_result.semantic_conflict_count

        result = self._accumulated_result
        assert result is not None

        # --- Menus ---
        for i, menu in enumerate(result.menus):
            if not isinstance(menu, dict):
                continue
            text = _as_str(menu.get("text")).strip()
            href = _as_str(menu.get("href")).strip()
            level = _as_int(menu.get("level") or 0, 0)
            source_url = _as_str(menu.get("source_url")).strip()
            if not text and not href:
                continue
            menu_id_src = f"{self.app_id}|{source_url}|{level}|{text}|{href}"
            self._menu_rows.append(
                {
                    "id": f"menu:{hashlib.md5(menu_id_src.encode()).hexdigest()[:12]}",
                    "text": text or href,
                    "href": href,
                    "level": level,
                    "order": i,
                    "is_active": True,
                    "ingest_version_id": self._ingest_version_id,
                }
            )
        if self._menu_rows:
            await manager.add_menus(
                app_id=self.app_id,
                menus=self._menu_rows,
                page_url=self.current_url or self.resolved_url,
                session_id=self.session_id,
            )
            for menu in self._menu_rows:
                menu_id = _as_str(menu.get("id"))
                if menu_id:
                    await manager.link_ingestion_emits_menu(
                        self._ingest_version_id, menu_id
                    )

        # --- Zones ---
        if getattr(result, "zone_hints", None):
            _status_priority = {
                "undiscovered": 0,
                "stale": 0,
                "discovered": 1,
                "partial": 2,
                "explored": 3,
                "validated": 4,
            }
            zone_state_map: dict[tuple[str, str, str], dict[str, object]] = {}
            for zone in result.zone_hints:
                if not isinstance(zone, dict):
                    continue
                selector = _as_str(zone.get("selector")).strip()
                z_type = _as_str(zone.get("zone_type") or "content").strip()
                source_url = _as_str(zone.get("source_url")).strip()
                source_url_key = clean_url(source_url) if source_url else ""
                if not selector:
                    continue
                key = (z_type, selector, source_url_key)
                status_raw = (
                    _as_str(zone.get("exploration_status") or "discovered")
                    .strip()
                    .lower()
                    or "discovered"
                )
                last_explored_iso = (
                    zone.get("last_explored") if zone.get("last_explored") else None
                )
                existing = zone_state_map.get(key)
                if existing is None or _status_priority.get(
                    status_raw, 0
                ) > _status_priority.get(str(existing.get("status") or ""), 0):
                    zone_state_map[key] = {
                        "status": status_raw,
                        "last_explored": last_explored_iso,
                    }
                elif last_explored_iso and not existing.get("last_explored"):
                    existing["last_explored"] = last_explored_iso

            zone_rows: list[dict[str, object]] = []
            zone_rows_by_id: dict[str, dict[str, object]] = {}
            for zone in result.zone_hints:
                if not isinstance(zone, dict):
                    continue
                selector = _as_str(zone.get("selector")).strip()
                z_type = _as_str(zone.get("zone_type") or "content").strip()
                summary = _as_str(zone.get("description")).strip()
                source_url = _as_str(zone.get("source_url")).strip()
                source_url_key = clean_url(source_url) if source_url else ""
                if not selector:
                    continue
                zid_src = f"{self.app_id}|{z_type}|{selector}|{source_url_key}"
                zone_id = f"zone:{hashlib.md5(zid_src.encode()).hexdigest()[:12]}"
                related_state_ids: set[str] = set()
                for key in (source_url, clean_url(source_url)):
                    if not key:
                        continue
                    related_state_ids.update(self._state_ids_by_url.get(key, set()))
                state_info = zone_state_map.get((z_type, selector, source_url_key), {})
                existing_row = zone_rows_by_id.get(zone_id)
                if existing_row is None:
                    row = {
                        "id": zone_id,
                        "type": z_type,
                        "selector": selector,
                        "element_count": 0,
                        "bounds": "",
                        "text_sample": summary,
                        "ingest_version_id": self._ingest_version_id,
                        "exploration_status": str(
                            state_info.get("status") or "discovered"
                        ),
                        "last_explored": state_info.get("last_explored"),
                        "state_ids": sorted(related_state_ids),
                    }
                    zone_rows.append(row)
                    zone_rows_by_id[zone_id] = row
                    continue

                raw_ids = existing_row.get("state_ids") or []
                merged_state_ids = set(raw_ids if isinstance(raw_ids, list) else [])
                merged_state_ids.update(related_state_ids)
                existing_row["state_ids"] = sorted(
                    _as_str(v).strip() for v in merged_state_ids if _as_str(v).strip()
                )
                if summary and not _as_str(existing_row.get("text_sample")).strip():
                    existing_row["text_sample"] = summary
                existing_status = (
                    _as_str(existing_row.get("exploration_status") or "discovered")
                    .strip()
                    .lower()
                )
                incoming_status = (
                    _as_str(state_info.get("status") or "discovered")
                    .strip()
                    .lower()
                )
                if _status_priority.get(incoming_status, 0) > _status_priority.get(
                    existing_status, 0
                ):
                    existing_row["exploration_status"] = incoming_status
                if (
                    state_info.get("last_explored")
                    and not existing_row.get("last_explored")
                ):
                    existing_row["last_explored"] = state_info.get("last_explored")
            if zone_rows:
                await manager.add_zones(app_id=self.app_id, zones=zone_rows)
                for zone in zone_rows:
                    zid = _as_str(zone.get("id"))
                    if zid:
                        await manager.link_ingestion_emits_zone(
                            self._ingest_version_id, zid
                        )

        # --- Menu-transition links ---
        if self._menu_rows:
            for transition in result.transitions:
                signal = f"{transition.selector} {transition.thought or ''}".lower()
                for menu in self._menu_rows:
                    text = str(menu.get("text") or "").strip().lower()
                    if text and text in signal:
                        await manager.link_transition_navigated_via(
                            transition.id, str(menu["id"])
                        )
                        break

        # --- Release ---
        visited_urls: list[str] = []
        if result.history:
            for h in result.history:
                if not isinstance(h, dict):
                    continue
                url = _as_str(h.get("url", ""))
                if url and is_http_url(url):
                    visited_urls.append(url)
        visited_urls = list(dict.fromkeys(visited_urls))

        last_history = result.history[-1] if result.history else {}
        final_text_str = _as_str(
            last_history.get("result", "") if isinstance(last_history, dict) else ""
        )
        mapping_stopped = False
        stop_reason = None
        for marker in ("Stopped:", "停止："):
            if marker in final_text_str:
                idx = final_text_str.find(marker)
                stop_reason = final_text_str[idx + len(marker) :].strip()
                if len(stop_reason) > 200:
                    stop_reason = stop_reason[:200] + "..."
                mapping_stopped = True
                break

        release_id = (
            f"release:{self.app_id}:{hashlib.md5(self._ingest_version_id.encode()).hexdigest()[:12]}"
        )
        await manager.add_graph_release(
            GraphRelease(
                id=release_id,
                app_id=self.app_id,
                base_ingest_ids=[self._ingest_version_id],
                status="active",
            )
        )
        await manager.deactivate_other_active_releases(
            app_id=self.app_id,
            keep_release_id=release_id,
        )
        for revision_id in self._active_revision_ids:
            await manager.link_release_revision(release_id, revision_id)

        # --- Coverage snapshot ---
        coverage_snapshot_id: str | None = None
        try:
            analyzer = CoverageAnalyzer(manager.get_driver())
            report = await analyzer.compute(app_id=self.app_id)
            coverage_snapshot_id = (
                f"cov:{self.session_id}:"
                f"{hashlib.md5(release_id.encode()).hexdigest()[:8]}"
            )
            snapshot = CoverageSnapshot(
                id=coverage_snapshot_id,
                app_id=self.app_id,
                session_id=self.session_id,
                release_id=release_id,
                menu_coverage=report.menu_coverage,
                zone_coverage=report.zone_coverage,
                interaction_coverage=report.interaction_coverage,
                state_coverage=report.state_coverage,
                overall_completeness=report.overall_completeness,
                transition_high=report.transition_confidence.high,
                transition_medium=report.transition_confidence.medium,
                transition_low=report.transition_confidence.low,
                recommendation=report.recommendation,
            )
            await manager.add_coverage_snapshot(snapshot)
            await manager.link_session_coverage(self.session_id, coverage_snapshot_id)
            await manager.link_release_coverage(release_id, coverage_snapshot_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[PERSIST] coverage snapshot skipped: %s", e)

        # --- Semantic metrics ---
        self._stats["semantic_mismatch_warnings"] = int(
            getattr(result, "semantic_conflict_count", 0) or 0
        )
        semantic_metrics = _build_semantic_metrics(result)
        baseline_row = await manager.get_latest_semantic_baseline(
            app_id=self.app_id,
            exclude_session_id=self.session_id,
        )
        semantic_stability_metrics = _calculate_semantic_stability(
            semantic_metrics=semantic_metrics,
            baseline_row=baseline_row,
            threshold=SEMANTIC_STABILITY_THRESHOLD,
        )

        layout_metrics_payload = (
            dict(result.layout_metrics)
            if isinstance(result.layout_metrics, dict)
            else {}
        )
        for key, default in {
            "captcha_action_count": 0,
            "captcha_ok_count": 0,
            "captcha_empty_count": 0,
            "captcha_manual_empty_count": 0,
            "captcha_fill_failed_count": 0,
            "captcha_manual_wait_ms_total": 0,
        }.items():
            layout_metrics_payload.setdefault(key, default)

        # --- Update ingestion run status ---
        await manager.run_write_transaction([
            (
                "MATCH (ir:IngestionRun {id: $id}) SET ir.status = $status",
                {"id": self._ingest_version_id, "status": "completed"},
            )
        ])

        await manager.update_session_stats(
            session_id=self.session_id,
            stats={
                "visited_urls": visited_urls,
                "mapping_stopped": mapping_stopped,
                "stop_reason": stop_reason,
                "start_url": self.resolved_url,
                "current_release_id": release_id,
                "latest_ingest_version_id": self._ingest_version_id,
                "current_coverage_snapshot_id": coverage_snapshot_id or "",
                "intervention_task_count": len(
                    getattr(result, "intervention_tasks", []) or []
                ),
                "intervention_tasks": list(
                    getattr(result, "intervention_tasks", []) or []
                ),
                **self._stats,
                **layout_metrics_payload,
                **semantic_metrics,
                **semantic_stability_metrics,
            },
        )
        await manager.touch_app_last_session(self.app_id)
        await manager.update_app_stats(self.app_id)

        self._finalized = True
        logger.info(
            "[PERSIST] Session finalized: app_id=%s ingest=%s "
            "(states=%d, transitions=%d)",
            self.app_id, self._ingest_version_id,
            self._stats["states_added"],
            self._stats["transitions_added"],
        )

    async def close(self) -> None:
        """Close the underlying GraphManager connection."""
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("[PERSIST] Error closing manager: %s", e)
            self._manager = None


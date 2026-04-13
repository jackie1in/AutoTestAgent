"""Cartography orchestrator — drives the entire exploration session.

All browser interactions go through browser-use's BrowserSession / Page APIs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neo4j import AsyncDriver

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession

from graph_agent.cartography.menu_extractor import MenuExtractor
from graph_agent.cartography.react_explorer import ReActExplorer
from graph_agent.cartography.snapshot import capture_dom_fingerprint
from graph_agent.cartography.state_id import generate_state_id, StateIdStrategy
from graph_agent.cartography.zone_discoverer import ZoneDiscoverer
from graph_agent.coverage.analyzer import CoverageAnalyzer
from graph_agent.coverage.scheduler import ExplorationScheduler, ScheduledTask
from graph_agent.graph.merger import CartographyResult, GraphMerger
from graph_agent.lib.page_controller import PageController
from graph_agent.lib.token_tracker import get_global_tracker
from graph_agent.models import App, Session, SessionFocus, State

logger = logging.getLogger(__name__)


class CartographyOrchestrator:
    """Orchestrates a complete cartography session."""

    def __init__(self, neo4j_driver: AsyncDriver, browser_session: "BrowserSession"):
        self._driver = neo4j_driver
        self._session = browser_session
        self._merger = GraphMerger(neo4j_driver)
        self._coverage = CoverageAnalyzer(neo4j_driver)
        self._scheduler = ExplorationScheduler()
        self._menu_extractor = MenuExtractor()
        self._zone_discoverer = ZoneDiscoverer()
        self._controller = PageController(browser_session)
        self._react_explorer = ReActExplorer(
            max_steps=25, browser_session=browser_session
        )

    async def _page(self):
        return await self._session.must_get_current_page()

    async def run_session(
        self,
        start_url: str,
        focus: str = "breadth",
        time_budget_ms: int = 21_600_000,
        app_id: str | None = None,
        app_name: str = "",
    ) -> dict[str, Any]:
        start_time = time.monotonic()
        session_id = f"session:{datetime.utcnow().isoformat()}"

        try:
            focus_enum = SessionFocus(focus)
        except ValueError:
            focus_enum = SessionFocus.BREADTH

        # Ensure the App node exists
        if app_id:
            async with self._driver.session() as db_session:
                r = await db_session.run("MATCH (a:App {id: $id}) RETURN a.id", id=app_id)
                existing = await r.single()
                now_iso = datetime.utcnow().isoformat()
                if existing is None:
                    app = App(
                        id=app_id,
                        name=app_name or app_id,
                        entry_url=start_url,
                    )
                    props = {k: v.isoformat() if isinstance(v, datetime) else v
                             for k, v in app.model_dump(exclude_none=True).items()}
                    await db_session.run(
                        "CREATE (a:App) SET a = $props", props=props,
                    )
                    logger.info("Created App node: %s", app_id)
                await db_session.run(
                    "MATCH (a:App {id: $id}) SET a.last_session_at = $now",
                    id=app_id, now=now_iso,
                )

        sess = Session(id=session_id, app_id=app_id, focus=focus_enum)

        async with self._driver.session() as db_session:
            await db_session.run(
                "MERGE (s:Session {id: $id}) SET s.timestamp = $ts, s.focus = $focus"
                + (", s.app_id = $app_id" if app_id else ""),
                id=session_id,
                ts=datetime.utcnow().isoformat(),
                focus=focus,
                **({"app_id": app_id} if app_id else {}),
            )
            if app_id:
                await db_session.run(
                    "MATCH (a:App {id: $app_id}), (sess:Session {id: $session_id}) "
                    "MERGE (a)-[:HAS_SESSION]->(sess)",
                    app_id=app_id, session_id=session_id,
                )

        coverage = await self._coverage.compute()

        # Extract menu structure
        logger.info("Extracting menu tree (incremental)...")
        page = await self._page()
        try:
            await page.goto(start_url)
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(2)

        menu_items = await self._menu_extractor.extract(page)
        logger.info("Menu extractor found %d items", len(menu_items))

        known_titles: set[str] = set()
        async with self._driver.session() as db_session:
            r = await db_session.run("MATCH (s:State) RETURN s.title AS title")
            records = await r.data()
            known_titles = {rec["title"] for rec in records if rec.get("title")}

        states_from_menu: list[State] = []
        has_spa_items = any(item.get("is_spa") for item in menu_items)

        if has_spa_items:
            for item in menu_items:
                text = item.get("text", "")
                if not text:
                    continue
                if text in known_titles:
                    logger.debug("  SPA menu [%s] already known, refreshing", text)
                try:
                    clicked = await self._click_link_by_text(page, text)
                    if clicked:
                        await asyncio.sleep(1.5)
                        url = await page.get_url()
                        fp = await capture_dom_fingerprint(page)
                        # Use URL + Fingerprint for unique state identification
                        state_id = generate_state_id(
                            url, fp, text, StateIdStrategy.URL_FINGERPRINT
                        )
                        state = State(
                            id=state_id, url=url, title=text,
                            fingerprint=fp, menu_path=[text],
                        )
                        states_from_menu.append(state)
                        logger.info("  SPA menu [%s] → %s (id: %s)", text, url, state_id[:30])
                except Exception as e:
                    logger.debug("SPA menu click failed for %s: %s", text, e)
        else:
            for item in menu_items:
                href = item.get("href", "")
                text = item.get("text", "")
                if href and text:
                    # Use URL-based ID for non-SPA static links
                    state_id = generate_state_id(
                        href, "", text, StateIdStrategy.URL_ONLY
                    )
                    state = State(
                        id=state_id,
                        url=href, title=text, menu_path=[text],
                    )
                    states_from_menu.append(state)

        if states_from_menu:
            result = CartographyResult(states=states_from_menu)
            merge_report = await self._merger.merge(result, session_id, app_id=app_id)
            sess.states_discovered = merge_report.states_created
            logger.info(
                "Menu extraction: %d new / %d updated states",
                merge_report.states_created, merge_report.states_updated,
            )

        tasks = await self._scheduler.schedule(self._driver, focus=focus)
        logger.info("Scheduled %d tasks (focus=%s)", len(tasks), focus)

        tasks_executed = 0
        last_save_time = time.monotonic()
        
        for task in tasks:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if elapsed_ms >= time_budget_ms:
                logger.info("Time budget exhausted (%.0fms / %dms)", elapsed_ms, time_budget_ms)
                break
            
            # 检查任务是否已完成（断点续传）
            if await self._is_task_completed(task.target_id, session_id):
                logger.debug("Task %s already completed, skipping", task.target_id)
                tasks_executed += 1
                continue
            
            try:
                await self._execute_task(task, session_id, app_id=app_id)
                tasks_executed += 1
                # 标记任务完成
                await self._mark_task_completed(task.target_id, session_id)
            except Exception as e:
                logger.warning("Task %s failed: %s", task.target_id, e)
                await self._mark_task_failed(task.target_id, session_id, str(e))
            
            # 每10分钟保存一次会话状态（断点续传）
            if time.monotonic() - last_save_time > 600:
                await self._save_session_checkpoint(session_id, tasks_executed, len(tasks))
                last_save_time = time.monotonic()

        final_coverage = await self._coverage.compute()
        elapsed_total = int((time.monotonic() - start_time) * 1000)

        async with self._driver.session() as db_session:
            await db_session.run(
                "MATCH (s:Session {id: $id}) "
                "SET s.duration_ms = $dur, "
                "s.states_discovered = $sd, s.transitions_discovered = $td, "
                "s.checkpoints_generated = $cg",
                id=session_id, dur=elapsed_total,
                sd=sess.states_discovered,
                td=sess.transitions_discovered,
                cg=sess.checkpoints_generated,
            )

        if app_id:
            async with self._driver.session() as db_session:
                await db_session.run(
                    "MATCH (a:App {id: $app_id}) "
                    "OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State) "
                    "WITH a, count(s) AS sc "
                    "OPTIONAL MATCH (a)-[:HAS_STATE]->(s2:State)<-[:FROM]-(t:Transition) "
                    "WITH a, sc, count(DISTINCT t) AS tc "
                    "OPTIONAL MATCH (a)-[:HAS_SESSION]->(sess:Session) "
                    "WITH a, sc, tc, count(sess) AS sessc "
                    "SET a.total_states = sc, a.total_transitions = tc, a.total_sessions = sessc",
                    app_id=app_id,
                )

        report = {
            "app_id": app_id,
            "session_id": session_id,
            "focus": focus,
            "duration_ms": elapsed_total,
            "tasks_scheduled": len(tasks),
            "tasks_executed": tasks_executed,
            "coverage_before": coverage.model_dump(),
            "coverage_after": final_coverage.model_dump(),
        }
        logger.info("Session %s complete: %d tasks in %dms", session_id, tasks_executed, elapsed_total)
        
        # Log final token usage summary for entire session
        token_tracker = get_global_tracker()
        token_tracker.log_summary()
        
        return report

    async def _execute_task(self, task: ScheduledTask, session_id: str, app_id: str | None = None) -> None:
        page = await self._page()

        match task.type:
            case "discover_page":
                url = task.context.get("url", "")
                title = task.context.get("title", "")
                if url:
                    await self._navigate_to(page, url, title)
                    fp = await capture_dom_fingerprint(page)
                    page_title = title or await page.get_title()
                    state = State(
                        id=task.target_id,
                        url=await page.get_url(),
                        title=page_title,
                        fingerprint=fp,
                    )
                    zones = await self._zone_discoverer.discover(
                        page, task.target_id, session=self._session
                    )
                    zone_state_map = {z.id: state.id for z in zones}
                    result = CartographyResult(
                        states=[state], zones=zones, zone_state_map=zone_state_map
                    )
                    await self._merger.merge(result, session_id, app_id=app_id)

            case "explore_zone":
                zone_id = task.target_id
                state_id = task.context.get("state_id", "")
                page_title = ""

                if state_id:
                    async with self._driver.session() as db_session:
                        r = await db_session.run(
                            "MATCH (s:State {id: $id}) RETURN s.url AS url, s.title AS title",
                            id=state_id,
                        )
                        rec = await r.single()
                        if rec:
                            page_title = rec.get("title", "")
                            await self._navigate_to(page, rec.get("url", ""), page_title)

                exploration_result = await self._react_explorer.explore_page(
                    self._session, state_id, page_title=page_title
                )
                await self._merger.merge(exploration_result, session_id, app_id=app_id)

                async with self._driver.session() as db_session:
                    await db_session.run(
                        "MATCH (z:Zone {id: $id}) "
                        "SET z.exploration_status = 'explored', z.last_explored = $now",
                        id=zone_id, now=datetime.utcnow().isoformat(),
                    )

            case "validate_transition":
                tid = task.target_id
                from_id = task.context.get("from_state_id", "")
                if from_id:
                    async with self._driver.session() as db_session:
                        r = await db_session.run(
                            "MATCH (s:State {id: $id}) RETURN s.url AS url",
                            id=from_id,
                        )
                        rec = await r.single()
                        if rec and rec["url"]:
                            try:
                                await page.goto(rec["url"])
                                await asyncio.sleep(2)
                            except Exception:
                                pass
                passed = True
                await self._merger.update_transition_confidence(tid, passed, session_id)

            case "probe_constraints":
                pass

    # ── Checkpoint / Resume Support ──────────────────────────────

    async def _is_task_completed(self, target_id: str, session_id: str) -> bool:
        """Check if a task has been completed (for resume support)."""
        try:
            async with self._driver.session() as db_session:
                r = await db_session.run(
                    "MATCH (t:TaskProgress {target_id: $target_id, session_id: $session_id}) "
                    "RETURN t.status AS status",
                    target_id=target_id, session_id=session_id,
                )
                rec = await r.single()
                return rec is not None and rec.get("status") == "completed"
        except Exception:
            return False

    async def _mark_task_completed(self, target_id: str, session_id: str) -> None:
        """Mark a task as completed."""
        try:
            async with self._driver.session() as db_session:
                await db_session.run(
                    "MERGE (t:TaskProgress {target_id: $target_id, session_id: $session_id}) "
                    "SET t.status = 'completed', t.completed_at = $now",
                    target_id=target_id, session_id=session_id, 
                    now=datetime.utcnow().isoformat(),
                )
        except Exception as e:
            logger.debug("Failed to mark task completed: %s", e)

    async def _mark_task_failed(self, target_id: str, session_id: str, error: str) -> None:
        """Mark a task as failed with error message."""
        try:
            async with self._driver.session() as db_session:
                await db_session.run(
                    "MERGE (t:TaskProgress {target_id: $target_id, session_id: $session_id}) "
                    "SET t.status = 'failed', t.error = $error, t.failed_at = $now",
                    target_id=target_id, session_id=session_id,
                    error=error[:500], now=datetime.utcnow().isoformat(),
                )
        except Exception as e:
            logger.debug("Failed to mark task failed: %s", e)

    async def _save_session_checkpoint(self, session_id: str, tasks_done: int, tasks_total: int) -> None:
        """Save session checkpoint for resume support."""
        try:
            async with self._driver.session() as db_session:
                await db_session.run(
                    "MATCH (s:Session {id: $id}) "
                    "SET s.checkpoint_at = $now, s.tasks_done = $done, s.tasks_total = $total",
                    id=session_id, now=datetime.utcnow().isoformat(),
                    done=tasks_done, total=tasks_total,
                )
            logger.info("Session checkpoint saved: %d/%d tasks", tasks_done, tasks_total)
        except Exception as e:
            logger.warning("Failed to save session checkpoint: %s", e)

    # ── Helpers ──────────────────────────────────────────────────

    async def _navigate_to(self, page, url: str, title: str = "") -> None:
        """Navigate: try clicking a sidebar link first (SPA), fallback to goto."""
        if title:
            clicked = await self._click_link_by_text(page, title)
            if clicked:
                await asyncio.sleep(2)
                return
        if url:
            try:
                await page.goto(url)
                await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(2)

    async def _click_link_by_text(self, page, text: str) -> bool:
        """Click a sidebar <a> whose textContent matches ``text``."""
        try:
            result = await page.evaluate(
                """(targetText) => {
                    const links = document.querySelectorAll('li > a');
                    for (const link of links) {
                        if (link.textContent?.trim() === targetText) {
                            link.click();
                            return true;
                        }
                    }
                    return false;
                }""",
                text,
            )
            return result == "true" or result is True
        except Exception:
            return False

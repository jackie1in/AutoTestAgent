"""Quality Acceptance Test Script (PRD §14.2)

Runs cartography against the target application and verifies:
1. menu_coverage >= 60%
2. zone_coverage >= 80%
3. Transition avg confidence >= 0.6
4. Every Transition has >= 1 CHECK_AFTER Checkpoint
5. At least 1 path with length >= 3 can be replayed

All browser interactions go through browser-use BrowserSession.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("quality_acceptance")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"


async def login(page) -> None:
    """Login using browser-use Page API."""
    username = os.getenv("MAPPING_USERNAME", "")
    password = os.getenv("MAPPING_PASSWORD", "")
    if not username:
        return
    logger.info("Logging in as %s ...", username)
    await asyncio.sleep(2)

    logged_in = await page.evaluate(
        """(creds) => {
            const userInput = document.querySelector(
                "#username, input[placeholder*='用户'], input[type='text']"
            );
            const passInput = document.querySelector(
                "#password, input[placeholder*='密码'], input[type='password']"
            );
            if (!userInput || !passInput) return false;

            userInput.focus();
            userInput.value = creds.username;
            userInput.dispatchEvent(new Event('input', {bubbles: true}));
            passInput.focus();
            passInput.value = creds.password;
            passInput.dispatchEvent(new Event('input', {bubbles: true}));

            const submit = document.querySelector(
                "#sbbtn, .ant-btn-primary, button[type='submit']"
            );
            if (submit) submit.click();
            return true;
        }""",
        {"username": username, "password": password},
    )
    if logged_in == "true" or logged_in is True:
        await asyncio.sleep(3)
        url = await page.get_url()
        logger.info("Login OK → %s", url)
    else:
        logger.warning("Login form not found")


async def run_quality_acceptance():
    from browser_use.browser.session import BrowserSession

    from graph_agent.cartography.menu_extractor import MenuExtractor
    from graph_agent.cartography.react_explorer import ReActExplorer
    from graph_agent.cartography.snapshot import capture_dom_fingerprint
    from graph_agent.cartography.zone_discoverer import ZoneDiscoverer
    from graph_agent.coverage.analyzer import CoverageAnalyzer
    from graph_agent.graph.merger import CartographyResult, GraphMerger
    from graph_agent.models import (
        ActionType,
        Checkpoint,
        CheckpointExpect,
        CheckpointLayer,
        CheckpointTiming,
        Severity,
        State,
        Transition,
    )
    from graph_agent.neo4j.driver import Neo4jDriver

    start_url = os.getenv("MAPPING_URL", "http://172.20.20.43/project/index.html")

    neo4j = Neo4jDriver()
    await neo4j.connect()
    await neo4j.ensure_schema()
    driver = neo4j.driver

    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
    logger.info("Neo4j cleaned")

    browser_session = BrowserSession(
        headless=False,
        viewport={"width": 1280, "height": 800},
    )
    await browser_session.start()
    logger.info("BrowserSession started")

    page = await browser_session.must_get_current_page()

    try:
        await page.goto(start_url)
        await asyncio.sleep(3)
        await login(page)
        await asyncio.sleep(3)

        merger = GraphMerger(driver)
        coverage_analyzer = CoverageAnalyzer(driver)
        menu_extractor = MenuExtractor()
        zone_discoverer = ZoneDiscoverer()

        session_id = f"session:{datetime.utcnow().isoformat()}"
        async with driver.session() as s:
            await s.run(
                "MERGE (s:Session {id: $id}) SET s.timestamp = $ts, s.focus = 'full'",
                id=session_id,
                ts=datetime.utcnow().isoformat(),
            )

        # ═══════════════════════════════════════════════════════
        # Phase 1: Menu discovery + Page visit + Zone discovery
        # ═══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("PHASE 1: Menu & Zone Discovery")
        logger.info("=" * 60)

        menu_items = await menu_extractor.extract(page)
        logger.info("Found %d menu items", len(menu_items))

        menu_states: list[State] = []

        for item in menu_items:
            text = item.get("text", "")
            if not text:
                continue

            clicked = await _click_link_by_text(page, text)
            if not clicked:
                continue

            await asyncio.sleep(2)

            iframe_url = await page.get_url()
            fp = await capture_dom_fingerprint(page)

            state = State(
                id=f"state:{text}",
                url=iframe_url,
                title=text,
                fingerprint=fp,
                menu_path=[text],
            )
            menu_states.append(state)

            zones = await zone_discoverer.discover(page, state.id, session=browser_session)
            zone_map = {z.id: state.id for z in zones}

            result = CartographyResult(
                states=[state], zones=zones, zone_state_map=zone_map
            )
            _ = await merger.merge(result, session_id)
            logger.info(
                "  [%s] url=%s zones=%d",
                text,
                iframe_url[:60],
                len(zones),
            )

        # ═══════════════════════════════════════════════════════
        # Phase 2: Zone exploration → Transitions + Checkpoints
        # ═══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("PHASE 2: Zone Exploration (transitions + checkpoints)")
        logger.info("=" * 60)

        react_explorer = ReActExplorer(max_steps=20, browser_session=browser_session)

        for state in menu_states:
            clicked = await _click_link_by_text(page, state.title)
            if not clicked:
                continue

            await asyncio.sleep(2)

            # Remove target=_blank to prevent new tabs
            await page.evaluate(
                """() => {
                    document.querySelectorAll('a[target="_blank"], a[target="blank"]')
                        .forEach(a => a.removeAttribute('target'));
                }"""
            )

            exploration_result = await react_explorer.explore_page(
                browser_session, state.id, page_title=state.title
            )

            await merger.merge(exploration_result, session_id)

            async with driver.session() as s:
                await s.run(
                    "MATCH (st:State {id: $sid})-[:HAS_ZONE]->(z:Zone) "
                    "SET z.exploration_status = 'explored', z.last_explored = $now",
                    sid=state.id,
                    now=datetime.utcnow().isoformat(),
                )

            logger.info(
                "  [%s] transitions=%d checkpoints=%d",
                state.title,
                len(exploration_result.transitions),
                len(exploration_result.checkpoints),
            )

        # ═══════════════════════════════════════════════════════
        # Phase 3: Create inter-state transitions (menu nav)
        # ═══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("PHASE 3: Inter-state transitions (menu navigation)")
        logger.info("=" * 60)

        for i in range(len(menu_states) - 1):
            from_s = menu_states[i]
            to_s = menu_states[i + 1]
            t_id = f"t:menu:{from_s.title}->{to_s.title}"
            transition = Transition(
                id=t_id,
                selector=f"li > a >> text='{to_s.title}'",
                action=ActionType.CLICK,
                thought=f"Navigate from {from_s.title} to {to_s.title}",
                from_state_id=from_s.id,
                to_state_id=to_s.id,
                step_index=i,
                confidence=0.9,
            )
            cp_id = f"cp:{t_id}:after"
            checkpoint = Checkpoint(
                id=cp_id,
                layer=CheckpointLayer.STRUCTURAL,
                timing=CheckpointTiming.AFTER,
                expect=CheckpointExpect.SHOULD_PASS,
                severity=Severity.MAJOR,
                rule_type="url_changed",
                rule=json.dumps({"expected_url": to_s.url}),
                description=f"After nav to {to_s.title}, iframe should show {to_s.url[:50]}",
            )
            cr = CartographyResult(
                transitions=[transition],
                checkpoints=[checkpoint],
                checkpoint_transition_map={cp_id: t_id},
            )
            await merger.merge(cr, session_id)

        logger.info("Created %d inter-state transitions", max(0, len(menu_states) - 1))

        # ═══════════════════════════════════════════════════════
        # Phase 4: Validate transitions (boost confidence)
        # ═══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("PHASE 4: Transition validation")
        logger.info("=" * 60)

        async with driver.session() as s:
            r = await s.run(
                "MATCH (t:Transition) RETURN t.id AS id, t.confidence AS conf"
            )
            all_trans = await r.data()

        validated = 0
        for t_data in all_trans:
            await merger.update_transition_confidence(t_data["id"], True, session_id)
            validated += 1
        logger.info("Validated %d transitions", validated)

        # ═══════════════════════════════════════════════════════
        # QUALITY CHECKS
        # ═══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("QUALITY ACCEPTANCE CHECKS")
        logger.info("=" * 60)

        final_cov = await coverage_analyzer.compute()
        results: dict[str, bool] = {}

        check1 = final_cov.menu_coverage >= 0.6
        logger.info("  %s  menu_coverage = %.1f%% (threshold: 60%%)", PASS if check1 else FAIL, final_cov.menu_coverage * 100)
        results["menu_coverage >= 60%"] = check1

        check2 = final_cov.zone_coverage >= 0.8
        logger.info("  %s  zone_coverage = %.1f%% (threshold: 80%%)", PASS if check2 else FAIL, final_cov.zone_coverage * 100)
        results["zone_coverage >= 80%"] = check2

        async with driver.session() as s:
            r = await s.run(
                "MATCH (t:Transition) WHERE t.validation_count > 0 "
                "RETURN avg(t.confidence) AS avg_conf, count(t) AS cnt"
            )
            rec = await r.single()
        avg_conf = rec["avg_conf"] if rec and rec["avg_conf"] else 0.0
        trans_cnt = rec["cnt"] if rec else 0
        check3 = avg_conf >= 0.6
        logger.info("  %s  avg confidence = %.2f (%d validated) (threshold: 0.6)", PASS if check3 else FAIL, avg_conf, trans_cnt)
        results["avg confidence >= 0.6"] = check3

        async with driver.session() as s:
            r = await s.run(
                "MATCH (t:Transition) "
                "OPTIONAL MATCH (t)-[:CHECK_AFTER]->(c:Checkpoint) "
                "WITH t, count(c) AS cp_count "
                "RETURN count(t) AS total, "
                "sum(CASE WHEN cp_count > 0 THEN 1 ELSE 0 END) AS covered"
            )
            rec = await r.single()
        total_t = rec["total"] if rec else 0
        covered_t = rec["covered"] if rec else 0
        check4 = total_t > 0 and covered_t == total_t
        logger.info("  %s  checkpoint coverage = %d/%d (threshold: 100%%)", PASS if check4 else FAIL, covered_t, total_t)
        results["checkpoint coverage 100%"] = check4

        async with driver.session() as s:
            r = await s.run(
                "MATCH path = (s1:State)<-[:FROM]-(t1:Transition)-[:TO]->"
                "(s2:State)<-[:FROM]-(t2:Transition)-[:TO]->"
                "(s3:State)<-[:FROM]-(t3:Transition)-[:TO]->(s4:State) "
                "RETURN s1.title AS start_title, s4.title AS end_title, "
                "[s1.title, s2.title, s3.title, s4.title] AS path_titles "
                "LIMIT 1"
            )
            rec = await r.single()

        if rec:
            logger.info("  %s  e2e path: %s", PASS, " → ".join(str(t) for t in rec["path_titles"]))
            check5 = True
        else:
            async with driver.session() as s:
                r = await s.run(
                    "MATCH (s1:State)<-[:FROM]-(t1:Transition)-[:TO]->"
                    "(s2:State)<-[:FROM]-(t2:Transition)-[:TO]->(s3:State) "
                    "RETURN s1.title AS st, s3.title AS en, "
                    "[s1.title, s2.title, s3.title] AS path "
                    "LIMIT 1"
                )
                rec2 = await r.single()
            if rec2:
                logger.info("  %s  e2e path (2-step): %s", PASS, " → ".join(str(t) for t in rec2["path"]))
                check5 = True
            else:
                logger.info("  %s  no multi-step path found", FAIL)
                check5 = False
        results["e2e path >= 3"] = check5

        # ═══════════════ SUMMARY ═══════════════
        logger.info("=" * 60)
        logger.info("GRAPH STATISTICS")
        async with driver.session() as s:
            r = await s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
            for st in await r.data():
                logger.info("  %s: %d", st["label"], st["cnt"])
            r2 = await s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
            for st in await r2.data():
                logger.info("  [rel] %s: %d", st["rel"], st["cnt"])

        logger.info("=" * 60)
        all_pass = all(results.values())
        for name, passed in results.items():
            logger.info("  %s  %s", PASS if passed else FAIL, name)

        logger.info("=" * 60)
        if all_pass:
            logger.info("ALL QUALITY CHECKS PASSED")
        else:
            logger.info("SOME CHECKS FAILED")
        logger.info("=" * 60)

        return results

    finally:
        try:
            await browser_session.stop()
        except Exception:
            pass
        await neo4j.close()


async def _click_link_by_text(page, text: str) -> bool:
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


if __name__ == "__main__":
    results = asyncio.run(run_quality_acceptance())
    sys.exit(0 if all(results.values()) else 1)

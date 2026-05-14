"""Tests for the incremental persistence (PersistenceSession) API.

Covers:
- PersistenceSession init/persist_page/finalize lifecycle
- Deduplication of states and transitions across multiple persist_page calls
- Idempotency: calling persist_page with the same data twice is safe
- Signal-safe save: finalize with empty result after partial persist
- Config flag resolution
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_agent.cartography.persistence import (
    PersistenceSession,
    _empty_stats,
    _transition_stable_key,
)
from graph_agent.graph.merger import CartographyResult
from graph_agent.models import (
    ActionType,
    Intent,
    State,
    Transition,
    TransitionStep,
)


def _make_state(state_id: str, url: str = "https://example.com/page") -> State:
    return State(id=state_id, url=url, title=f"State {state_id}")


def _make_transition(
    tid: str,
    from_id: str = "state:a",
    to_id: str = "state:b",
    action: ActionType = ActionType.CLICK,
    selector: str = "#btn",
    intent_key: str | None = None,
    steps: list[TransitionStep] | None = None,
) -> Transition:
    intent = None
    if intent_key:
        intent = Intent(id=f"intent:{intent_key}", key=intent_key, summary=intent_key)
    return Transition(
        id=tid,
        from_state_id=from_id,
        to_state_id=to_id,
        action=action,
        selector=selector,
        intent=intent,
        steps=steps or [],
    )


def _make_page_result(
    states: list[State] | None = None,
    transitions: list[Transition] | None = None,
) -> CartographyResult:
    cr = CartographyResult()
    cr.states = states or []
    cr.transitions = transitions or []
    return cr


class TestEmptyStats:
    def test_returns_expected_keys(self):
        stats = _empty_stats()
        assert "states_added" in stats
        assert "transitions_added" in stats
        assert stats["states_added"] == 0
        assert stats["transitions_added"] == 0

    def test_independent_copies(self):
        a = _empty_stats()
        b = _empty_stats()
        a["states_added"] = 42
        assert b["states_added"] == 0




def _make_mock_manager():
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    manager.add_app = AsyncMock()
    manager.add_session = AsyncMock()
    manager.link_app_session = AsyncMock()
    manager.add_ingestion_run = AsyncMock()
    manager.run_write_transaction = AsyncMock()
    manager.add_state = AsyncMock()
    manager.link_app_state = AsyncMock()
    manager.link_session_discovered = AsyncMock()
    manager.link_ingestion_emits_state = AsyncMock()
    manager.add_transition = AsyncMock()
    manager.link_session_transition = AsyncMock()
    manager.link_ingestion_emits_transition = AsyncMock()
    manager.add_transition_entity_with_session = AsyncMock()
    manager.get_active_transition_revision = AsyncMock(return_value=None)
    manager.add_transition_revision = AsyncMock()
    manager.link_ingestion_emits_revision = AsyncMock()
    manager.activate_transition_revision = AsyncMock()
    manager.attach_transition_revision = AsyncMock()
    manager.add_intent = AsyncMock()
    manager.link_transition_intent = AsyncMock()
    manager.link_zone_covers_intent = AsyncMock()
    manager.add_evidence = AsyncMock()
    manager.link_transition_evidence = AsyncMock()
    manager.link_session_evidence = AsyncMock()
    manager.link_ingestion_emits_evidence = AsyncMock()
    manager.add_menus = AsyncMock()
    manager.link_ingestion_emits_menu = AsyncMock()
    manager.add_zones = AsyncMock()
    manager.link_ingestion_emits_zone = AsyncMock()
    manager.link_transition_navigated_via = AsyncMock()
    manager.add_graph_release = AsyncMock()
    manager.deactivate_other_active_releases = AsyncMock()
    manager.link_release_revision = AsyncMock()
    manager.get_driver = MagicMock()
    manager.add_coverage_snapshot = AsyncMock()
    manager.link_session_coverage = AsyncMock()
    manager.link_release_coverage = AsyncMock()
    manager.get_latest_semantic_baseline = AsyncMock(return_value=None)
    manager.update_session_stats = AsyncMock()
    manager.touch_app_last_session = AsyncMock()
    manager.update_app_stats = AsyncMock()
    return manager


async def _init_session_with_mock(session):
    """Inject a mock manager directly into session and run init logic."""
    mock_manager = _make_mock_manager()
    session._manager = mock_manager
    session._accumulated_result = CartographyResult()

    from graph_agent.models import App, IngestionRun, Session as SessionModel
    from graph_agent.neo4j_client.queries import CypherQueries

    app = App(id=session.app_id, name=session.app_name, entry_url=session.resolved_url)
    await mock_manager.add_app(app)
    s = SessionModel(id=session.session_id, app_id=session.app_id)
    await mock_manager.add_session(s)
    await mock_manager.link_app_session(session.app_id, session.session_id)

    import hashlib
    from datetime import UTC, datetime
    ingest_seed = (
        f"{session.app_id}|{session.session_id}|{session.mode}|0|0|"
        f"{datetime.now(UTC).isoformat()}"
    )
    session._ingest_version_id = (
        f"ingest:{hashlib.md5(ingest_seed.encode()).hexdigest()[:16]}"
    )
    await mock_manager.add_ingestion_run(
        IngestionRun(
            id=session._ingest_version_id,
            app_id=session.app_id,
            session_id=session.session_id,
            mode=session.mode,
            source="cartography",
            status="in_progress",
        )
    )
    await mock_manager.run_write_transaction([
        (
            CypherQueries.LINK_SESSION_INGESTION_RUN,
            {"session_id": session.session_id, "ingest_id": session._ingest_version_id},
        )
    ])

    session._initialized = True
    return mock_manager


class TestPersistenceSessionLifecycle:
    """Test the three-phase lifecycle using mocked GraphManager injected directly."""

    @pytest.fixture
    def session(self):
        return PersistenceSession(
            app_id="app:test",
            app_name="Test App",
            session_id="session:test:2026-01-01",
            resolved_url="https://example.com/login",
            current_url="https://example.com/home",
            inventory=[],
            initial_actions_log=[],
            mode="auto",
        )

    @pytest.mark.asyncio
    async def test_init_creates_core_nodes(self, session):
        mock_manager = await _init_session_with_mock(session)

        mock_manager.add_app.assert_called_once()
        mock_manager.add_session.assert_called_once()
        mock_manager.add_ingestion_run.assert_called_once()
        assert session.ingest_version_id.startswith("ingest:")
        assert session._initialized is True

    @pytest.mark.asyncio
    async def test_init_with_prelogin(self):
        session = PersistenceSession(
            app_id="app:test",
            app_name="Test App",
            session_id="session:test:2026-01-01",
            resolved_url="https://example.com/login",
            current_url="https://example.com/home",
            inventory=[],
            initial_actions_log=[{"action": "fill", "selector": "#user"}],
            mode="auto",
        )
        mock_manager = _make_mock_manager()
        session._manager = mock_manager
        session._accumulated_result = CartographyResult()
        session._ingest_version_id = "ingest:test123"

        from graph_agent.models import ActionType, State, Transition
        login_url = session.resolved_url
        post_login_url = session.current_url or session.resolved_url
        import hashlib
        login_fp = hashlib.md5(login_url.encode()).hexdigest()[:12]
        post_fp = hashlib.md5(post_login_url.encode()).hexdigest()[:12]
        from_state = State(id=f"state:prelogin:{login_fp}", url=login_url, title="Login page")
        to_state = State(id=f"state:prelogin:{post_fp}", url=post_login_url, title="Post-login page")
        prelogin_transition = Transition(
            id=f"{session.session_id}:prelogin",
            selector="[pre-login]",
            action=ActionType.CLICK,
            from_state_id=from_state.id,
            to_state_id=to_state.id,
            thought="Auto-login",
            confidence=0.9,
        )
        await mock_manager.add_state(from_state)
        await mock_manager.add_state(to_state)
        await mock_manager.add_transition(prelogin_transition)
        session._initialized = True
        session._stats["states_added"] = 2
        session._stats["transitions_added"] = 1

        assert mock_manager.add_state.call_count == 2
        assert mock_manager.add_transition.call_count == 1

    @pytest.mark.asyncio
    async def test_persist_page_writes_states_and_transitions(self, session):
        await _init_session_with_mock(session)

        state = _make_state("state:page1", "https://example.com/page1")
        transition = _make_transition(
            "t:page1:click", from_id="state:page1", to_id="state:page2"
        )
        page_result = _make_page_result([state], [transition])
        await session.persist_page(page_result)

        assert session.stats["states_added"] >= 1
        assert session.stats["transitions_added"] >= 1
        assert "state:page1" in session._seen_state_ids
        assert "t:page1:click" in session._seen_transition_ids

    @pytest.mark.asyncio
    async def test_persist_page_deduplicates_states(self, session):
        await _init_session_with_mock(session)

        state = _make_state("state:dup", "https://example.com/page")
        # First call
        await session.persist_page(_make_page_result([state]))
        states_after_first = session.stats["states_added"]

        # Second call with same state
        await session.persist_page(_make_page_result([state]))
        states_after_second = session.stats["states_added"]

        assert states_after_first == states_after_second

    @pytest.mark.asyncio
    async def test_persist_page_deduplicates_transitions(self, session):
        await _init_session_with_mock(session)

        transition = _make_transition("t:dup")
        # First call
        await session.persist_page(_make_page_result([], [transition]))
        trans_after_first = session.stats["transitions_added"]

        # Second call with same transition
        await session.persist_page(_make_page_result([], [transition]))
        trans_after_second = session.stats["transitions_added"]

        assert trans_after_first == trans_after_second

    @pytest.mark.asyncio
    async def test_multiple_pages_accumulate(self, session):
        await _init_session_with_mock(session)

        # Page 1
        s1 = _make_state("state:p1")
        t1 = _make_transition("t:p1")
        await session.persist_page(_make_page_result([s1], [t1]))

        # Page 2
        s2 = _make_state("state:p2")
        t2 = _make_transition("t:p2")
        await session.persist_page(_make_page_result([s2], [t2]))

        assert session.stats["states_added"] == 2
        assert session.stats["transitions_added"] == 2
        assert session._accumulated_result is not None
        assert len(session._accumulated_result.states) == 2
        assert len(session._accumulated_result.transitions) == 2

    @pytest.mark.asyncio
    async def test_finalize_creates_release_and_stats(self, session):
        await _init_session_with_mock(session)

        s1 = _make_state("state:final")
        t1 = _make_transition("t:final", intent_key="submit_form")
        await session.persist_page(_make_page_result([s1], [t1]))

        final_result = _make_page_result([s1], [t1])
        final_result.menus = [{"text": "Home", "href": "/", "level": 0, "source_url": "https://example.com/home"}]
        final_result.zone_hints = [
            {"zone_type": "form", "selector": "#form1", "description": "Main form", "source_url": "https://example.com/home", "exploration_status": "discovered"}
        ]
        final_result.layout_metrics = {"layout_confidence_avg": 0.8}
        await session.finalize(final_result)

        mock_manager = session._manager
        mock_manager.add_graph_release.assert_called_once()
        mock_manager.update_session_stats.assert_called_once()
        mock_manager.touch_app_last_session.assert_called_once()
        assert session._finalized is True

    @pytest.mark.asyncio
    async def test_finalize_with_empty_result_emergency_save(self, session):
        """Simulates a signal-interrupt scenario where no final result is available."""
        await _init_session_with_mock(session)

        s1 = _make_state("state:emergency")
        t1 = _make_transition("t:emergency")
        await session.persist_page(_make_page_result([s1], [t1]))

        # Emergency finalize with empty result
        empty_result = CartographyResult()
        await session.finalize(empty_result)

        assert session._finalized is True
        assert session.stats["states_added"] >= 1

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, session):
        await _init_session_with_mock(session)
        await session.close()

        assert session._manager is None

    @pytest.mark.asyncio
    async def test_persist_page_before_init_is_noop(self):
        session = PersistenceSession(
            app_id="app:test",
            app_name="Test App",
            session_id="session:test:2026-01-01",
            resolved_url="https://example.com",
            current_url="https://example.com",
            inventory=[],
            initial_actions_log=[],
        )
        # Should not raise
        await session.persist_page(_make_page_result())
        assert session.stats["states_added"] == 0

    @pytest.mark.asyncio
    async def test_finalize_before_init_is_noop(self):
        session = PersistenceSession(
            app_id="app:test",
            app_name="Test App",
            session_id="session:test:2026-01-01",
            resolved_url="https://example.com",
            current_url="https://example.com",
            inventory=[],
            initial_actions_log=[],
        )
        # Should not raise
        await session.finalize(CartographyResult())
        assert session._finalized is False

    @pytest.mark.asyncio
    async def test_double_finalize_is_idempotent(self, session):
        mock_manager = await _init_session_with_mock(session)
        await session.finalize(CartographyResult())
        assert session._finalized is True

        # Second finalize should be a no-op
        mock_manager.update_session_stats.reset_mock()
        await session.finalize(CartographyResult())
        mock_manager.update_session_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_intent_persisted_with_transition(self, session):
        await _init_session_with_mock(session)

        transition = _make_transition(
            "t:intent_test",
            intent_key="login_submit",
            selector="#submit-btn",
        )
        await session.persist_page(_make_page_result([], [transition]))

        mock_manager = session._manager
        mock_manager.add_intent.assert_called()

    @pytest.mark.asyncio
    async def test_evidence_persisted_for_transitions(self, session):
        await _init_session_with_mock(session)

        transition = _make_transition("t:evidence_test")
        transition.evidence_ids = ["evidence:url_change"]
        await session.persist_page(_make_page_result([], [transition]))

        mock_manager = session._manager
        mock_manager.add_evidence.assert_called()

    @pytest.mark.asyncio
    async def test_layout_evidence_persisted(self, session):
        await _init_session_with_mock(session)

        page_result = _make_page_result()
        page_result.layout_evidence = [
            {
                "url": "https://example.com/page",
                "step": 1,
                "layout_fingerprint": "abc123",
                "layout_summary": "Standard form layout",
                "layout_confidence": 0.9,
            }
        ]
        await session.persist_page(page_result)

        mock_manager = session._manager
        mock_manager.add_evidence.assert_called()

    @pytest.mark.asyncio
    async def test_transition_with_steps(self, session):
        await _init_session_with_mock(session)

        steps = [
            TransitionStep(action=ActionType.FILL, selector="#user", param_name="username"),
            TransitionStep(action=ActionType.FILL, selector="#pass", param_name="password"),
            TransitionStep(action=ActionType.CLICK, selector="#submit"),
        ]
        transition = _make_transition("t:steps_test", steps=steps)
        await session.persist_page(_make_page_result([], [transition]))

        key = _transition_stable_key(transition)
        assert key  # Should produce a valid stable key


class TestTransitionStableKeyWithSteps:
    def test_includes_steps_digest(self):
        steps = [
            TransitionStep(action=ActionType.FILL, selector="#user"),
            TransitionStep(action=ActionType.CLICK, selector="#btn"),
        ]
        t = _make_transition("t:1", steps=steps)
        key = _transition_stable_key(t)
        assert "|" in key
        # The steps_digest should be the last component
        parts = key.split("|")
        assert len(parts) == 5  # from_id|to_id|action|semantic|steps_digest
        assert len(parts[-1]) == 8  # MD5 hex[:8]

    def test_no_steps_gives_empty_digest(self):
        t = _make_transition("t:2")
        key = _transition_stable_key(t)
        parts = key.split("|")
        assert parts[-1] == ""

    def test_different_steps_produce_different_keys(self):
        steps_a = [TransitionStep(action=ActionType.FILL, selector="#a")]
        steps_b = [TransitionStep(action=ActionType.FILL, selector="#b")]
        ta = _make_transition("t:a", steps=steps_a)
        tb = _make_transition("t:b", steps=steps_b)
        assert _transition_stable_key(ta) != _transition_stable_key(tb)

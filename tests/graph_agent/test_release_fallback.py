from __future__ import annotations

import pytest

from graph_agent.cartography.knowledge_broker import KnowledgeBroker, KnowledgeQueryInput
from graph_agent.web.app import _get_edges_from_neo4j


class _FakeResult:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records
        self._idx = 0

    async def peek(self):
        return self._records[0] if self._records else None

    async def single(self):
        return self._records[0] if self._records else None

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= len(self._records):
            raise StopAsyncIteration
        row = self._records[self._idx]
        self._idx += 1
        return row


class _FakeSession:
    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[str] = []
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, query: str, **kwargs):
        self.calls.append(query)
        if self._idx >= len(self._results):
            return _FakeResult([])
        result = self._results[self._idx]
        self._idx += 1
        return result


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def session(self):
        return self._session


def _row() -> dict[str, object]:
    return {
        "id": "t:1",
        "step_index": 1,
        "from_state_id": "s:1",
        "to_state_id": "s:2",
        "source_url": "https://app/a",
        "target_url": "https://app/b",
        "selector": "#go",
        "action": "click",
        "tab_id": "tab-0",
        "target_tab_id": None,
        "tab_action": None,
        "intent_failure_reason": None,
        "param_name": None,
        "action_value": None,
        "thought": "go next",
        "element_snapshot": None,
        "frame_path": None,
        "intent": None,
    }


@pytest.mark.asyncio
async def test_get_edges_prefers_release_when_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAPPING_RELEASE_ID", "release:test")
    monkeypatch.delenv("MAPPING_APP_NAME", raising=False)
    fake_session = _FakeSession(results=[_FakeResult([_row()])])
    driver = _FakeDriver(fake_session)

    edges = await _get_edges_from_neo4j(driver)

    assert len(edges) == 1
    assert edges[0].edge_id == "t:1"
    assert len(fake_session.calls) == 1


@pytest.mark.asyncio
async def test_get_edges_fallbacks_to_current_when_release_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAPPING_RELEASE_ID", "release:missing")
    monkeypatch.delenv("MAPPING_APP_NAME", raising=False)
    fake_session = _FakeSession(
        results=[_FakeResult([]), _FakeResult([_row()])]
    )
    driver = _FakeDriver(fake_session)

    edges = await _get_edges_from_neo4j(driver)

    assert len(edges) == 1
    assert edges[0].edge_id == "t:1"
    assert len(fake_session.calls) == 2


class _BrokerFakeManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get_knowledge_release_rows(self, **kwargs):
        return []

    async def get_knowledge_legacy_rows(self, **kwargs):
        return [
            {
                "id": "t:legacy",
                "confidence": 0.8,
                "selector": ".legacy-selector",
                "action": "click",
                "source_url": "https://app/a",
                "target_url": "https://app/b",
                "intent": {"key": "legacy.intent", "summary": "legacy path"},
            }
        ]


@pytest.mark.asyncio
async def test_knowledge_broker_release_fallback_to_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "graph_agent.cartography.knowledge_broker.GraphManager",
        _BrokerFakeManager,
    )
    broker = KnowledgeBroker()
    result = await broker.query(
        KnowledgeQueryInput(
            app_id="app:demo:1",
            session_id="session:1",
            current_url="https://app/a",
            page_type="dashboard",
            release_id="release:missing",
        )
    )
    assert result.meta.source == "legacy"
    assert result.transition_hints


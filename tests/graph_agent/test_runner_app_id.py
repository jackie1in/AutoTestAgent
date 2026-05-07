from __future__ import annotations

import pytest

from graph_agent.cartography import runner
from graph_agent.cartography.runner import (
    _normalize_app_name,
    _stable_app_id_from_app_name,
)


def test_normalize_app_name_to_stable_slug():
    assert _normalize_app_name(" Demo.EXAMPLE.com ") == "demo.example.com"
    assert _normalize_app_name("demo example/com") == "demo-example-com"
    assert _normalize_app_name("___") == "___"


def test_stable_app_id_same_for_same_app_name():
    first = _stable_app_id_from_app_name("Demo.Example.com")
    second = _stable_app_id_from_app_name(" demo.example.com ")
    assert first == second
    assert first == "app:demo.example.com"


@pytest.mark.asyncio
async def test_run_manual_mapping_uses_stable_app_id_when_not_provided(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeManualCaptureSession:
        @classmethod
        def from_graph_context(cls, **_kwargs):
            return cls()

        @classmethod
        def from_raw(cls, **_kwargs):
            return cls()

        def record_action(self, **_kwargs):
            return None

        async def to_cartography_result(self):
            return object()

    async def _fake_persist_mapping_result(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(runner, "ManualCaptureSession", _FakeManualCaptureSession)
    monkeypatch.setattr(runner, "persist_mapping_result", _fake_persist_mapping_result)

    resolved = await runner.run_manual_mapping(
        app_name=" Demo.Example.com ",
        session_id="session:manual:1",
        events=[],
        mode="manual_raw",
        start_url="https://demo.example.com",
    )

    assert resolved == "app:demo.example.com"
    assert captured.get("app_id") == "app:demo.example.com"


@pytest.mark.asyncio
async def test_run_manual_mapping_respects_explicit_app_id(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeManualCaptureSession:
        @classmethod
        def from_graph_context(cls, **_kwargs):
            return cls()

        @classmethod
        def from_raw(cls, **_kwargs):
            return cls()

        def record_action(self, **_kwargs):
            return None

        async def to_cartography_result(self):
            return object()

    async def _fake_persist_mapping_result(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(runner, "ManualCaptureSession", _FakeManualCaptureSession)
    monkeypatch.setattr(runner, "persist_mapping_result", _fake_persist_mapping_result)

    resolved = await runner.run_manual_mapping(
        app_name="demo.example.com",
        session_id="session:manual:2",
        events=[],
        mode="manual_graph_assisted",
        app_id="app:custom-manual",
        operator_id="human:operator",
        start_state_hint="state:start",
        start_url="https://demo.example.com/start",
    )

    assert resolved == "app:custom-manual"
    assert captured.get("app_id") == "app:custom-manual"

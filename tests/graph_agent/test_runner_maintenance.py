from __future__ import annotations

import pytest

from graph_agent.cartography import runner


@pytest.mark.asyncio
async def test_run_post_mapping_maintenance_dry_run(monkeypatch):
    class _FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def evaluate_semantic_stability_gate(self, *, app_id: str):
            return {"app_id": app_id, "passed": True}

        async def evaluate_retention_plan(self, **kwargs):
            return {"mode": "dry_run", **kwargs}

        async def run_retention(self, **_kwargs):
            raise AssertionError("run_retention should not be called in dry-run")

    monkeypatch.setattr(runner, "GraphManager", _FakeManager)
    report = await runner.run_post_mapping_maintenance(
        app_id="app:demo",
        retention_execute=False,
        keep_releases=7,
        min_age_days=30,
        batch_size=222,
    )
    assert report["app_id"] == "app:demo"
    assert report["retention"]["mode"] == "dry_run"
    assert report["retention"]["keep_releases"] == 7
    assert report["post_gate"]["passed"] is True


@pytest.mark.asyncio
async def test_run_post_mapping_maintenance_execute(monkeypatch):
    class _FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def evaluate_semantic_stability_gate(self, *, app_id: str):
            return {"app_id": app_id, "passed": True}

        async def evaluate_retention_plan(self, **_kwargs):
            raise AssertionError("evaluate_retention_plan should not be called in execute")

        async def run_retention(self, **kwargs):
            return {"mode": "execute", **kwargs}

    monkeypatch.setattr(runner, "GraphManager", _FakeManager)
    report = await runner.run_post_mapping_maintenance(
        app_id="app:demo",
        retention_execute=True,
        keep_releases=5,
        min_age_days=14,
        batch_size=500,
    )
    assert report["retention"]["mode"] == "execute"
    assert report["retention"]["batch_size"] == 500

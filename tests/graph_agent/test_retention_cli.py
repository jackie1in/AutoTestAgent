from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO

import pytest

from graph_agent.cartography import retention


@pytest.mark.asyncio
async def test_run_retention_dry_run(monkeypatch):
    class _FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def evaluate_retention_plan(self, **kwargs):
            return {
                "app_id": kwargs["app_id"],
                "candidate_counts": {"releases": 2},
                "protected_counts": {"active_releases": 1},
            }

        async def run_retention(self, **_kwargs):
            raise AssertionError("run_retention should not be called in dry-run")

    monkeypatch.setattr(retention, "GraphManager", _FakeManager)
    report = await retention._run_retention(
        app_id="app:demo",
        keep_releases=5,
        min_age_days=14,
        dry_run=True,
        batch_size=500,
    )
    assert report["dry_run"] is True
    assert report["app_id"] == "app:demo"
    assert report["warnings"] == []


def test_retention_main_execute_exit_code_with_warning(monkeypatch):
    report = {
        "app_id": "app:demo",
        "deleted_counts": {"releases": 1},
        "protected_counts": {"active_releases": 2},
        "warnings": ["active_release_count_expected_1_got_2"],
    }

    async def _fake_run_retention(**_kwargs):
        return report

    monkeypatch.setattr(retention, "_run_retention", _fake_run_retention)
    monkeypatch.setattr(
        retention.sys,
        "argv",
        ["retention", "--app-id", "app:demo", "--execute"],
    )
    buf = StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        retention.main()
    assert exc.value.code == 1
    parsed = json.loads(buf.getvalue())
    assert parsed["app_id"] == "app:demo"


def test_retention_main_dry_run_exit_zero(monkeypatch):
    report = {
        "app_id": "app:demo",
        "candidate_counts": {"releases": 0},
        "protected_counts": {"active_releases": 1},
        "warnings": [],
    }

    async def _fake_run_retention(**_kwargs):
        return report

    monkeypatch.setattr(retention, "_run_retention", _fake_run_retention)
    monkeypatch.setattr(retention.sys, "argv", ["retention", "--app-id", "app:demo"])
    with pytest.raises(SystemExit) as exc:
        retention.main()
    assert exc.value.code == 0

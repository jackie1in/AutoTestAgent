from __future__ import annotations

import os

import pytest

from graph_agent.cartography import runner


def _baseline_urls() -> list[str]:
    raw = (os.getenv("CARTOGRAPHY_E2E_BASELINE_URLS") or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _baseline_enabled() -> bool:
    return (os.getenv("CARTOGRAPHY_E2E_BASELINE_ENABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.asyncio
async def test_e2e_baseline_mapping_semantic_stability():
    urls = _baseline_urls()
    if not _baseline_enabled() or len(urls) < 1:
        pytest.skip("set CARTOGRAPHY_E2E_BASELINE_ENABLED=1 and baseline URLs to run")
    app_id = await runner.run_mapping(url=urls[0], inventory_path="graph_agent/data/element_inventory.json")
    report = await runner.run_post_mapping_maintenance(app_id=app_id, retention_execute=False)
    assert report["pre_gate"]["sample_size"] >= 1


@pytest.mark.asyncio
async def test_e2e_baseline_retention_dry_run():
    urls = _baseline_urls()
    if not _baseline_enabled() or len(urls) < 2:
        pytest.skip("set at least 2 baseline URLs to run this test")
    app_id = await runner.run_mapping(url=urls[1], inventory_path="graph_agent/data/element_inventory.json")
    report = await runner.run_post_mapping_maintenance(
        app_id=app_id,
        retention_execute=False,
        keep_releases=5,
        min_age_days=14,
    )
    assert "candidate_counts" in report["retention"]

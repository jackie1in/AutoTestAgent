from __future__ import annotations

import pytest

from graph_agent.neo4j_client.manager import GraphManager


@pytest.mark.asyncio
async def test_update_session_stats_collects_dynamic_prefixed_fields():
    manager = GraphManager()
    captured: dict[str, object] = {}

    async def _fake_run_write(query: str, **params):
        captured["query"] = query
        captured["params"] = params

    manager._run_write = _fake_run_write  # type: ignore[method-assign]

    await manager.update_session_stats(
        session_id="session:1",
        stats={
            "states_added": 1,
            "layout_confidence_avg": 0.91,
            "skip_page_count": 2,
            "knowledge_source": "release",
            "semantic_state_signature": "abc123",
            "random_metric": "ignored",
            "layout_nested": {"k": "v"},
        },
    )

    params = captured["params"]  # type: ignore[assignment]
    assert isinstance(params, dict)
    dynamic_stats = params["dynamic_stats"]
    assert isinstance(dynamic_stats, dict)
    assert dynamic_stats["layout_confidence_avg"] == pytest.approx(0.91)
    assert dynamic_stats["skip_page_count"] == 2
    assert dynamic_stats["knowledge_source"] == "release"
    assert dynamic_stats["semantic_state_signature"] == "abc123"
    assert dynamic_stats["layout_nested"] == '{"k": "v"}'
    assert "random_metric" not in dynamic_stats


@pytest.mark.asyncio
async def test_clear_page_menus_calls_repository_soft_deactivate():
    manager = GraphManager()

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def mark_page_menus_inactive(self, app_id: str, page_url: str) -> None:
            self.calls.append((app_id, page_url))

    repo = _FakeRepo()
    manager._repo = repo  # type: ignore[assignment]

    await manager._clear_page_menus("app:demo", "https://demo/page")
    assert repo.calls == [("app:demo", "https://demo/page")]


@pytest.mark.asyncio
async def test_get_semantic_stability_trend_delegates_to_repository():
    manager = GraphManager()

    class _FakeRepo:
        async def get_semantic_stability_trend(
            self,
            *,
            app_id: str,
            limit: int = 10,
            default_threshold: float = 90.0,
        ):
            return [
                {
                    "session_id": "session:1",
                    "score": 96.2,
                    "threshold": default_threshold,
                    "passed": True,
                    "app_id": app_id,
                    "limit": limit,
                }
            ]

    manager._repo = _FakeRepo()  # type: ignore[assignment]
    rows = await manager.get_semantic_stability_trend(
        app_id="app:demo",
        limit=5,
        default_threshold=92.0,
    )
    assert len(rows) == 1
    assert rows[0]["app_id"] == "app:demo"
    assert rows[0]["limit"] == 5
    assert rows[0]["threshold"] == pytest.approx(92.0)


@pytest.mark.asyncio
async def test_evaluate_semantic_stability_gate_passes():
    manager = GraphManager()

    async def _fake_get_semantic_stability_trend(**_kwargs):
        return [
            {"score": 94.0, "passed": True},
            {"score": 95.0, "passed": True},
            {"score": 93.0, "passed": True},
            {"score": 96.0, "passed": True},
            {"score": 94.0, "passed": True},
            {"score": 95.0, "passed": True},
            {"score": 93.5, "passed": True},
            {"score": 94.5, "passed": True},
            {"score": 96.0, "passed": True},
            {"score": 95.0, "passed": True},
        ]

    manager.get_semantic_stability_trend = (  # type: ignore[method-assign]
        _fake_get_semantic_stability_trend
    )

    result = await manager.evaluate_semantic_stability_gate(app_id="app:demo")
    assert result["passed"] is True
    assert result["sample_size"] == 10
    assert result["consecutive_pass_observed"] == 10
    assert result["failure_reasons"] == []


@pytest.mark.asyncio
async def test_evaluate_semantic_stability_gate_fails_with_reason():
    manager = GraphManager()

    async def _fake_get_semantic_stability_trend(**_kwargs):
        return [
            {"score": 92.0, "passed": True},
            {"score": 88.0, "passed": False},
            {"score": 91.0, "passed": True},
        ]

    manager.get_semantic_stability_trend = (  # type: ignore[method-assign]
        _fake_get_semantic_stability_trend
    )

    result = await manager.evaluate_semantic_stability_gate(
        app_id="app:demo",
        window=5,
        min_score=90.0,
        avg_score=92.0,
        consecutive_pass_required=3,
    )
    assert result["passed"] is False
    reasons = result["failure_reasons"]
    assert isinstance(reasons, list)
    assert any(str(item).startswith("insufficient_samples") for item in reasons)
    assert any(str(item).startswith("consecutive_passed") for item in reasons)


@pytest.mark.asyncio
async def test_evaluate_retention_plan_delegates_to_repository():
    manager = GraphManager()

    class _FakeRepo:
        async def plan_retention(
            self,
            *,
            app_id: str,
            keep_releases: int,
            min_age_days: int,
        ):
            return {
                "app_id": app_id,
                "keep_releases": keep_releases,
                "min_age_days": min_age_days,
                "candidate_counts": {"releases": 2},
            }

    manager._repo = _FakeRepo()  # type: ignore[assignment]
    plan = await manager.evaluate_retention_plan(
        app_id="app:demo",
        keep_releases=7,
        min_age_days=30,
    )
    assert plan["app_id"] == "app:demo"
    assert plan["keep_releases"] == 7
    assert plan["min_age_days"] == 30
    assert plan["candidate_counts"]["releases"] == 2


@pytest.mark.asyncio
async def test_run_retention_delegates_to_repository():
    manager = GraphManager()

    class _FakeRepo:
        async def execute_retention(
            self,
            *,
            app_id: str,
            keep_releases: int,
            min_age_days: int,
            batch_size: int,
        ):
            return {
                "app_id": app_id,
                "keep_releases": keep_releases,
                "min_age_days": min_age_days,
                "deleted_counts": {"releases": 3},
                "batch_size": batch_size,
            }

    manager._repo = _FakeRepo()  # type: ignore[assignment]
    report = await manager.run_retention(
        app_id="app:demo",
        keep_releases=5,
        min_age_days=14,
        batch_size=256,
    )
    assert report["app_id"] == "app:demo"
    assert report["keep_releases"] == 5
    assert report["min_age_days"] == 14
    assert report["batch_size"] == 256
    assert report["deleted_counts"]["releases"] == 3


@pytest.mark.asyncio
async def test_runner_queries_prefer_app_id_path():
    manager = GraphManager()
    captured: dict[str, object] = {}

    class _FakeRepo:
        async def get_runner_warm_start_candidates(self, **kwargs):
            captured["warm"] = kwargs
            return [{"target_url": "https://demo"}]

        async def get_latest_active_release_by_app_id(self, app_id: str):
            captured["release_app_id"] = app_id
            return "release:1"

    manager._repo = _FakeRepo()  # type: ignore[assignment]
    rows = await manager.get_runner_warm_start_candidates(
        app_id="app:demo",
        app_name="demo.local",
        limit=12,
    )
    release_id = await manager.get_latest_active_release_by_app_id("app:demo")
    assert rows and rows[0]["target_url"] == "https://demo"
    assert release_id == "release:1"
    warm_kwargs = captured.get("warm")
    assert isinstance(warm_kwargs, dict)
    assert warm_kwargs["app_id"] == "app:demo"
    assert warm_kwargs["app_name"] == "demo.local"
    assert warm_kwargs["limit"] == 12
    assert captured["release_app_id"] == "app:demo"

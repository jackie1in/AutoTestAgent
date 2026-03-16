from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from graph_agent.run_auto_repair_loop import REPAIRABLE_ROOT_CAUSES, run_auto_repair_loop


@pytest.mark.asyncio
async def test_auto_repair_loop_stops_after_success(tmp_path: Path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"directed": true, "multigraph": true, "graph": {}, "nodes": [], "links": []}',
        encoding="utf-8",
    )

    diagnostics_fail = {
        "playback": {
            "success": False,
            "error": "selector: timeout",
            "failed_edge_id": "step-1",
            "failed_step_index": 0,
        }
    }
    diagnostics_ok = {
        "playback": {
            "success": True,
            "error": None,
            "failed_edge_id": None,
            "failed_step_index": None,
        }
    }

    with (
        patch(
            "graph_agent.run_auto_repair_loop.load_graph",
            return_value=type("G", (), {"graph": {"start_url": "https://example.com"}})(),
        ),
        patch(
            "graph_agent.run_auto_repair_loop.collect_playback_diagnostics",
            new_callable=AsyncMock,
            side_effect=[diagnostics_fail, diagnostics_ok],
        ) as mock_diag,
        patch(
            "graph_agent.run_auto_repair_loop.re_infer_with_feedback",
            new_callable=AsyncMock,
            return_value={"total": 1, "succeeded": 1, "failed": 0},
        ) as mock_reinfer,
        patch(
            "graph_agent.run_auto_repair_loop._snapshot_graph_metrics",
            side_effect=[
                {"intent_missing_count": 2, "intent_success_rate": 0.5, "semantic_consistency_rate": 0.8},
                {"intent_missing_count": 1, "intent_success_rate": 0.75, "semantic_consistency_rate": 0.9},
            ],
        ),
    ):
        result = await run_auto_repair_loop(
            graph_path=graph_path,
            intent_query="auth.login",
            max_rounds=2,
        )

    assert result["success"] is True
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["repair"]["succeeded"] == 1
    assert result["rounds"][0]["root_cause"] == "selector"
    assert result["rounds"][0]["root_cause_detail"] == "selector.timeout"
    assert result["rounds"][0]["before"]["intent_missing_count"] == 2
    assert result["rounds"][0]["after"]["intent_missing_count"] == 1
    assert result["rounds"][0]["delta"]["intent_missing_count"] == -1
    assert result["rounds"][0]["delta"]["intent_success_rate"] == 0.25
    assert result["rounds"][1]["repair"]["total"] == 0
    assert mock_diag.await_count == 2
    mock_reinfer.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_repair_loop_breaks_when_no_repairable_failures(tmp_path: Path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"directed": true, "multigraph": true, "graph": {}, "nodes": [], "links": []}',
        encoding="utf-8",
    )

    diagnostics_fail = {
        "playback": {
            "success": False,
            "error": "tab: missing tab",
            "failed_edge_id": "step-2",
            "failed_step_index": 1,
        }
    }

    with (
        patch(
            "graph_agent.run_auto_repair_loop.load_graph",
            return_value=type("G", (), {"graph": {"start_url": "https://example.com"}})(),
        ),
        patch(
            "graph_agent.run_auto_repair_loop.collect_playback_diagnostics",
            new_callable=AsyncMock,
            side_effect=[diagnostics_fail, diagnostics_fail],
        ),
        patch(
            "graph_agent.run_auto_repair_loop.re_infer_with_feedback",
            new_callable=AsyncMock,
            return_value={"total": 0, "succeeded": 0, "failed": 0},
        ) as mock_reinfer,
    ):
        result = await run_auto_repair_loop(
            graph_path=graph_path,
            intent_query="auth.login",
            max_rounds=2,
        )

    assert result["success"] is False
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["skip_reason"] is not None
    assert "REPAIRABLE_ROOT_CAUSES" in result["rounds"][0]["skip_reason"]
    mock_reinfer.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_repair_loop_skips_iframe_root_cause(tmp_path: Path):
    """iframe root_cause is not in REPAIRABLE_ROOT_CAUSES so re-infer should be skipped."""
    assert "iframe" not in REPAIRABLE_ROOT_CAUSES

    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"directed": true, "multigraph": true, "graph": {}, "nodes": [], "links": []}',
        encoding="utf-8",
    )

    diagnostics_fail = {
        "playback": {
            "success": False,
            "error": "iframe: Failed to locate iframe",
            "failed_edge_id": "step-3",
            "failed_step_index": 2,
        }
    }

    with (
        patch(
            "graph_agent.run_auto_repair_loop.load_graph",
            return_value=type("G", (), {"graph": {"start_url": "https://example.com"}})(),
        ),
        patch(
            "graph_agent.run_auto_repair_loop.collect_playback_diagnostics",
            new_callable=AsyncMock,
            side_effect=[diagnostics_fail, diagnostics_fail],
        ),
        patch(
            "graph_agent.run_auto_repair_loop.re_infer_with_feedback",
            new_callable=AsyncMock,
        ) as mock_reinfer,
    ):
        result = await run_auto_repair_loop(
            graph_path=graph_path,
            intent_query="some.intent",
            max_rounds=3,
        )

    assert result["success"] is False
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["root_cause"] == "iframe"
    assert result["rounds"][0]["root_cause_detail"] == "iframe.not_found"
    assert "skip_reason" in result["rounds"][0]
    mock_reinfer.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_repair_loop_before_after_delta_with_none_metrics(tmp_path: Path):
    """When graph metrics are None, delta should be None without error."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"directed": true, "multigraph": true, "graph": {}, "nodes": [], "links": []}',
        encoding="utf-8",
    )

    diagnostics_fail = {
        "playback": {
            "success": False,
            "error": "selector: not found",
            "failed_edge_id": "step-1",
            "failed_step_index": 0,
        }
    }
    diagnostics_ok = {
        "playback": {
            "success": True,
            "error": None,
            "failed_edge_id": None,
            "failed_step_index": None,
        }
    }

    with (
        patch(
            "graph_agent.run_auto_repair_loop.load_graph",
            return_value=type("G", (), {"graph": {"start_url": "https://x.com"}})(),
        ),
        patch(
            "graph_agent.run_auto_repair_loop.collect_playback_diagnostics",
            new_callable=AsyncMock,
            side_effect=[diagnostics_fail, diagnostics_ok],
        ),
        patch(
            "graph_agent.run_auto_repair_loop.re_infer_with_feedback",
            new_callable=AsyncMock,
            return_value={"total": 1, "succeeded": 1, "failed": 0},
        ),
        patch(
            "graph_agent.run_auto_repair_loop._snapshot_graph_metrics",
            return_value={
                "intent_missing_count": None,
                "intent_success_rate": None,
                "semantic_consistency_rate": None,
            },
        ),
    ):
        result = await run_auto_repair_loop(
            graph_path=graph_path,
            intent_query="auth.login",
            max_rounds=2,
        )

    r0 = result["rounds"][0]
    assert r0["delta"]["intent_missing_count"] is None
    assert r0["delta"]["intent_success_rate"] is None
    assert r0["delta"]["semantic_consistency_rate"] is None

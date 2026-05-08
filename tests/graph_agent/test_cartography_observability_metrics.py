from __future__ import annotations

from graph_agent.cartography.intervention_queue import evaluate_intervention_need
from graph_agent.cartography.mapping_pipeline import summarize_captcha_metrics_from_history
from graph_agent.cartography.runtime_watchdog import RuntimeWatchdog, RuntimeWatchdogConfig


def test_summarize_captcha_metrics_from_history_aggregates_codes_and_wait() -> None:
    history = [
        {"action_name": "solve_captcha", "action_result": "CAPTCHA_OK mode=auto wait_ms=0"},
        {
            "action_name": "solve_captcha",
            "action_result": "CAPTCHA_MANUAL_EMPTY mode=manual wait_ms=1280",
        },
        {
            "action_name": "solve_captcha",
            "action_result": "CAPTCHA_FILL_FAILED target_input_not_found wait_ms=300",
        },
        {"action_name": "click", "action_result": "Clicked [1]"},
    ]

    metrics = summarize_captcha_metrics_from_history(history)

    assert metrics["captcha_action_count"] == 3
    assert metrics["captcha_ok_count"] == 1
    assert metrics["captcha_manual_empty_count"] == 1
    assert metrics["captcha_fill_failed_count"] == 1
    assert metrics["captcha_manual_wait_ms_total"] == 1580


def test_runtime_watchdog_records_captcha_failure_breakdown() -> None:
    watchdog = RuntimeWatchdog(
        RuntimeWatchdogConfig(
            enabled=True,
            max_captcha_attempts=10,
            max_captcha_failures=2,
        )
    )
    decision = watchdog.record_action_result(
        action_type="solve_captcha",
        result_text="CAPTCHA_MANUAL_EMPTY mode=manual wait_ms=1200",
    )
    assert decision.should_stop is False

    decision = watchdog.record_action_result(
        action_type="solve_captcha",
        result_text="CAPTCHA_FILL_FAILED target_input_not_found",
    )
    assert decision.should_stop is True
    assert decision.code == "captcha_failure_limit"
    breakdown = decision.evidence.get("failure_breakdown", {})
    assert isinstance(breakdown, dict)
    assert breakdown.get("manual_empty") == 1
    assert breakdown.get("fill_failed") == 1


def test_intervention_need_adds_captcha_manual_and_looping_tasks() -> None:
    tasks = evaluate_intervention_need(
        session_id="session:obs:1",
        source_url="https://demo.local/login",
        page_type="login",
        low_layout_confidence_hits=0,
        failed_action_count=0,
        semantic_conflict_count=0,
        has_cross_origin=False,
        has_iframe=False,
        has_captcha=True,
        captcha_action_count=4,
        captcha_empty_code_count=2,
        captcha_manual_empty_count=1,
        captcha_fill_failed_count=0,
    )
    reasons = {task.reason for task in tasks}
    assert "captcha_manual_required" in reasons
    assert "captcha_looping" in reasons

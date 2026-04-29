from __future__ import annotations

from graph_agent.cartography.llm_planning import (
    PAGE_TYPE_ACTION_POLICY,
    build_login_hint_from_env,
)
from graph_agent.cartography.captcha import _normalize_captcha_code
from graph_agent.cartography.captcha import _needs_arithmetic_retry


def test_login_policy_mentions_captcha_handling():
    login_policy = PAGE_TYPE_ACTION_POLICY["login"]
    assert "captcha" in login_policy.lower()
    assert "fill the captcha field" in login_policy.lower()


def test_build_login_hint_includes_captcha_instruction(monkeypatch):
    monkeypatch.setenv("MAPPING_USERNAME", "demo_user")
    monkeypatch.setenv("MAPPING_PASSWORD", "demo_pass")
    hint = build_login_hint_from_env()
    assert "username=demo_user" in hint
    assert "password=demo_pass" in hint
    assert "验证码" in hint


def test_normalize_captcha_code_solves_arithmetic_expression():
    assert _normalize_captcha_code("9+8=?") == "17"
    assert _normalize_captcha_code("9*3=?") == "27"


def test_needs_arithmetic_retry_when_operator_missing_but_question_pattern_exists():
    assert _needs_arithmetic_retry("10=?") is True
    assert _needs_arithmetic_retry("7+1=?") is False
    assert _needs_arithmetic_retry("ABCD") is False

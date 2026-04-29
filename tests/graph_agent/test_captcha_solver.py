from __future__ import annotations

from graph_agent.cartography.llm_planning import (
    PAGE_TYPE_ACTION_POLICY,
    build_login_hint_from_env,
)


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

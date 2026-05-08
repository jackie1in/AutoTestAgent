from __future__ import annotations

from graph_agent.cartography.llm_planning import (
    LLM_ZONE_TYPE_ALIASES,
    PAGE_TYPE_ACTION_POLICY,
    PAGE_TYPE_ALIASES,
    LLMFunctionalZone,
    LLMPageAnalysis,
    build_exploration_guidance,
    build_login_hint_from_env,
    normalize_llm_zone_type,
    normalize_page_type,
)
from graph_agent.cartography.mapping_pipeline import map_llm_zone_type
from graph_agent.cartography.captcha import _normalize_captcha_code
from graph_agent.cartography.captcha import _needs_arithmetic_retry
from graph_agent.cartography.captcha import _should_keep_img_candidate
from graph_agent.cartography.captcha import _score_captcha_img_node
from graph_agent.cartography.captcha import normalize_manual_captcha_code
from graph_agent.cartography.captcha import resolve_captcha_solve_mode


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


def test_login_guidance_skips_zone_order_when_disabled():
    guidance = build_exploration_guidance(
        LLMPageAnalysis(
            page_type="login",
            functional_zones=[
                LLMFunctionalZone(
                    zone_type="form",
                    selector="form#login",
                    description="login form",
                )
            ],
        ),
        include_zone_order=False,
    )
    assert "ACTION POLICY: Login page" in guidance
    assert "ZONE ORDER" not in guidance


def test_normalize_page_type_handles_alias_and_unknown():
    assert normalize_page_type("Sign-In") == "login"
    assert normalize_page_type("HOME") == "welcome"
    assert normalize_page_type("mystery_page") == "unknown"
    assert PAGE_TYPE_ALIASES["signin"] == "login"


def test_zone_type_filter_aliases_are_consistent():
    assert normalize_llm_zone_type("filter") == "filter"
    assert normalize_llm_zone_type("filter-panel") == "filter"
    assert normalize_llm_zone_type("filter_panel") == "filter"
    assert map_llm_zone_type("filter") is not None
    assert map_llm_zone_type("filter-panel") == map_llm_zone_type("filter")
    assert map_llm_zone_type("filter_panel") == map_llm_zone_type("filter")
    assert "filter" in LLM_ZONE_TYPE_ALIASES


def test_map_llm_zone_type_unknown_returns_none():
    assert map_llm_zone_type("totally_new_zone") is None


def test_normalize_captcha_code_solves_arithmetic_expression():
    assert _normalize_captcha_code("9+8=?") == "17"
    assert _normalize_captcha_code("9*3=?") == "27"


def test_needs_arithmetic_retry_when_operator_missing_but_question_pattern_exists():
    assert _needs_arithmetic_retry("10=?") is True
    assert _needs_arithmetic_retry("7+1=?") is False
    assert _needs_arithmetic_retry("ABCD") is False


def test_score_captcha_img_node_prefers_title_and_same_form():
    input_xpaths = ["/html/body/form[1]/div[2]/input[3]"]
    form_xpaths = ["/html/body/form[1]"]
    captcha_score = _score_captcha_img_node(
        attrs={
            "id": "verifyImg",
            "title": "请输入验证码",
            "alt": "",
            "class": "captcha-img",
            "src": "data:image/png;base64,AAA",
        },
        node_value="",
        img_xpath="/html/body/form[1]/div[2]/img[1]",
        captcha_id_norm="",
        input_xpaths=input_xpaths,
        form_xpaths=form_xpaths,
    )
    wrong_score = _score_captcha_img_node(
        attrs={
            "id": "logoCode",
            "title": "company logo",
            "alt": "logo code",
            "class": "header-logo",
            "src": "data:image/png;base64,BBB",
        },
        node_value="",
        img_xpath="/html/body/header/div[1]/img[1]",
        captcha_id_norm="",
        input_xpaths=input_xpaths,
        form_xpaths=form_xpaths,
    )
    assert captcha_score > wrong_score


def test_score_captcha_img_node_prefers_exact_captcha_id():
    score_with_id = _score_captcha_img_node(
        attrs={
            "id": "captchaImg",
            "title": "",
            "alt": "",
            "class": "",
            "src": "https://example.com/captcha",
        },
        node_value="",
        img_xpath="/html/body/div/img[1]",
        captcha_id_norm="captchaimg",
        input_xpaths=[],
        form_xpaths=[],
    )
    score_without_id = _score_captcha_img_node(
        attrs={
            "id": "randomImg",
            "title": "",
            "alt": "",
            "class": "",
            "src": "https://example.com/captcha",
        },
        node_value="",
        img_xpath="/html/body/div/img[2]",
        captcha_id_norm="captchaimg",
        input_xpaths=[],
        form_xpaths=[],
    )
    assert score_with_id > score_without_id


def test_should_keep_img_candidate_restricts_to_same_form_when_form_exists():
    assert (
        _should_keep_img_candidate(
            img_xpath="/html/body/form[1]/div/img[1]",
            form_xpaths=["/html/body/form[1]"],
            login_anchor_xpaths=["/html/body/form[1]/input[2]"],
        )
        is True
    )
    assert (
        _should_keep_img_candidate(
            img_xpath="/html/body/header/img[1]",
            form_xpaths=["/html/body/form[1]"],
            login_anchor_xpaths=["/html/body/form[1]/input[2]"],
        )
        is False
    )


def test_should_keep_img_candidate_uses_xpath_proximity_without_form():
    # common prefix depth >= 4 -> keep
    assert (
        _should_keep_img_candidate(
            img_xpath="/html/body/div[2]/section[1]/img[1]",
            form_xpaths=[],
            login_anchor_xpaths=["/html/body/div[2]/section[1]/input[1]"],
        )
        is True
    )
    # common prefix depth < 4 -> drop
    assert (
        _should_keep_img_candidate(
            img_xpath="/html/body/header/img[1]",
            form_xpaths=[],
            login_anchor_xpaths=["/html/body/div[2]/section[1]/input[1]"],
        )
        is False
    )


def test_resolve_captcha_solve_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("CAPTCHA_SOLVE_MODE", raising=False)
    assert resolve_captcha_solve_mode() == "auto"
    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "MANUAL")
    assert resolve_captcha_solve_mode() == "manual"
    monkeypatch.setenv("CAPTCHA_SOLVE_MODE", "invalid")
    assert resolve_captcha_solve_mode() == "auto"


def test_normalize_manual_captcha_code_strips_noise():
    assert normalize_manual_captcha_code("  A1 b-2  ") == "A1b2"
    assert normalize_manual_captcha_code("`unknown`") == ""

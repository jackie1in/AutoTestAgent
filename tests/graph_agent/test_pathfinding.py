"""Tests for pathfinding (T6: empty intent tolerance)."""

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import (
    ActionType,
    Intent,
    GraphEdge,
)


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


def _make_edge(
    source: str,
    target: str,
    selector: str = "",
    action: ActionType = ActionType.CLICK,
    intent: Intent | None = None,
    step_index: int | None = None,
    edge_id: str | None = None,
    intent_failure_reason: str | None = None,
) -> GraphEdge:
    return GraphEdge(
        source=source,
        target=target,
        selector=selector,
        action=action,
        intent=intent,
        step_index=step_index,
        edge_id=edge_id,
        intent_failure_reason=intent_failure_reason,
    )


def test_pathfinding_skips_null_intent_for_matching():
    """Edges with intent=None are skipped for matching but traversable."""
    intent = _make_intent("Click login", key="submit_login")
    edges = [
        _make_edge("a", "b", selector="#nav", intent=None, intent_failure_reason="parse failed"),
        _make_edge("b", "c", selector="#login", intent=intent),
    ]

    path = get_path_from_intent("submit_login", edges)
    assert len(path) == 2
    assert path[0].source == "a" and path[0].target == "b"
    assert path[0].intent is None
    assert path[0].selector == "#nav"
    assert path[1].source == "b" and path[1].target == "c"
    assert path[1].intent is not None
    assert path[1].intent.key == "submit_login"


def test_pathfinding_all_null_intent_returns_empty():
    """When all edges have null intent, returns [] without crashing."""
    edges = [
        _make_edge("a", "b", selector="#btn", intent=None, intent_failure_reason="LLM failed"),
    ]

    path = get_path_from_intent("login", edges)
    assert path == []


def test_pathfinding_mixed_intents_returns_valid_path():
    """Path with mix of null and non-null intents is valid for playback."""
    intent = _make_intent("Fill username", key="fill_username")
    edges = [
        _make_edge("a", "b", selector="#x", intent=None, intent_failure_reason="x"),
        _make_edge("b", "c", selector="#user", action=ActionType.FILL, intent=intent),
    ]

    path = get_path_from_intent("fill_username", edges)
    assert len(path) == 2
    for edge in path:
        assert edge.selector
        assert edge.action
        assert edge.source and edge.target


def test_pathfinding_no_crash_on_null_intent_access():
    """No crash when iterating edges with intent=None."""
    edges = [
        _make_edge("a", "b", selector="#btn", intent=None),
    ]

    path = get_path_from_intent("anything", edges)
    assert path == []


def test_pathfinding_skips_low_confidence_for_matching():
    """Low-confidence intent edge should not be selected as match target."""
    low_conf_intent = Intent(
        raw="click login",
        verb="Click",
        object="Login",
        summary="Click login button",
        key="submit_login",
        confidence=0.2,
    )
    edges = [
        _make_edge("a", "b", selector="#login", intent=low_conf_intent),
    ]
    path = get_path_from_intent("submit_login", edges)
    assert path == []


def test_pathfinding_matches_dot_key_query():
    """Should match modern dot-separated intent key directly."""
    intent = _make_intent("Fill username", key="auth.fill.username")
    edges = [
        _make_edge("a", "b", selector="#username", action=ActionType.FILL, intent=intent),
    ]
    path = get_path_from_intent("auth.fill.username", edges)
    assert len(path) == 1
    assert path[0].intent is not None
    assert path[0].intent.key == "auth.fill.username"


def test_pathfinding_matches_chinese_login_query_to_dot_key():
    """Chinese login query should match auth.submit.login style keys."""
    intent = _make_intent("Submit login form", key="auth.submit.login")
    edges = [
        _make_edge("a", "b", selector="#login", intent=intent),
    ]
    path = get_path_from_intent("登录", edges)
    assert len(path) == 1
    assert path[0].intent is not None
    assert path[0].intent.key == "auth.submit.login"


def test_pathfinding_supports_self_loop_intent_match():
    """Self-loop edge with matching intent should be returned for playback."""
    intent = _make_intent("Accept terms", key="legal.accept.terms")
    edges = [
        _make_edge("a", "a", selector="#accept", intent=intent),
    ]

    path = get_path_from_intent("legal.accept.terms", edges)
    assert len(path) == 1
    assert path[0].source == "a"
    assert path[0].target == "a"
    assert path[0].selector == "#accept"


def test_pathfinding_keeps_self_loop_prerequisite_before_target():
    """When a self-loop is a prerequisite, it should remain before downstream match."""
    prereq = _make_intent("Accept terms", key="legal.accept.terms")
    target = _make_intent("Start workflow", key="workflow.start.blind_rating")
    edges = [
        _make_edge("a", "a", selector="#agree", intent=prereq, step_index=1),
        _make_edge("a", "b", selector="#enter", intent=target, step_index=2),
    ]

    path = get_path_from_intent("workflow.start.blind_rating", edges)
    assert len(path) == 2
    assert path[0].source == "a" and path[0].target == "a"
    assert path[0].selector == "#agree"
    assert path[1].source == "a" and path[1].target == "b"
    assert path[1].selector == "#enter"


def test_pathfinding_submit_login_query_prefers_submit_edge_over_go_to_login():
    """Specific submit_login query should not collapse to a generic go_to_login edge."""
    edges = [
        _make_edge("home", "login", selector='a[href="/login"]', intent=_make_intent("Go to login page", key="go_to_login")),
        _make_edge("login", "user", selector="#username", action=ActionType.FILL, intent=_make_intent("Fill username", key="fill_username"), step_index=1),
        _make_edge("user", "pass", selector="#password", action=ActionType.FILL, intent=_make_intent("Fill password", key="fill_password"), step_index=2),
        _make_edge("pass", "secure", selector='button[type="submit"]', intent=_make_intent("Submit login form", key="submit_login"), step_index=3),
    ]

    path = get_path_from_intent("submit_login", edges)

    assert len(path) >= 2
    assert path[-1].selector == 'button[type="submit"]'
    assert path[-1].intent is not None
    assert path[-1].intent.key == "submit_login"


def test_pathfinding_atomic_intent_matches_navigation_key():
    """Direct key match for navigation intent without business template aggregation."""
    edges = [
        _make_edge("login", "user", selector="#username", action=ActionType.FILL, intent=_make_intent("Fill username", key="auth.fill.username"), step_index=1),
        _make_edge("user", "pass", selector="#password", action=ActionType.FILL, intent=_make_intent("Fill password", key="auth.fill.password"), step_index=2),
        _make_edge("pass", "secure", selector="button[type='submit']", intent=_make_intent("Submit login", key="auth.submit.login"), step_index=3),
        _make_edge("secure", "module", selector='a[href="/add_remove_elements/"]', intent=_make_intent("Navigate to module", key="elements.navigation.select"), step_index=4),
    ]

    # "elements.navigation.select" matches by exact key
    path = get_path_from_intent("elements.navigation.select", edges)
    assert len(path) == 4
    assert path[-1].target == "module"
    assert path[-1].intent is not None
    assert path[-1].intent.key == "elements.navigation.select"

    # "navigate" matches by summary substring
    path = get_path_from_intent("navigate", edges)
    assert len(path) == 4
    assert path[-1].target == "module"


def test_pathfinding_long_query_falls_back_to_single_intent_match():
    """Without business templates, compound queries like '登录后进入目标模块' won't aggregate.
    They should match a single intent or return empty."""
    edges = [
        _make_edge("login", "user", selector="#username", action=ActionType.FILL, intent=_make_intent("Fill username", key="auth.fill.username"), step_index=1),
        _make_edge("user", "pass", selector="#password", action=ActionType.FILL, intent=_make_intent("Fill password", key="auth.fill.password"), step_index=2),
        _make_edge("pass", "secure", selector="button[type='submit']", intent=_make_intent("Submit login", key="auth.submit.login"), step_index=3),
        _make_edge("secure", "module", selector='a[href="/add_remove_elements/"]', intent=_make_intent("Navigate to module", key="elements.navigation.select"), step_index=4),
    ]

    # Compound query without business template aggregation returns empty
    path = get_path_from_intent("登录后进入目标模块", edges)
    # This is expected: without business template, compound query cannot be resolved
    # The query contains both login and module terms, but no single intent matches both
    assert path == []

    # But individual queries still work
    path = get_path_from_intent("登录", edges)
    assert len(path) == 3
    assert path[-1].target == "secure"

    # "module" matches by summary substring
    path = get_path_from_intent("module", edges)
    assert len(path) == 4
    assert path[-1].target == "module"

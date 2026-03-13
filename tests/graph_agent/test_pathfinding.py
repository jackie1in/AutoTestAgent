"""Tests for pathfinding (T6: empty intent tolerance)."""

import networkx as nx

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    Intent,
)


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


def test_pathfinding_skips_null_intent_for_matching():
    """Edges with intent=None are skipped for matching but traversable."""
    intent = _make_intent("Click login", key="submit_login")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    # a -> b has null intent, b -> c has submit_login
    G.add_edge(
        "a",
        "b",
        selector="#nav",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="parse failed",
    )
    G.add_edge(
        "b",
        "c",
        selector="#login",
        action=ActionType.CLICK,
        intent=intent,
        intent_failure_reason=None,
    )

    path = get_path_from_intent("submit_login", G)
    assert len(path) == 2
    assert path[0].source == "a" and path[0].target == "b"
    assert path[0].intent is None
    assert path[0].selector == "#nav"
    assert path[1].source == "b" and path[1].target == "c"
    assert path[1].intent is not None
    assert path[1].intent.key == "submit_login"


def test_pathfinding_all_null_intent_returns_empty():
    """When all edges have null intent, returns [] without crashing."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a",
        "b",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM failed",
    )

    path = get_path_from_intent("login", G)
    assert path == []


def test_pathfinding_mixed_intents_returns_valid_path():
    """Path with mix of null and non-null intents is valid for playback."""
    intent = _make_intent("Fill username", key="fill_username")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge(
        "a",
        "b",
        selector="#x",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="x",
    )
    G.add_edge(
        "b",
        "c",
        selector="#user",
        action=ActionType.FILL,
        intent=intent,
        intent_failure_reason=None,
    )

    path = get_path_from_intent("fill_username", G)
    assert len(path) == 2
    for edge in path:
        assert edge.selector
        assert edge.action
        assert edge.source and edge.target


def test_pathfinding_no_crash_on_null_intent_access():
    """No crash when iterating edges with intent=None."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#btn", action=ActionType.CLICK, intent=None)

    path = get_path_from_intent("anything", G)
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
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a", "b", selector="#login", action=ActionType.CLICK, intent=low_conf_intent
    )
    path = get_path_from_intent("submit_login", G)
    assert path == []


def test_pathfinding_matches_dot_key_query():
    """Should match modern dot-separated intent key directly."""
    intent = _make_intent("Fill username", key="auth.fill.username")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#username", action=ActionType.FILL, intent=intent)
    path = get_path_from_intent("auth.fill.username", G)
    assert len(path) == 1
    assert path[0].intent is not None
    assert path[0].intent.key == "auth.fill.username"


def test_pathfinding_matches_chinese_login_query_to_dot_key():
    """Chinese login query should match auth.submit.login style keys."""
    intent = _make_intent("Submit login form", key="auth.submit.login")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#login", action=ActionType.CLICK, intent=intent)
    path = get_path_from_intent("登录", G)
    assert len(path) == 1
    assert path[0].intent is not None
    assert path[0].intent.key == "auth.submit.login"


def test_pathfinding_supports_self_loop_intent_match():
    """Self-loop edge with matching intent should be returned for playback."""
    intent = _make_intent("Accept terms", key="legal.accept.terms")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_edge("a", "a", selector="#accept", action=ActionType.CLICK, intent=intent)

    path = get_path_from_intent("legal.accept.terms", G)
    assert len(path) == 1
    assert path[0].source == "a"
    assert path[0].target == "a"
    assert path[0].selector == "#accept"


def test_pathfinding_keeps_self_loop_prerequisite_before_target():
    """When a self-loop is a prerequisite, it should remain before downstream match."""
    prereq = _make_intent("Accept terms", key="legal.accept.terms")
    target = _make_intent("Start workflow", key="workflow.start.blind_rating")
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a",
        "a",
        selector="#agree",
        action=ActionType.CLICK,
        intent=prereq,
        step_index=1,
    )
    G.add_edge(
        "a",
        "b",
        selector="#enter",
        action=ActionType.CLICK,
        intent=target,
        step_index=2,
    )

    path = get_path_from_intent("workflow.start.blind_rating", G)
    assert len(path) == 2
    assert path[0].source == "a" and path[0].target == "a"
    assert path[0].selector == "#agree"
    assert path[1].source == "a" and path[1].target == "b"
    assert path[1].selector == "#enter"


def test_pathfinding_prefers_business_template_over_atomic_intent():
    """Template match for login should beat a shorter atomic submit edge."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("user", url="https://a.com/login")
    G.add_node("pass", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("shortcut", url="https://a.com/secure")
    G.add_edge(
        "login",
        "user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "user",
        "pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login form", key="auth.submit.login"),
    )
    G.add_edge(
        "login",
        "shortcut",
        key="step-direct",
        edge_id="step-direct",
        step_index=4,
        selector="#submit-shortcut",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login form", key="auth.submit.login"),
    )
    template = BusinessTemplate(
        template_id="tpl-auth-login",
        business_key="auth.login",
        summary="用户登录流程",
        entry_node="login",
        exit_node="secure",
        path_length=3,
        confidence=0.95,
        steps=[
            BusinessTemplateStep(
                edge_id="step-1",
                source="login",
                target="user",
                selector="#username",
                action=ActionType.FILL,
                intent_key="auth.fill.username",
                param_name="username",
            ),
            BusinessTemplateStep(
                edge_id="step-2",
                source="user",
                target="pass",
                selector="#password",
                action=ActionType.FILL,
                intent_key="auth.fill.password",
                param_name="password",
            ),
            BusinessTemplateStep(
                edge_id="step-3",
                source="pass",
                target="secure",
                selector="button[type='submit']",
                action=ActionType.CLICK,
                intent_key="auth.submit.login",
                param_name=None,
            ),
        ],
        slots={"username": 0, "password": 1, "submit": 2},
        evidence={
            "intent_keys": [
                "auth.fill.username",
                "auth.fill.password",
                "auth.submit.login",
            ]
        },
    )
    G.graph["business_templates"] = [template.model_dump(mode="json")]

    path = get_path_from_intent("登录", G)

    assert len(path) == 3
    assert [edge.edge_id for edge in path] == ["step-1", "step-2", "step-3"]


def test_pathfinding_falls_back_to_atomic_intent_when_template_misses():
    """Atomic intent resolution should still work when no template matches the query."""
    G = nx.MultiDiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://a.com/secure")
    G.add_edge(
        "a",
        "b",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#logout",
        action=ActionType.CLICK,
        intent=_make_intent("Logout", key="auth.logout"),
    )
    G.graph["business_templates"] = [
        BusinessTemplate(
            template_id="tpl-auth-login",
            business_key="auth.login",
            summary="用户登录流程",
            entry_node="a",
            exit_node="b",
            path_length=1,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-x",
                    source="a",
                    target="b",
                    selector="#submit",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                )
            ],
            slots={"submit": 0},
            evidence={},
        ).model_dump(mode="json")
    ]

    path = get_path_from_intent("logout", G)

    assert len(path) == 1
    assert path[0].selector == "#logout"
    assert path[0].intent is not None
    assert path[0].intent.key == "auth.logout"


def test_pathfinding_expands_template_dependencies_before_business_steps():
    """Business template replay should prepend its explicit prerequisites."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("dashboard", url="https://a.com/dashboard")
    G.add_edge(
        "login",
        "secure",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#login",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="auth.submit.login"),
    )
    G.add_edge(
        "secure",
        "dashboard",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#project",
        action=ActionType.CLICK,
        intent=_make_intent("Open dashboard", key="project.dashboard.open"),
    )
    G.graph["business_templates"] = [
        BusinessTemplate(
            template_id="tpl-auth-login",
            business_key="auth.login",
            summary="用户登录流程",
            entry_node="login",
            exit_node="secure",
            path_length=1,
            confidence=0.95,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-1",
                    source="login",
                    target="secure",
                    selector="#login",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                )
            ],
            slots={"submit": 0},
            evidence={},
        ).model_dump(mode="json"),
        BusinessTemplate(
            template_id="tpl-project-dashboard",
            business_key="project.dashboard.open",
            summary="打开项目看板",
            entry_node="secure",
            exit_node="dashboard",
            path_length=1,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-2",
                    source="secure",
                    target="dashboard",
                    selector="#project",
                    action=ActionType.CLICK,
                    intent_key="project.dashboard.open",
                    param_name=None,
                )
            ],
            slots={},
            evidence={},
            depends_on=["auth.login"],
        ).model_dump(mode="json"),
    ]

    path = get_path_from_intent("project.dashboard.open", G)

    assert [edge.edge_id for edge in path] == ["step-1", "step-2"]


def test_pathfinding_submit_login_query_prefers_submit_edge_over_go_to_login():
    """Specific submit_login query should not collapse to a generic go_to_login edge."""
    G = nx.DiGraph()
    G.add_node("home", url="https://a.com/")
    G.add_node("login", url="https://a.com/login")
    G.add_node("user", url="https://a.com/login")
    G.add_node("pass", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_edge(
        "home",
        "login",
        selector='a[href="/login"]',
        action=ActionType.CLICK,
        intent=_make_intent("Go to login page", key="go_to_login"),
    )
    G.add_edge(
        "login",
        "user",
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="fill_username"),
        step_index=1,
    )
    G.add_edge(
        "user",
        "pass",
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="fill_password"),
        step_index=2,
    )
    G.add_edge(
        "pass",
        "secure",
        selector='button[type="submit"]',
        action=ActionType.CLICK,
        intent=_make_intent("Submit login form", key="submit_login"),
        step_index=3,
    )

    path = get_path_from_intent("submit_login", G)

    assert len(path) >= 2
    assert path[-1].selector == 'button[type="submit"]'
    assert path[-1].intent is not None
    assert path[-1].intent.key == "submit_login"


def test_pathfinding_intent_b_login_then_enter_module_resolvable():
    """意图 B: 登录后进入目标模块 - 应解析为 auth.login + navigation.module.select 路径。"""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("user", url="https://a.com/login")
    G.add_node("pass", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("module", url="https://a.com/add_remove_elements/")
    G.add_edge(
        "login",
        "user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "user",
        "pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="auth.submit.login"),
    )
    G.add_edge(
        "secure",
        "module",
        key="step-4",
        edge_id="step-4",
        step_index=4,
        selector='a[href="/add_remove_elements/"]',
        action=ActionType.CLICK,
        intent=_make_intent("Navigate to Add/Remove module", key="elements.navigation.select"),
    )
    G.graph["business_templates"] = [
        BusinessTemplate(
            template_id="tpl-auth",
            business_key="auth.login",
            summary="用户登录流程",
            entry_node="login",
            exit_node="secure",
            path_length=3,
            confidence=0.95,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-1",
                    source="login",
                    target="user",
                    selector="#username",
                    action=ActionType.FILL,
                    intent_key="auth.fill.username",
                    param_name="username",
                ),
                BusinessTemplateStep(
                    edge_id="step-2",
                    source="user",
                    target="pass",
                    selector="#password",
                    action=ActionType.FILL,
                    intent_key="auth.fill.password",
                    param_name="password",
                ),
                BusinessTemplateStep(
                    edge_id="step-3",
                    source="pass",
                    target="secure",
                    selector="button[type='submit']",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                ),
            ],
            slots={"username": 0, "password": 1, "submit": 2},
            evidence={},
        ).model_dump(mode="json"),
        BusinessTemplate(
            template_id="tpl-nav-module",
            business_key="navigation.module.select",
            summary="登录后进入目标模块",
            entry_node="secure",
            exit_node="module",
            path_length=1,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-4",
                    source="secure",
                    target="module",
                    selector='a[href="/add_remove_elements/"]',
                    action=ActionType.CLICK,
                    intent_key="elements.navigation.select",
                    param_name=None,
                )
            ],
            slots={},
            evidence={},
            depends_on=["auth.login"],
        ).model_dump(mode="json"),
    ]

    for query in [
        "登录后进入目标模块",
        "navigation.module.select",
        "进入目标模块",
        "project.navigation.report_access",
        "项目管理",
    ]:
        path = get_path_from_intent(query, G)
        assert len(path) >= 4, (
            f"意图 B: query={query!r} 应解析为至少 4 条边 (auth.login 3 + 进入模块 1)"
        )
        assert path[-1].target == "module"


def test_pathfinding_intent_c_project_list_enter_subproject_open_overview():
    """意图 C: 项目列表进入子项目并打开概览 - 应解析为 auth.login + elements.management.add 路径。"""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("user", url="https://a.com/login")
    G.add_node("pass", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("add_remove", url="https://a.com/add_remove_elements/")
    G.add_node("overview", url="https://a.com/add_remove_elements/")
    G.add_edge(
        "login",
        "user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_make_intent("Fill username", key="auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "user",
        "pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_make_intent("Fill password", key="auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="auth.submit.login"),
    )
    G.add_edge(
        "secure",
        "add_remove",
        key="step-4",
        edge_id="step-4",
        step_index=4,
        selector='a[href="/add_remove_elements/"]',
        action=ActionType.CLICK,
        intent=_make_intent("Navigate to Add/Remove", key="elements.navigation.select"),
    )
    G.add_edge(
        "add_remove",
        "overview",
        key="step-5",
        edge_id="step-5",
        step_index=5,
        selector='button[onclick="addElement()"]',
        action=ActionType.CLICK,
        intent=_make_intent("Click Add Element", key="elements.add.click"),
    )
    G.graph["business_templates"] = [
        BusinessTemplate(
            template_id="tpl-auth",
            business_key="auth.login",
            summary="用户登录",
            entry_node="login",
            exit_node="secure",
            path_length=3,
            confidence=0.95,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-1",
                    source="login",
                    target="user",
                    selector="#username",
                    action=ActionType.FILL,
                    intent_key="auth.fill.username",
                    param_name="username",
                ),
                BusinessTemplateStep(
                    edge_id="step-2",
                    source="user",
                    target="pass",
                    selector="#password",
                    action=ActionType.FILL,
                    intent_key="auth.fill.password",
                    param_name="password",
                ),
                BusinessTemplateStep(
                    edge_id="step-3",
                    source="pass",
                    target="secure",
                    selector="button[type='submit']",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                ),
            ],
            slots={},
            evidence={},
        ).model_dump(mode="json"),
        BusinessTemplate(
            template_id="tpl-management-add",
            business_key="elements.management.add",
            summary="项目列表进入子项目并打开概览",
            entry_node="secure",
            exit_node="overview",
            path_length=2,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-4",
                    source="secure",
                    target="add_remove",
                    selector='a[href="/add_remove_elements/"]',
                    action=ActionType.CLICK,
                    intent_key="elements.navigation.select",
                    param_name=None,
                ),
                BusinessTemplateStep(
                    edge_id="step-5",
                    source="add_remove",
                    target="overview",
                    selector='button[onclick="addElement()"]',
                    action=ActionType.CLICK,
                    intent_key="elements.add.click",
                    param_name=None,
                ),
            ],
            slots={},
            evidence={},
            depends_on=["auth.login"],
        ).model_dump(mode="json"),
    ]

    for query in ["项目列表进入子项目并打开概览", "elements.management.add", "项目列表进入子项目"]:
        path = get_path_from_intent(query, G)
        assert len(path) >= 5, (
            f"意图 C: query={query!r} 应解析为至少 5 条边 (auth.login 3 + 进入子项目 1 + 打开概览 1)"
        )
        assert path[-1].target == "overview"


def test_pathfinding_inserts_connecting_path_when_dependency_exit_differs_from_template_entry():
    """When dependency exit_node != template entry_node, pathfinding should insert connecting graph edges."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://a.com/login")
    G.add_node("secure", url="https://a.com/secure")
    G.add_node("bridge", url="https://a.com/bridge")
    G.add_node("target", url="https://a.com/target")
    G.add_edge(
        "login",
        "secure",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#login",
        action=ActionType.CLICK,
        intent=_make_intent("Submit login", key="auth.submit.login"),
    )
    G.add_edge(
        "secure",
        "bridge",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#bridge",
        action=ActionType.CLICK,
        intent=_make_intent("Go to bridge", key="nav.bridge"),
    )
    G.add_edge(
        "bridge",
        "target",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="#target",
        action=ActionType.CLICK,
        intent=_make_intent("Open target", key="project.target.open"),
    )
    G.graph["business_templates"] = [
        BusinessTemplate(
            template_id="tpl-auth",
            business_key="auth.login",
            summary="登录",
            entry_node="login",
            exit_node="secure",
            path_length=1,
            confidence=0.95,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-1",
                    source="login",
                    target="secure",
                    selector="#login",
                    action=ActionType.CLICK,
                    intent_key="auth.submit.login",
                    param_name=None,
                )
            ],
            slots={},
            evidence={},
        ).model_dump(mode="json"),
        BusinessTemplate(
            template_id="tpl-target",
            business_key="project.target.open",
            summary="打开目标",
            entry_node="bridge",
            exit_node="target",
            path_length=1,
            confidence=0.9,
            steps=[
                BusinessTemplateStep(
                    edge_id="step-3",
                    source="bridge",
                    target="target",
                    selector="#target",
                    action=ActionType.CLICK,
                    intent_key="project.target.open",
                    param_name=None,
                )
            ],
            slots={},
            evidence={},
            depends_on=["auth.login"],
        ).model_dump(mode="json"),
    ]

    path = get_path_from_intent("project.target.open", G)

    assert [edge.edge_id for edge in path] == ["step-1", "step-2", "step-3"]

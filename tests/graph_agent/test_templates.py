"""Tests for offline business template generation."""

from unittest.mock import AsyncMock

import pytest
import networkx as nx

from graph_agent.graph.templates import generate_business_templates  # type: ignore[import-untyped]
from graph_agent.models import ActionType, Intent


def _intent(summary: str, key: str | None, confidence: float | None = 0.9) -> Intent:
    return Intent(
        raw=summary,
        verb="Interact",
        object="Element",
        summary=summary,
        key=key,
        confidence=confidence,
    )


def _build_login_graph(include_null_prefix: bool = False) -> nx.MultiDiGraph:
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("home", url="https://example.com/")
    G.add_node("login", url="https://example.com/login")
    G.add_node("state-user", url="https://example.com/login")
    G.add_node("state-pass", url="https://example.com/login")
    G.add_node("secure", url="https://example.com/secure")
    if include_null_prefix:
        G.add_edge(
            "home",
            "login",
            key="step-0",
            edge_id="step-0",
            step_index=0,
            selector="a[href='/login']",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="parse failed",
        )
    G.add_edge(
        "login",
        "state-user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_intent("Fill username", "auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "state-user",
        "state-pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_intent("Fill password", "auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "state-pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_intent("Submit login form", "auth.submit.login"),
    )
    return G


@pytest.mark.asyncio
async def test_generate_business_templates_builds_login_template():
    """Login flow should produce one business template with replayable steps."""
    G = _build_login_graph()
    classifier = AsyncMock(
        return_value={
            "is_business_flow": True,
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "slots": {"username": 0, "password": 1, "submit": 2},
            "confidence": 0.94,
            "evidence": {"intent_keys": ["auth.fill.username", "auth.fill.password", "auth.submit.login"]},
        }
    )

    templates = await generate_business_templates(G, classifier=classifier)

    assert len(templates) == 1
    assert templates[0].business_key == "auth.login"
    assert templates[0].path_length == 3
    assert [step.edge_id for step in templates[0].steps] == ["step-1", "step-2", "step-3"]


@pytest.mark.asyncio
async def test_generate_business_templates_requires_minimum_path_length():
    """Single-edge flows should not become business templates."""
    G = nx.MultiDiGraph()
    G.add_node("login", url="https://example.com/login")
    G.add_node("secure", url="https://example.com/secure")
    G.add_edge(
        "login",
        "secure",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_intent("Submit login form", "auth.submit.login"),
    )
    classifier = AsyncMock()

    templates = await generate_business_templates(G, classifier=classifier)

    assert templates == []
    classifier.assert_not_called()


@pytest.mark.asyncio
async def test_generate_business_templates_keeps_null_intent_prefix():
    """A null-intent prefix edge may remain in the chosen template path."""
    G = _build_login_graph(include_null_prefix=True)
    classifier = AsyncMock(
        return_value={
            "is_business_flow": True,
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "slots": {"username": 1, "password": 2, "submit": 3},
            "confidence": 0.9,
            "evidence": {"selectors": ["a[href='/login']", "#username", "#password", "button[type='submit']"]},
        }
    )

    templates = await generate_business_templates(G, classifier=classifier)

    assert len(templates) == 1
    assert [step.edge_id for step in templates[0].steps] == ["step-0", "step-1", "step-2", "step-3"]


@pytest.mark.asyncio
async def test_generate_business_templates_skips_low_confidence_terminal_intent():
    """Low-confidence terminal intents should not trigger template classification."""
    G = _build_login_graph()
    data = next(iter(G.get_edge_data("state-pass", "secure").values()))
    data["intent"] = _intent("Submit login form", "auth.submit.login", confidence=0.2)
    classifier = AsyncMock()

    templates = await generate_business_templates(G, classifier=classifier)

    assert templates == []
    classifier.assert_not_called()


@pytest.mark.asyncio
async def test_generate_business_templates_dedupes_same_terminal_flow():
    """Overlapping candidates ending at the same terminal edge should dedupe to one template."""
    G = _build_login_graph(include_null_prefix=True)
    classifier = AsyncMock(
        side_effect=[
            {
                "is_business_flow": True,
                "business_key": "auth.login",
                "summary": "用户登录流程",
                "slots": {"username": 0, "password": 1, "submit": 2},
                "confidence": 0.88,
                "evidence": {},
            },
            {
                "is_business_flow": True,
                "business_key": "auth.login",
                "summary": "用户登录流程",
                "slots": {"username": 1, "password": 2, "submit": 3},
                "confidence": 0.88,
                "evidence": {},
            },
        ]
    )

    templates = await generate_business_templates(G, classifier=classifier)

    assert len(templates) == 1
    assert templates[0].steps[-1].edge_id == "step-3"
    assert len(templates[0].steps) == 4


@pytest.mark.asyncio
async def test_generate_business_templates_uses_node_url_metadata_in_payload():
    """Classifier payload should use node url metadata, not opaque state ids, for URLs."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("state-login-empty", url="https://example.com/login")
    G.add_node("state-user", url="https://example.com/login")
    G.add_node("state-pass", url="https://example.com/login")
    G.add_node("state-secure", url="https://example.com/secure")
    G.add_edge(
        "state-login-empty",
        "state-user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_intent("Fill username", "auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "state-user",
        "state-pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_intent("Fill password", "auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "state-pass",
        "state-secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_intent("Submit login form", "auth.submit.login"),
    )
    classifier = AsyncMock(
        return_value={
            "is_business_flow": True,
            "business_key": "auth.login",
            "summary": "用户登录流程",
            "slots": {"username": 0, "password": 1, "submit": 2},
            "confidence": 0.9,
            "evidence": {},
        }
    )

    await generate_business_templates(G, classifier=classifier)

    payload = classifier.await_args.args[0]
    assert payload["source_url"] == "https://example.com/login"
    assert payload["target_url"] == "https://example.com/secure"


@pytest.mark.asyncio
async def test_generate_business_templates_accepts_pure_click_business_flow():
    """Pure click paths with clear business progression should still be classified."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("projects", url="https://example.com/projects")
    G.add_node("project-detail", url="https://example.com/projects/42")
    G.add_node("project-tasks", url="https://example.com/projects/42/tasks")
    G.add_edge(
        "projects",
        "project-detail",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="a[data-project='42']",
        action=ActionType.CLICK,
        intent=_intent("Open project detail", "project.detail.open"),
    )
    G.add_edge(
        "project-detail",
        "project-tasks",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="a[href='/projects/42/tasks']",
        action=ActionType.CLICK,
        intent=_intent("Open project tasks", "project.task.view"),
    )
    classifier = AsyncMock(
        return_value={
            "is_business_flow": True,
            "business_key": "project.task.view",
            "summary": "查看项目任务",
            "slots": {},
            "confidence": 0.9,
            "evidence": {"intent_keys": ["project.detail.open", "project.task.view"]},
        }
    )

    templates = await generate_business_templates(G, classifier=classifier)

    assert len(templates) == 1
    assert templates[0].business_key == "project.task.view"
    assert [step.edge_id for step in templates[0].steps] == ["step-1", "step-2"]
    classifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_business_templates_rejects_navigation_only_click_noise():
    """Navigation-only click chains should not be sent to the classifier."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("dashboard-overview", url="https://example.com/dashboard")
    G.add_node("dashboard-milestones", url="https://example.com/dashboard")
    G.add_node("dashboard-reports", url="https://example.com/dashboard")
    G.add_edge(
        "dashboard-overview",
        "dashboard-milestones",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="li[data-tab='milestones']",
        action=ActionType.CLICK,
        intent=_intent("Open milestones tab", "project.navigation.select_milestone_tab"),
    )
    G.add_edge(
        "dashboard-milestones",
        "dashboard-reports",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="li[data-tab='reports']",
        action=ActionType.CLICK,
        intent=_intent("Open reports tab", "project.navigation.select_tab"),
    )
    classifier = AsyncMock(
        return_value={
            "is_business_flow": True,
            "business_key": "project.navigation.tabs",
            "summary": "切换项目标签",
            "slots": {},
            "confidence": 0.8,
            "evidence": {},
        }
    )

    templates = await generate_business_templates(G, classifier=classifier)

    assert templates == []
    classifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_business_templates_attaches_auth_login_dependency():
    """Post-login business templates should depend on auth.login when reachable from login exit."""
    G = _build_login_graph()
    G.add_node("task-form", url="https://example.com/secure")
    G.add_node("task-saved", url="https://example.com/secure")
    G.add_edge(
        "secure",
        "task-form",
        key="step-4",
        edge_id="step-4",
        step_index=4,
        selector="#task-name",
        action=ActionType.FILL,
        intent=_intent("Fill task name", "project.task.fill.name"),
        param_name="taskName",
    )
    G.add_edge(
        "task-form",
        "task-saved",
        key="step-5",
        edge_id="step-5",
        step_index=5,
        selector="#save-task",
        action=ActionType.CLICK,
        intent=_intent("Save task", "project.task.save"),
    )
    async def classifier(payload: dict[str, object]) -> dict[str, object] | None:
        edge_ids = [step.get("edge_id") for step in payload["steps"]]  # type: ignore[index]
        if edge_ids == ["step-1", "step-2", "step-3"]:
            return {
                "is_business_flow": True,
                "business_key": "auth.login",
                "summary": "用户登录流程",
                "slots": {"username": 0, "password": 1, "submit": 2},
                "confidence": 0.95,
                "evidence": {},
            }
        if edge_ids == ["step-4", "step-5"]:
            return {
                "is_business_flow": True,
                "business_key": "project.task.save",
                "summary": "保存任务",
                "slots": {"name": 0, "submit": 1},
                "confidence": 0.9,
                "evidence": {},
            }
        return {
            "is_business_flow": False,
            "business_key": "",
            "summary": "",
            "slots": {},
            "confidence": 0.0,
            "evidence": {},
        }

    templates = await generate_business_templates(
        G,
        classifier=classifier,
        max_path_length=3,
    )
    by_key = {template.business_key: template for template in templates}

    assert by_key["project.task.save"].depends_on == ["auth.login"]


@pytest.mark.asyncio
async def test_generate_business_templates_prefers_direct_business_dependency_before_auth():
    """Later business templates should point to their direct predecessor, not only auth.login."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("login", url="https://example.com/login")
    G.add_node("login-user", url="https://example.com/login")
    G.add_node("login-pass", url="https://example.com/login")
    G.add_node("secure", url="https://example.com/secure")
    G.add_node("dashboard", url="https://example.com/secure")
    G.add_node("sidebar-open", url="https://example.com/secure")
    G.add_node("create-form", url="https://example.com/secure")
    G.add_node("form-filled", url="https://example.com/secure")
    G.add_node("saved", url="https://example.com/secure")
    G.add_edge(
        "login",
        "login-user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_intent("Fill username", "auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "login-user",
        "login-pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_intent("Fill password", "auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "login-pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_intent("Submit login form", "auth.submit.login"),
    )
    G.add_edge(
        "dashboard",
        "sidebar-open",
        key="step-4",
        edge_id="step-4",
        step_index=4,
        selector="#open-sidebar",
        action=ActionType.CLICK,
        intent=_intent("Open sidebar", "project.action.button"),
    )
    G.add_edge(
        "sidebar-open",
        "create-form",
        key="step-5",
        edge_id="step-5",
        step_index=5,
        selector="a[data-create='item']",
        action=ActionType.CLICK,
        intent=_intent("Open create form", "project.dashboard.open_sidebar_item"),
    )
    G.add_edge(
        "create-form",
        "form-filled",
        key="step-6",
        edge_id="step-6",
        step_index=6,
        selector="#label",
        action=ActionType.FILL,
        intent=_intent("Fill label", "form.fill.input"),
        param_name="label",
    )
    G.add_edge(
        "form-filled",
        "saved",
        key="step-7",
        edge_id="step-7",
        step_index=7,
        selector="#confirm",
        action=ActionType.CLICK,
        intent=_intent("Confirm create", "project.action.confirm"),
    )

    async def classifier(payload: dict[str, object]) -> dict[str, object] | None:
        edge_ids = [step.get("edge_id") for step in payload["steps"]]  # type: ignore[index]
        if edge_ids == ["step-1", "step-2", "step-3"]:
            return {
                "is_business_flow": True,
                "business_key": "auth.login",
                "summary": "用户登录流程",
                "slots": {"username": 0, "password": 1, "submit": 2},
                "confidence": 0.95,
                "evidence": {},
            }
        if edge_ids == ["step-4", "step-5"]:
            return {
                "is_business_flow": True,
                "business_key": "navigation.sidebar.select",
                "summary": "打开创建侧边栏项",
                "slots": {"trigger": 0, "item": 1},
                "confidence": 0.9,
                "evidence": {},
            }
        if edge_ids == ["step-6", "step-7"]:
            return {
                "is_business_flow": True,
                "business_key": "project.create.submit",
                "summary": "提交创建表单",
                "slots": {"label": 0, "submit": 1},
                "confidence": 0.9,
                "evidence": {},
            }
        return {
            "is_business_flow": False,
            "business_key": "",
            "summary": "",
            "slots": {},
            "confidence": 0.0,
            "evidence": {},
        }

    templates = await generate_business_templates(G, classifier=classifier, max_path_length=4)
    by_key = {template.business_key: template for template in templates}

    assert by_key["navigation.sidebar.select"].depends_on == ["auth.login"]
    assert by_key["project.create.submit"].depends_on == ["navigation.sidebar.select"]


@pytest.mark.asyncio
async def test_generate_business_templates_attaches_auth_login_when_no_direct_template_exists():
    """Non-auth templates should still fall back to auth.login when no direct predecessor template exists."""
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node("login", url="https://example.com/login")
    G.add_node("login-user", url="https://example.com/login")
    G.add_node("login-pass", url="https://example.com/login")
    G.add_node("secure", url="https://example.com/secure")
    G.add_node("dashboard", url="https://example.com/secure")
    G.add_node("project-form", url="https://example.com/secure")
    G.add_node("project-saved", url="https://example.com/secure")
    G.add_edge(
        "login",
        "login-user",
        key="step-1",
        edge_id="step-1",
        step_index=1,
        selector="#username",
        action=ActionType.FILL,
        intent=_intent("Fill username", "auth.fill.username"),
        param_name="username",
    )
    G.add_edge(
        "login-user",
        "login-pass",
        key="step-2",
        edge_id="step-2",
        step_index=2,
        selector="#password",
        action=ActionType.FILL,
        intent=_intent("Fill password", "auth.fill.password"),
        param_name="password",
    )
    G.add_edge(
        "login-pass",
        "secure",
        key="step-3",
        edge_id="step-3",
        step_index=3,
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent=_intent("Submit login form", "auth.submit.login"),
    )
    G.add_edge(
        "project-form",
        "project-saved",
        key="step-4",
        edge_id="step-4",
        step_index=4,
        selector="#save-project",
        action=ActionType.CLICK,
        intent=_intent("Save project", "project.save.submit"),
    )
    G.add_edge(
        "dashboard",
        "project-form",
        key="step-5",
        edge_id="step-5",
        step_index=5,
        selector="#open-project-form",
        action=ActionType.CLICK,
        intent=_intent("Open project form", "project.form.open"),
    )

    async def classifier(payload: dict[str, object]) -> dict[str, object] | None:
        edge_ids = [step.get("edge_id") for step in payload["steps"]]  # type: ignore[index]
        if edge_ids == ["step-1", "step-2", "step-3"]:
            return {
                "is_business_flow": True,
                "business_key": "auth.login",
                "summary": "用户登录流程",
                "slots": {"username": 0, "password": 1, "submit": 2},
                "confidence": 0.95,
                "evidence": {},
            }
        if edge_ids == ["step-5", "step-4"]:
            return {
                "is_business_flow": True,
                "business_key": "project.create.submit",
                "summary": "打开并提交项目表单",
                "slots": {"open": 0, "submit": 1},
                "confidence": 0.9,
                "evidence": {},
            }
        return {
            "is_business_flow": False,
            "business_key": "",
            "summary": "",
            "slots": {},
            "confidence": 0.0,
            "evidence": {},
        }

    templates = await generate_business_templates(G, classifier=classifier, max_path_length=3)
    by_key = {template.business_key: template for template in templates}

    assert by_key["project.create.submit"].depends_on == ["auth.login"]

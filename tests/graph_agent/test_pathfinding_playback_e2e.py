"""E2E test: pathfinding -> playback (PRD 6.2 联动验收)."""

import os

import pytest
import networkx as nx

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    Intent,
)
from graph_agent.playback.engine import run_playback

# data URL with inline HTML - loads without network, has #btn and #user elements
_DATA_HTML = (
    "data:text/html,"
    "<html><body>"
    "<button id='btn'>click</button>"
    "<input id='user' type='text' placeholder='username'/>"
    "</body></html>"
)


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


@pytest.mark.asyncio
async def test_pathfinding_to_playback_e2e():
    """
    PRD 6.2: pathfinding -> playback 端到端测试至少 1 条通过。
    图含空意图边，pathfinding 返回路径后，playback 可成功执行。
    """
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        # Build graph: a -> b (null intent) -> c (fill_username intent)
        intent = _make_intent("Fill username", key="fill_username")
        G = nx.DiGraph()
        G.add_node("a", url="https://a.com")
        G.add_node("b", url="https://b.com")
        G.add_node("c", url="https://c.com")
        G.add_edge(
            "a",
            "b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="parse failed",
        )
        G.add_edge(
            "b",
            "c",
            selector="#user",
            action=ActionType.FILL,
            intent=intent,
            intent_failure_reason=None,
            param_name="username",
        )

        # Pathfinding: get path for fill_username (traverses through null-intent a->b)
        path = get_path_from_intent("fill_username", G)
        assert len(path) == 2
        assert path[0].intent is None
        assert path[0].selector == "#btn"
        assert path[1].intent is not None
        assert path[1].selector == "#user"

        # Playback: execute path on data HTML
        result = await run_playback(
            path,
            test_data={"username": "testuser"},
            start_url=_DATA_HTML,
        )

        assert result["success"] is True
        assert result["error"] is None
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_template_query_to_playback_e2e():
    """Template-backed login query should expand to a multi-step playback path."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        G = nx.MultiDiGraph()
        G.add_node("login-empty", url="https://example.com/login")
        G.add_node("login-user", url="https://example.com/login")
        G.add_node("login-password", url="https://example.com/login")
        G.add_edge(
            "login-empty",
            "login-user",
            key="step-1",
            edge_id="step-1",
            selector="#btn",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="parse failed",
        )
        G.add_edge(
            "login-user",
            "login-password",
            key="step-2",
            edge_id="step-2",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            intent_failure_reason=None,
            param_name="username",
        )
        G.graph["business_templates"] = [
            BusinessTemplate(
                template_id="tpl-auth-login",
                business_key="auth.login",
                summary="用户登录流程",
                entry_node="login-empty",
                exit_node="login-password",
                path_length=2,
                confidence=0.9,
                steps=[
                    BusinessTemplateStep(
                        edge_id="step-1",
                        source="login-empty",
                        target="login-user",
                        selector="#btn",
                        action=ActionType.CLICK,
                        intent_key=None,
                        param_name=None,
                    ),
                    BusinessTemplateStep(
                        edge_id="step-2",
                        source="login-user",
                        target="login-password",
                        selector="#user",
                        action=ActionType.FILL,
                        intent_key="auth.fill.username",
                        param_name="username",
                    ),
                ],
                slots={"username": 1},
                evidence={"intent_keys": ["auth.fill.username"]},
            ).model_dump(mode="json")
        ]

        path = get_path_from_intent("登录", G)
        assert len(path) == 2
        assert [edge.edge_id for edge in path] == ["step-1", "step-2"]

        result = await run_playback(
            path,
            test_data={"username": "testuser"},
            start_url=_DATA_HTML,
        )

        assert result["success"] is True
        assert result["error"] is None
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_template_dependency_query_to_playback_e2e():
    """Dependent business template query should replay prerequisite login steps first."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    dependency_html = (
        "data:text/html,"
        "<html><body>"
        "<button id='login'>login</button>"
        "<button id='project'>project</button>"
        "</body></html>"
    )
    try:
        G = nx.MultiDiGraph()
        G.add_node("login", url="https://example.com/login")
        G.add_node("secure", url="https://example.com/secure")
        G.add_node("dashboard", url="https://example.com/dashboard")
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

        result = await run_playback(
            path,
            test_data={},
            start_url=dependency_html,
        )

        assert result["success"] is True
        assert result["error"] is None
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)

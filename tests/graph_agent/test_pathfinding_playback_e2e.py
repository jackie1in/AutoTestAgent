"""E2E test: pathfinding -> playback (PRD 6.2 联动验收)."""

import os

import pytest

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import (
    ActionType,
    Intent,
    GraphEdge,
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
        # Build edge list: a -> b (null intent) -> c (fill_username intent)
        edges = [
            GraphEdge(
                source="a",
                target="b",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="parse failed",
            ),
            GraphEdge(
                source="b",
                target="c",
                selector="#user",
                action=ActionType.FILL,
                intent=_make_intent("Fill username", key="fill_username"),
                intent_failure_reason=None,
                param_name="username",
            ),
        ]

        # Pathfinding: get path for fill_username (traverses through null-intent a->b)
        path = get_path_from_intent("fill_username", edges)
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
async def test_atomic_intent_query_to_playback_e2e():
    """Atomic intent query should match directly without Business Template expansion."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        edges = [
            GraphEdge(
                source="login-empty",
                target="login-user",
                edge_id="step-1",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="parse failed",
            ),
            GraphEdge(
                source="login-user",
                target="login-password",
                edge_id="step-2",
                selector="#user",
                action=ActionType.FILL,
                intent=_make_intent("Fill username", key="auth.fill.username"),
                intent_failure_reason=None,
                param_name="username",
            ),
        ]

        path = get_path_from_intent("auth.fill.username", edges)
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
async def test_prerequisite_intent_query_to_playback_e2e():
    """Prerequisite self-loop intent should be included in playback path."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    dependency_html = (
        "data:text/html,"
        "<html><body>"
        "<button id='login'>login</button>"
        "<button id='project'>project</button>"
        "</body></html>"
    )
    try:
        edges = [
            GraphEdge(
                source="login",
                target="secure",
                edge_id="step-1",
                step_index=1,
                selector="#login",
                action=ActionType.CLICK,
                intent=_make_intent("Submit login", key="auth.submit.login"),
            ),
            GraphEdge(
                source="secure",
                target="dashboard",
                edge_id="step-2",
                step_index=2,
                selector="#project",
                action=ActionType.CLICK,
                intent=_make_intent("Open dashboard", key="project.dashboard.open"),
            ),
        ]

        path = get_path_from_intent("project.dashboard.open", edges)
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

"""E2E test: pathfinding -> playback (PRD 6.2 联动验收)."""

import os

import pytest
import networkx as nx

from graph_agent.graph.pathfinding import get_path_from_intent
from graph_agent.models import ActionType, Intent
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
            "a", "b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="parse failed",
        )
        G.add_edge(
            "b", "c",
            selector="#user",
            action=ActionType.FILL,
            intent=intent,
            intent_failure_reason=None,
            data_key="username",
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

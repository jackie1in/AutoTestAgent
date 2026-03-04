"""Tests for playback engine (T7: null intent compatibility)."""

import os

import pytest

from graph_agent.models import ActionType, GraphEdge, Intent
from graph_agent.playback.engine import run_playback


def _make_intent(summary: str, key: str | None = None) -> Intent:
    return Intent(
        raw=summary,
        verb="Click",
        object="Button",
        summary=summary,
        key=key,
    )


# data URL with inline HTML - loads without network, has #btn and #user elements
_DATA_HTML = (
    "data:text/html,"
    "<html><body>"
    "<button id='btn'>click</button>"
    "<input id='user' type='text' placeholder='username'/>"
    "</body></html>"
)


@pytest.mark.asyncio
async def test_playback_null_intent_log_not_crash():
    """含空意图边的路径回放不因日志字段报错 (T7)."""
    # Ensure headless for CI
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        edge_list = [
            GraphEdge(
                source="a",
                target="b",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="parse failed",
            ),
        ]
        logs = []

        async def capture_log(entry):
            logs.append(entry)

        result = await run_playback(
            edge_list,
            test_data={},
            start_url=_DATA_HTML,
            log_callback=capture_log,
        )

        assert result["success"] is True
        assert len(logs) == 1
        assert logs[0]["intent"] == ""
        assert logs[0]["selector"] == "#btn"
        assert logs[0]["action"] == ActionType.CLICK
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


@pytest.mark.asyncio
async def test_playback_mixed_intents_log_correct():
    """混合空意图与有效意图边时，日志字段正确 (T7)."""
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        intent = _make_intent("Fill username", key="fill_username")
        edge_list = [
            GraphEdge(
                source="a",
                target="b",
                selector="#btn",
                action=ActionType.CLICK,
                intent=None,
                intent_failure_reason="x",
            ),
            GraphEdge(
                source="b",
                target="c",
                selector="#user",
                action=ActionType.FILL,
                intent=intent,
                data_key="username",
            ),
        ]
        logs = []

        async def capture_log(entry):
            logs.append(entry)

        result = await run_playback(
            edge_list,
            test_data={"username": "testuser"},
            start_url=_DATA_HTML,
            log_callback=capture_log,
        )

        assert result["success"] is True
        assert len(logs) == 2
        assert logs[0]["intent"] == ""
        assert logs[0]["selector"] == "#btn"
        assert logs[1]["intent"] == "Fill username"
        assert logs[1]["selector"] == "#user"
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)

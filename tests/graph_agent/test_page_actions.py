"""Unit tests for custom exploration actions registered via @tools.action().

Tests verify that create_explorer_tools() registers the correct 6 custom actions
into a browser-use Tools instance.  Handler-level execution is tested via
integration with the matching pipeline.
"""

from graph_agent.cartography.react_explorer.page_actions import create_explorer_tools


def test_create_explorer_tools_registers_6_actions():
    """create_explorer_tools() returns Tools with exactly 6 custom actions."""
    tools = create_explorer_tools()
    registry = tools.registry

    expected = {
        "scroll_horizontally",
        "close_overlay",
        "query_knowledge",
        "discover_zones",
        "extract_menu",
        "solve_captcha",
    }

    # All 6 custom actions must be registered
    for name in expected:
        assert name in registry.registry.actions, f"custom action '{name}' missing"

    # browser-use built-ins must be present
    for builtin in ("click", "input", "done"):
        assert builtin in registry.registry.actions, f"built-in '{builtin}' missing"

    # Total actions: built-ins (minus 8 excluded) + 6 custom
    total = len(registry.registry.actions)
    assert 16 <= total <= 30, f"expected 16-30 actions, got {total}"

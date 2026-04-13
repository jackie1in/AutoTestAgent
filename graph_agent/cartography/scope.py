from __future__ import annotations

from graph_agent.lib.scope_manager import ExplorationScope

SCOPE_MENU_DISCOVERY = ExplorationScope(
    id="scope:menu-discovery",
    name="菜单发现",
    include_selectors=["nav", ".sidebar", ".menu", "[role='navigation']", ".ant-menu"],
    exclude_selectors=["main", ".content", "form", ".modal"],
    allow_navigation=True,
    allow_iframe=False,
)

SCOPE_PAGE_EXPLORATION = ExplorationScope(
    id="scope:page-exploration",
    name="页面内探索",
    include_selectors=["main", ".content", "[role='main']", ".ant-layout-content"],
    exclude_selectors=["nav", ".sidebar", ".menu", "[role='navigation']"],
    allow_navigation=False,
    allow_iframe=True,
)


def zone_scope(zone_selector: str) -> ExplorationScope:
    """Create a scope limited to a specific zone."""
    return ExplorationScope(
        id=f"scope:zone:{zone_selector}",
        name=f"Zone探索: {zone_selector}",
        include_selectors=[zone_selector],
        exclude_selectors=[],
        allow_navigation=False,
        allow_iframe=True,
    )

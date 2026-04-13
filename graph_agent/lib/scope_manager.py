"""Scope management for browser-use pages.

Applies CSS-attribute based scope constraints to limit exploration to a
specific region of the page.  All evaluate() calls go through
browser-use Page API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.actor.page import Page


@dataclass
class ExplorationScope:
    id: str
    name: str
    include_selectors: list[str] = field(default_factory=list)
    exclude_selectors: list[str] = field(default_factory=list)
    allow_navigation: bool = False
    allow_iframe: bool = True
    max_iframe_depth: int = 3


_APPLY_SCOPE_JS = """
(excludeSelectors) => {
    document.querySelectorAll('[data-scope-excluded]').forEach(el => {
        el.removeAttribute('data-scope-excluded');
    });

    const interactiveSel = 'a, button, input, select, textarea, ' +
        '[role="button"], [role="tab"], [tabindex], [onclick]';

    for (const sel of excludeSelectors) {
        document.querySelectorAll(sel).forEach(container => {
            container.querySelectorAll(interactiveSel).forEach(el => {
                el.setAttribute('data-scope-excluded', '');
            });
        });
    }

    document.querySelectorAll(
        '.ant-modal, .el-dialog, [role="dialog"]'
    ).forEach(modal => {
        modal.querySelectorAll('[data-scope-excluded]').forEach(el => {
            el.removeAttribute('data-scope-excluded');
        });
    });
}
"""

_CLEAR_SCOPE_JS = """
() => {
    document.querySelectorAll('[data-scope-excluded]').forEach(el => {
        el.removeAttribute('data-scope-excluded');
    });
}
"""


class ScopeManager:
    """Applies and clears exploration scopes on browser-use pages."""

    async def apply_scope(self, page: "Page", scope: ExplorationScope) -> None:
        await page.evaluate(_APPLY_SCOPE_JS, scope.exclude_selectors)

    async def clear_scope(self, page: "Page") -> None:
        await page.evaluate(_CLEAR_SCOPE_JS)

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.actor.page import Page


async def capture_dom_fingerprint(page: "Page") -> str:
    """Capture a DOM structure fingerprint for state deduplication."""
    html = await page.evaluate(
        """() => {
            function simplify(el, depth) {
                if (depth > 5) return '';
                const tag = el.tagName?.toLowerCase() || '';
                const children = Array.from(el.children)
                    .map(c => simplify(c, depth + 1))
                    .filter(Boolean)
                    .join('');
                return '<' + tag + '>' + children;
            }
            return simplify(document.body, 0);
        }"""
    )
    return hashlib.sha256(html.encode()).hexdigest()[:16]


async def capture_page_snapshot(page: "Page") -> dict[str, str]:
    """Capture a full page snapshot for diff comparison."""
    url = await page.get_url()
    title = await page.get_title()
    return {
        "url": url,
        "title": title,
        "fingerprint": await capture_dom_fingerprint(page),
    }


async def capture_composite_fingerprint(
    page: "Page",
    *,
    layout_enabled: bool = True,
    layout_limit: int = 200,
) -> str:
    """Optional composite fingerprint: dom + layout."""
    dom_fp = await capture_dom_fingerprint(page)
    if not layout_enabled:
        return dom_fp
    try:
        from graph_agent.cartography.layout_snapshot import (
            capture_layout_snapshot,
            compute_layout_fingerprint,
        )

        layout_snapshot = await capture_layout_snapshot(page, limit=layout_limit)
        layout_fp = compute_layout_fingerprint(layout_snapshot)
        if not layout_fp:
            return dom_fp
        return hashlib.sha256(f"{dom_fp}:{layout_fp}".encode()).hexdigest()[:16]
    except Exception:
        return dom_fp

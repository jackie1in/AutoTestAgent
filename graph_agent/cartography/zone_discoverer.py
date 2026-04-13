from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.browser.session import BrowserSession

from graph_agent.models import Zone, ZoneType

logger = logging.getLogger(__name__)

_DISCOVER_ZONES_JS = """() => {
    const zones = [];
    const seen = new Set();

    const zonePatterns = [
        { selector: 'form', type: 'detail_form' },
        { selector: 'table', type: 'data_table' },
        { selector: '.ant-form', type: 'detail_form' },
        { selector: '.ant-table-wrapper', type: 'data_table' },
        { selector: '.ant-table', type: 'data_table' },
        { selector: '.el-form', type: 'detail_form' },
        { selector: '.el-table', type: 'data_table' },
        { selector: '.ant-tabs', type: 'tab_panel' },
        { selector: '.el-tabs', type: 'tab_panel' },
        { selector: '.ant-tree', type: 'tree_panel' },
        { selector: '.el-tree', type: 'tree_panel' },
        { selector: '.ant-modal-content', type: 'modal' },
        { selector: '.el-dialog', type: 'modal' },
        { selector: '[role="search"]', type: 'search_form' },
        { selector: '[role="toolbar"]', type: 'action_bar' },
        { selector: '[role="tabpanel"]', type: 'tab_panel' },
        { selector: '[role="tree"]', type: 'tree_panel' },
        { selector: '[role="grid"]', type: 'data_table' },
    ];

    for (const pattern of zonePatterns) {
        document.querySelectorAll(pattern.selector).forEach((el, idx) => {
            const rect = el.getBoundingClientRect();
            if (rect.width < 50 || rect.height < 30) return;

            const key = pattern.type + ':' + Math.round(rect.left) + ',' + Math.round(rect.top);
            if (seen.has(key)) return;
            seen.add(key);

            const interactiveCount = el.querySelectorAll(
                'input, select, textarea, button, a, [role="button"], '
                + '.ant-select, .ant-input, .ant-btn, .el-input, .el-button, .el-select'
            ).length;
            if (interactiveCount === 0) return;

            const summary = Array.from(el.querySelectorAll(
                'label, th, .ant-form-item-label, .el-form-item__label, '
                + '.ant-tabs-tab, legend, caption, h3, h4'
            ))
                .map(l => l.textContent?.trim())
                .filter(Boolean)
                .slice(0, 8)
                .join(', ');

            zones.push({
                selector: pattern.selector + ':nth-of-type(' + (idx + 1) + ')',
                type: pattern.type,
                interactive_count: interactiveCount,
                summary: summary || pattern.type,
                width: Math.round(rect.width),
                height: Math.round(rect.height),
            });
        });
    }

    if (zones.length === 0) {
        const containers = document.querySelectorAll(
            'main, [role="main"], .content, .page-content, '
            + '.ant-layout-content, .el-main, section, article, aside'
        );
        containers.forEach((el, idx) => {
            const ic = el.querySelectorAll(
                'input, select, textarea, button, a[href], [role="button"]'
            ).length;
            if (ic >= 3) {
                const rect = el.getBoundingClientRect();
                const tag = el.tagName.toLowerCase();
                zones.push({
                    selector: tag + ':nth-of-type(' + (idx + 1) + ')',
                    type: 'detail_form',
                    interactive_count: ic,
                    summary: 'auto-detected container',
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                });
            }
        });
    }

    return zones;
}"""


class ZoneDiscoverer:
    """Discovers functional zones within a page, including child iframes."""

    async def discover(
        self, page: "Page", state_id: str, session: "BrowserSession | None" = None
    ) -> list[Zone]:
        raw_str = await page.evaluate(_DISCOVER_ZONES_JS)
        raw_zones: list[dict[str, Any]] = (
            json.loads(raw_str) if isinstance(raw_str, str) else raw_str
        )

        if not raw_zones and session:
            for bu_page in await session.get_pages():
                try:
                    frame_raw = await bu_page.evaluate(_DISCOVER_ZONES_JS)
                    frame_zones = (
                        json.loads(frame_raw) if isinstance(frame_raw, str) else frame_raw
                    )
                    if frame_zones:
                        raw_zones = frame_zones
                        page_url = await bu_page.get_url()
                        logger.info(
                            "Found %d zones in iframe: %s",
                            len(raw_zones),
                            page_url[:80],
                        )
                        break
                except Exception:
                    continue

        zones: list[Zone] = []
        for i, rz in enumerate(raw_zones):
            zone = Zone(
                id=f"zone:{state_id}:{rz['type']}-{i}",
                zone_type=ZoneType(rz["type"]),
                root_selector=rz["selector"],
                summary=rz.get("summary", ""),
                interactive_count=rz.get("interactive_count", 0),
            )
            zones.append(zone)

        logger.info(
            "Discovered %d zones in %s (types: %s)",
            len(zones),
            state_id,
            ", ".join(z.zone_type.value for z in zones) if zones else "none",
        )
        return zones

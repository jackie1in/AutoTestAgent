"""OverlayHandler — phase-based modal/dialog/drawer detection and dismissal."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.actor.page import Page
    from browser_use.browser.session import BrowserSession
    from browser_use.llm.base import BaseChatModel
    from graph_agent.lib.page_controller import PageController

logger = logging.getLogger(__name__)


class OverlayHandler:
    """Phase-based overlay detection and dismissal.

    Five phases:
      1. JS identify overlay element + try close buttons INSIDE it
      2. Escape key
      3. JS verify overlay gone; if still present → backdrop click
      4. LLM fallback with overlay-only DOM content
      5. CDP coordinate click (last resort)
    """

    def __init__(
        self,
        browser: "BrowserSession | None",
        llm: "BaseChatModel",
        controller: "PageController | None",
    ):
        self._browser = browser
        self._llm = llm
        self._controller = controller

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def close_overlay(self) -> str:
        browser = self._browser
        if browser is None:
            return "Close overlay failed: no browser"
        page = await browser.get_current_page()
        if page is None:
            return "Close overlay failed: no active page"

        # ── Phase 1: Find overlay; try close buttons scoped to it ──────
        overlay_info = await self._detect_overlay(page)
        if not overlay_info.get("found"):
            return "No overlay detected on page"

        overlay_tag = overlay_info.get("tag", "div")
        overlay_selector = overlay_info.get("selector", overlay_tag)
        logger.debug(
            "Overlay detected: tag=%s selector=%s z=%s",
            overlay_tag, overlay_selector, overlay_info.get("z_index", 0),
        )

        closed = await self._click_overlay_close_button(page, overlay_selector)
        if closed > 0:
            return (
                f"Closed {closed} overlay(s) via close button inside "
                f"{overlay_selector}"
            )

        # ── Phase 2: Escape key ────────────────────────────────────────
        await page.evaluate(
            "() => document.dispatchEvent(new KeyboardEvent('keydown', "
            "{key:'Escape', keyCode:27, bubbles:true}))"
        )
        await asyncio.sleep(0.3)
        if not await self._is_element_visible(page, overlay_selector):
            return f"Closed overlay via Escape key ({overlay_selector})"

        # ── Phase 3: Backdrop click on the overlay element ──────────────
        await self._click_overlay_backdrop(page, overlay_selector)
        await asyncio.sleep(0.3)
        if not await self._is_element_visible(page, overlay_selector):
            return f"Closed overlay via backdrop click ({overlay_selector})"

        # Update controller DOM so LLM sees post-JS state
        if self._controller is not None:
            try:
                await self._controller.update_tree()
            except Exception:
                pass

        # ── Phase 4: LLM with overlay-only content ──────────────────────
        llm_result = await self._llm_close_overlay(browser, page, overlay_selector)
        if llm_result is not None:
            return llm_result

        # ── Phase 5: CDP coordinate click (last resort) ─────────────────
        from browser_use.browser.events import ClickCoordinateEvent

        vp_w, vp_h = 1024, 768
        try:
            size = await page.evaluate("() => ({w: innerWidth, h: innerHeight})")
            size_obj = json.loads(size) if isinstance(size, str) else size
            vp_w = int(size_obj.get("w", 1024))
            vp_h = int(size_obj.get("h", 768))
        except Exception:
            pass
        cx = max(0, vp_w + 200)
        cy = max(0, vp_h + 200)
        event = browser.event_bus.dispatch(
            ClickCoordinateEvent(coordinate_x=cx, coordinate_y=cy, force=True)
        )
        await event
        await event.event_result(raise_if_any=False, raise_if_none=False)
        return f"Closed overlay via CDP backdrop click ({cx},{cy})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _detect_overlay(self, page: "Page") -> dict:
        """Find the topmost overlay element on the page.

        Searches the full DOM tree (not just body.children) so dynamically
        created overlays inside React roots / wrapper containers are found.

        Returns dict with: found (bool), selector (str), tag (str),
        z_index (int), bbox (dict with left/top/width/height).
        """
        try:
            raw = await page.evaluate(
                """() => {
                    const overlaySelectors = [
                        // Role-based (most reliable, works regardless of CSS)
                        '[role="dialog"]', '[role="alertdialog"]',
                        // Common UI frameworks
                        '.ant-modal-wrap', '.ant-modal', '.ant-drawer', '.ant-drawer-content-wrapper',
                        '.el-dialog__wrapper', '.el-drawer__wrapper',
                        '.modal', '.modal-dialog', '.modal-content',
                        '.dialog', '.drawer', '.overlay', '.popup',
                        // Generic high-z-index containers
                        '[style*="z-index"]',
                    ];

                    let best = null;
                    let bestScore = -1;
                    let bestBBox = null;

                    // Score an element: higher = more likely to be an overlay
                    const scoreElement = (el) => {
                        if (!(el instanceof HTMLElement)) return 0;
                        const style = getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') return 0;
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 20 || rect.height < 20) return 0;
                        // Must be visible in viewport (at least partially)
                        if (rect.bottom < -10 || rect.right < -10) return 0;
                        if (rect.top > innerHeight + 10 || rect.left > innerWidth + 10) return 0;

                        const coverage = (rect.width * rect.height) / (innerWidth * innerHeight);
                        // Skip tiny elements (<2% of viewport)
                        if (coverage < 0.02) return 0;

                        let score = coverage * 100; // base: coverage

                        const pos = style.position;
                        if (pos === 'fixed') score += 50;
                        else if (pos === 'absolute') score += 30;

                        const z = parseInt(style.zIndex, 10) || 0;
                        if (z > 100) score += 40;
                        else if (z > 10) score += 20;
                        else if (z > 0) score += 10;

                        const tag = (el.tagName || '').toLowerCase();
                        const role = (el.getAttribute('role') || '').toLowerCase();
                        const cls = (el.className || '').toLowerCase();
                        const id = (el.id || '').toLowerCase();

                        // Role hints
                        if (role === 'dialog' || role === 'alertdialog') score += 60;
                        // Tag hints
                        if (tag === 'dialog') score += 60;
                        // Class hints (common overlay/modal/drawer patterns)
                        const overlayKeywords = /modal|dialog|drawer|overlay|popup|popover|panel/;
                        if (overlayKeywords.test(cls)) score += 30;
                        if (overlayKeywords.test(id)) score += 30;

                        return score;
                    };

                    // Strategy 1: Check known overlay selectors first (fast path)
                    for (const sel of overlaySelectors) {
                        try {
                            const els = document.querySelectorAll(sel);
                            for (const el of els) {
                                const s = scoreElement(el);
                                if (s > bestScore) {
                                    bestScore = s;
                                    best = el;
                                    const r = el.getBoundingClientRect();
                                    bestBBox = {left: r.left, top: r.top, width: r.width, height: r.height};
                                }
                            }
                        } catch(e) {}
                    }

                    // Strategy 2: If no overlay found via selectors, scan
                    // body's deep descendants for high-z-index / fixed elements.
                    // Walk to reasonable depth only (max depth 8 from body).
                    if (bestScore < 30) {
                        const walk = (el, depth) => {
                            if (depth > 8 || !el || !el.children) return;
                            const s = scoreElement(el);
                            if (s > bestScore) {
                                bestScore = s;
                                best = el;
                                const r = el.getBoundingClientRect();
                                bestBBox = {left: r.left, top: r.top, width: r.width, height: r.height};
                            }
                            for (const child of el.children) {
                                walk(child, depth + 1);
                            }
                        };
                        walk(document.body, 0);
                    }

                    if (!best || bestScore < 15) return JSON.stringify({found: false});

                    const tag = (best.tagName || 'div').toLowerCase();
                    let selector = tag;
                    if (best.id) {
                        selector = '#' + CSS.escape(best.id);
                    } else if (best.className && typeof best.className === 'string') {
                        const cls = best.className.trim().split(/\\s+/).slice(0, 2)
                            .filter(c => c).map(c => '.' + CSS.escape(c)).join('');
                        if (cls) selector = tag + cls;
                    }
                    return JSON.stringify({
                        found: true, selector, tag,
                        z_index: bestScore, bbox: bestBBox,
                        score: bestScore
                    });
                }"""
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            result = parsed if isinstance(parsed, dict) else {"found": False}
            if result.get("found"):
                logger.debug(
                    "Overlay detected: selector=%s score=%.1f",
                    result.get("selector"), result.get("score", 0),
                )
            return result
        except Exception:
            return {"found": False}

    async def _click_overlay_close_button(
        self, page: "Page", overlay_selector: str
    ) -> int:
        """Click close buttons scoped strictly to the overlay element."""
        try:
            esc_selector = json.dumps(overlay_selector)
            result = await page.evaluate(
                """(escapedSelector) => {
                    const closeSelectors = [
                        '.ant-drawer-close', '.ant-modal-close',
                        '.el-drawer__close-btn', '.el-dialog__close',
                        '[aria-label="Close"]', '[aria-label="close"]',
                        '[data-dismiss="modal"]',
                        'button[title="Close"]', 'button[title="close"]',
                        '.modal-header .btn-close',
                        'dialog button[value="cancel"]',
                        '.close', '.dialog-close', '.drawer-close',
                        'i[class*="close"]', 'svg[class*="close"]',
                        '[class*="close-btn"]', '[class*="closeButton"]',
                    ];

                    const overlay = document.querySelector(escapedSelector);
                    if (!overlay) return 0;

                    let closed = 0;
                    for (const sel of closeSelectors) {
                        const btns = overlay.querySelectorAll(sel);
                        for (const btn of btns) {
                            const rect = btn.getBoundingClientRect();
                            // Must be visible AND inside the overlay bounds
                            if (rect.width > 0 && rect.height > 0 && btn.offsetParent !== null) {
                                btn.click();
                                closed++;
                            }
                        }
                        if (closed > 0) break;
                    }
                    return closed;
                }""",
                esc_selector,
            )
            return int(result) if result else 0
        except Exception:
            return 0

    async def _is_element_visible(self, page: "Page", selector: str) -> bool:
        """Check if an element is still present and visible on the page."""
        try:
            esc = json.dumps(selector)
            result = await page.evaluate(
                """(escapedSelector) => {
                    const el = document.querySelector(escapedSelector);
                    if (!el) return false;
                    const style = getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') return false;
                    const rect = el.getBoundingClientRect();
                    return rect.width > 2 && rect.height > 2;
                }""",
                esc,
            )
            return bool(result)
        except Exception:
            return False

    async def _click_overlay_backdrop(
        self, page: "Page", overlay_selector: str
    ) -> None:
        """Click the overlay element itself (backdrop area)."""
        try:
            esc = json.dumps(overlay_selector)
            await page.evaluate(
                "(escapedSelector) => {"
                "const el = document.querySelector(escapedSelector);"
                "if (el) el.click();"
                "}",
                esc,
            )
        except Exception:
            pass

    async def _llm_close_overlay(
        self,
        browser: "BrowserSession",
        page: "Page",
        overlay_selector: str,
    ) -> str | None:
        """LLM fallback using ONLY the overlay's inner DOM content.

        Returns result string on success, None if LLM couldn't help.
        """
        try:
            from browser_use.llm.messages import SystemMessage, UserMessage

            esc = json.dumps(overlay_selector)
            overlay_html = await page.evaluate(
                "(escapedSelector) => {"
                "const el = document.querySelector(escapedSelector);"
                "return el ? el.outerHTML.slice(0, 3000) : '';"
                "}",
                esc,
            )
            if not overlay_html or len(overlay_html) < 20:
                return None

            prompt = f"""Find the close button in this overlay HTML. The overlay is a modal/dialog/drawer.

Only consider elements that are clearly close controls:
- X buttons, "Close"/"Cancel" text, "×" symbols, close icons
- Buttons with class names containing "close", "cancel", "dismiss"
- aria-label containing "Close" or "close"

If a close button exists inside this overlay:
  {{"action": "click_selector", "selector": "<css_selector_inside_overlay>"}}
If no close button is visible:
  {{"action": "none"}}

Return ONLY valid JSON, no other text.

<overlay_html>
{overlay_html}
</overlay_html>"""

            resp = await self._llm.ainvoke(
                [
                    SystemMessage(
                        content="You find close buttons inside overlay HTML. Return ONLY valid JSON."
                    ),
                    UserMessage(content=prompt),
                ]
            )
            raw = resp.completion if hasattr(resp, "completion") else str(resp)
            from graph_agent.cartography.react_explorer.page_actions import _parse_menu_json

            plan = _parse_menu_json(raw)

            action = plan.get("action", "none")
            if action == "click_selector":
                css = plan.get("selector", "")
                if css and isinstance(css, str):
                    full_selector = f"{overlay_selector} {css}"
                    await page.evaluate(
                        """(sel) => {
                            const el = document.querySelector(sel);
                            if (el) el.click();
                        }""",
                        full_selector,
                    )
                    return f"Closed overlay via LLM-scoped click: {full_selector}"
            return None
        except Exception as e:
            logger.warning("LLM overlay close fallback failed: %s", e)
            return None

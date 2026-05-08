"""PageController — DOM extraction and element interaction via browser-use.

Aligned with page-agent's PageController architecture:
- W3C pointer event simulation for click
- Contenteditable multi-strategy input
- Horizontal scroll support
- React/AntD patches for better DOM extraction
- Element text map for richer action results

All operations go through browser-use's BrowserSession / DomService / Page / Element APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.dom.views import EnhancedDOMTreeNode

from graph_agent.lib.types import ActionResult, BrowserState, PageInfo

logger = logging.getLogger(__name__)
DOMSelectorMap = dict[int, "EnhancedDOMTreeNode"]

# ---------------------------------------------------------------------------
# JS snippets — ported from page-agent/packages/page-controller/src/
# ---------------------------------------------------------------------------

_PAGE_INFO_JS = """() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pw = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth || 0);
    const ph = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight || 0);
    const sx = window.scrollX || window.pageXOffset || document.documentElement.scrollLeft || 0;
    const sy = window.scrollY || window.pageYOffset || document.documentElement.scrollTop || 0;
    const pb = Math.max(0, ph - (vh + sy));
    return JSON.stringify({
        viewport_width: vw, viewport_height: vh,
        page_width: pw, page_height: ph,
        scroll_x: sx, scroll_y: sy,
        pixels_above: sy, pixels_below: pb,
        pages_above: vh > 0 ? sy / vh : 0,
        pages_below: vh > 0 ? pb / vh : 0,
        total_pages: vh > 0 ? ph / vh : 0,
        current_page_position: sy / Math.max(1, ph - vh)
    });
}"""

# W3C pointer event simulation — mirrors page-agent actions.ts clickElement()
_CLICK_ELEMENT_JS = """(el) => {
    // scrollIntoView
    if (typeof el.scrollIntoViewIfNeeded === 'function') {
        el.scrollIntoViewIfNeeded(true);
    } else {
        el.scrollIntoView({behavior: 'auto', block: 'center', inline: 'nearest'});
    }

    const rect = el.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    // Hit-test: find deepest element at click coordinates
    const hitTarget = document.elementFromPoint(x, y);
    const target = (hitTarget instanceof HTMLElement && el.contains(hitTarget))
        ? hitTarget : el;

    const pointerOpts = {
        bubbles: true, cancelable: true,
        clientX: x, clientY: y, pointerType: 'mouse'
    };
    const mouseOpts = {
        bubbles: true, cancelable: true,
        clientX: x, clientY: y, button: 0
    };

    // Hover — pointer events first, then mouse events (W3C spec order)
    target.dispatchEvent(new PointerEvent('pointerover', pointerOpts));
    target.dispatchEvent(new PointerEvent('pointerenter', {...pointerOpts, bubbles: false}));
    target.dispatchEvent(new MouseEvent('mouseover', mouseOpts));
    target.dispatchEvent(new MouseEvent('mouseenter', {...mouseOpts, bubbles: false}));

    // Press
    target.dispatchEvent(new PointerEvent('pointerdown', pointerOpts));
    target.dispatchEvent(new MouseEvent('mousedown', mouseOpts));

    // Focus the original element (nearest focusable ancestor)
    el.focus({preventScroll: true});

    // Release
    target.dispatchEvent(new PointerEvent('pointerup', pointerOpts));
    target.dispatchEvent(new MouseEvent('mouseup', mouseOpts));

    // Click — activation behavior triggers via bubbling
    target.click();

    // Detect _blank links
    const isAnchor = el.tagName === 'A';
    const isBlank = isAnchor && el.target === '_blank';
    return JSON.stringify({isAnchor, isBlank});
}"""

# Contenteditable multi-strategy input — mirrors page-agent actions.ts inputTextElement()
_INPUT_TEXT_JS = """(el, text) => {
    const isContentEditable = el.isContentEditable;
    const isInput = el.tagName === 'INPUT';
    const isTextArea = el.tagName === 'TEXTAREA';

    if (isContentEditable) {
        // Plan A: Synthetic InputEvent (works for React contenteditable, Quill)
        if (el.dispatchEvent(new InputEvent('beforeinput', {
            bubbles: true, cancelable: true, inputType: 'deleteContent'
        }))) {
            el.innerText = '';
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'deleteContent'
            }));
        }
        if (el.dispatchEvent(new InputEvent('beforeinput', {
            bubbles: true, cancelable: true, inputType: 'insertText', data: text
        }))) {
            el.innerText = text;
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'insertText', data: text
            }));
        }

        // Verify Plan A
        const planAOk = el.innerText.trim() === text.trim();
        if (!planAOk) {
            // Plan B: execCommand fallback (works for Quill, Slate.js)
            el.focus();
            const doc = el.ownerDocument;
            const sel = (doc.defaultView || window).getSelection();
            const range = doc.createRange();
            range.selectNodeContents(el);
            sel.removeAllRanges();
            sel.addRange(range);
            doc.execCommand('delete', false);
            doc.execCommand('insertText', false, text);
        }
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.blur();
        return JSON.stringify({success: true, method: planAOk ? 'synthetic' : 'execCommand'});
    }

    if (isInput || isTextArea) {
        // Use native value setter for React compatibility
        const proto = Object.getPrototypeOf(el);
        const setter = Object.getOwnPropertyDescriptor(proto, 'value');
        if (setter && setter.set) {
            setter.set.call(el, text);
        } else {
            el.value = text;
        }
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.blur();
        return JSON.stringify({success: true, method: 'nativeValueSetter'});
    }

    return JSON.stringify({success: false, method: 'unsupported'});
}"""

# Smart vertical scroll — mirrors page-agent actions.ts scrollVertically()
_SCROLL_VERTICAL_JS = """(el, dy) => {
    if (el) {
        let cur = el;
        let attempts = 0;
        while (cur && attempts < 10) {
            const cs = window.getComputedStyle(cur);
            const hasScrollY = /(auto|scroll|overlay)/.test(cs.overflowY);
            const canScroll = cur.scrollHeight > cur.clientHeight;
            if (hasScrollY && canScroll) {
                const before = cur.scrollTop;
                const max = cur.scrollHeight - cur.clientHeight;
                let amt = dy / 3;
                if (amt > 0) amt = Math.min(amt, max - before);
                else amt = Math.max(amt, -before);
                cur.scrollTop = before + amt;
                const delta = cur.scrollTop - before;
                if (Math.abs(delta) > 0.5) {
                    return JSON.stringify({
                        success: true,
                        message: 'Scrolled container (' + cur.tagName + ') by ' + delta + 'px'
                    });
                }
            }
            if (cur === document.body || cur === document.documentElement) break;
            cur = cur.parentElement;
            attempts++;
        }
        return JSON.stringify({
            success: false,
            message: 'No scrollable container found for element (' + el.tagName + ')'
        });
    }

    // Page-level scroll
    const before = window.scrollY;
    const maxS = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollBy(0, dy);
    const after = window.scrollY;
    const scrolled = after - before;
    if (Math.abs(scrolled) < 1) {
        const msg = dy > 0
            ? 'Already at the bottom of the page'
            : 'Already at the top of the page';
        return JSON.stringify({success: false, message: msg});
    }
    const atEnd = (dy > 0 && after >= maxS - 1) || (dy < 0 && after <= 1);
    const edge = dy > 0 ? 'bottom' : 'top';
    const msg = atEnd
        ? 'Scrolled page by ' + scrolled + 'px. Reached the ' + edge + '.'
        : 'Scrolled page by ' + scrolled + 'px.';
    return JSON.stringify({success: true, message: msg});
}"""

# Smart horizontal scroll — mirrors page-agent actions.ts scrollHorizontally()
_SCROLL_HORIZONTAL_JS = """(el, dx) => {
    if (el) {
        let cur = el;
        let attempts = 0;
        while (cur && attempts < 10) {
            const cs = window.getComputedStyle(cur);
            const hasScrollX = /(auto|scroll|overlay)/.test(cs.overflowX);
            const canScroll = cur.scrollWidth > cur.clientWidth;
            if (hasScrollX && canScroll) {
                const before = cur.scrollLeft;
                const max = cur.scrollWidth - cur.clientWidth;
                let amt = dx / 3;
                if (amt > 0) amt = Math.min(amt, max - before);
                else amt = Math.max(amt, -before);
                cur.scrollLeft = before + amt;
                const delta = cur.scrollLeft - before;
                if (Math.abs(delta) > 0.5) {
                    return JSON.stringify({
                        success: true,
                        message: 'Scrolled container (' + cur.tagName + ') horizontally by ' + delta + 'px'
                    });
                }
            }
            if (cur === document.body || cur === document.documentElement) break;
            cur = cur.parentElement;
            attempts++;
        }
        return JSON.stringify({
            success: false,
            message: 'No horizontally scrollable container for element (' + el.tagName + ')'
        });
    }

    // Page-level scroll
    const before = window.scrollX;
    const maxS = document.documentElement.scrollWidth - window.innerWidth;
    window.scrollBy(dx, 0);
    const after = window.scrollX;
    const scrolled = after - before;
    if (Math.abs(scrolled) < 1) {
        const msg = dx > 0
            ? 'Already at the right edge of the page'
            : 'Already at the left edge of the page';
        return JSON.stringify({success: false, message: msg});
    }
    const atEnd = (dx > 0 && after >= maxS - 1) || (dx < 0 && after <= 1);
    const edge = dx > 0 ? 'right edge' : 'left edge';
    const msg = atEnd
        ? 'Scrolled page horizontally by ' + scrolled + 'px. Reached the ' + edge + '.'
        : 'Scrolled page horizontally by ' + scrolled + 'px.';
    return JSON.stringify({success: true, message: msg});
}"""

# React/AntD patches — mirrors page-agent patches/react.ts
_PATCH_REACT_JS = """() => {
    const roots = document.querySelectorAll(
        '[data-reactroot], [data-reactid], [data-react-checksum], ' +
        '#root, #app, [id^="root-"], [id^="app-"]'
    );
    roots.forEach(el => el.setAttribute('data-page-agent-not-interactive', 'true'));
}"""

_PATCH_ANTD_JS = """() => {
    const selectors = [
        '.ant-select-selector',
        '.ant-select-dropdown',
        '.ant-select-item-option',
        '.ant-picker-panel',
        '.ant-picker-dropdown',
        '.ant-cascader-menus',
        '.ant-tree-select-dropdown'
    ];
    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            if (el instanceof HTMLElement) {
                el.setAttribute('data-page-agent-not-interactive', 'true');
            }
        });
    });
}"""

_CLICK_BY_XPATH_JS = """(xpath) => {
    const byXpath = (xp) => {
        if (!xp) return null;
        try {
            return document.evaluate(
                xp,
                document,
                null,
                XPathResult.FIRST_ORDERED_NODE_TYPE,
                null
            ).singleNodeValue;
        } catch (e) {
            return null;
        }
    };
    const el = byXpath(xpath);
    if (!(el instanceof HTMLElement)) {
        return JSON.stringify({success: false, reason: 'not-found'});
    }
    try {
        el.scrollIntoView({block: 'center', inline: 'nearest'});
    } catch (e) {}
    try {
        el.click();
        return JSON.stringify({success: true, tag: (el.tagName || '').toLowerCase()});
    } catch (e) {
        return JSON.stringify({success: false, reason: String(e)});
    }
}"""

_TOP_LAYER_INFO_JS = """(el) => {
    if (!(el instanceof HTMLElement)) {
        return JSON.stringify({is_visible: false, is_top: false, reason: 'not-html'});
    }
    const rect = el.getBoundingClientRect();
    const isVisible = !!(
        rect &&
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom >= 0 &&
        rect.top <= window.innerHeight &&
        rect.right >= 0 &&
        rect.left <= window.innerWidth
    );
    if (!isVisible) {
        return JSON.stringify({
            is_visible: false,
            is_top: false,
            reason: 'out-of-viewport',
            bbox: {left: rect.left, top: rect.top, width: rect.width, height: rect.height}
        });
    }
    const margin = 5;
    const checkPoints = [
        {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2},
        {x: rect.left + margin, y: rect.top + margin},
        {x: rect.right - margin, y: rect.bottom - margin},
    ];
    let hitSource = 'document';
    const rootNode = el.getRootNode && el.getRootNode();

    const isInAncestorChain = (topEl, targetEl, stopAt) => {
        let cur = topEl;
        while (cur && cur !== stopAt) {
            if (cur === targetEl) return true;
            cur = cur.parentElement;
        }
        return false;
    };

    const checkPoint = ({x, y}) => {
        try {
            if (rootNode instanceof ShadowRoot && typeof rootNode.elementFromPoint === 'function') {
                const shadowTop = rootNode.elementFromPoint(x, y);
                if (shadowTop && isInAncestorChain(shadowTop, el, rootNode)) {
                    hitSource = 'shadow';
                    return true;
                }
            }
        } catch (e) {}
        try {
            const topEl = document.elementFromPoint(x, y);
            if (!topEl) return false;
            return isInAncestorChain(topEl, el, document.documentElement);
        } catch (e) {
            return true;
        }
    };

    const isTop = checkPoints.some(checkPoint);
    return JSON.stringify({
        is_visible: true,
        is_top: isTop,
        hit_source: hitSource,
        bbox: {left: rect.left, top: rect.top, width: rect.width, height: rect.height}
    });
}"""


class PageController:
    """Manages DOM state and element interaction entirely via browser-use.

    Architecture mirrors page-agent's PageController:
    - updateTree() → getBrowserState() → index-based actions
    - W3C pointer event simulation for click
    - Multi-strategy contenteditable input
    - Smart container-aware scrolling (vertical + horizontal)
    """

    def __init__(self, browser_session: "BrowserSession") -> None:
        self._session = browser_session
        self._selector_map: DOMSelectorMap = {}
        self._element_text_map: dict[int, str] = {}
        self._simplified_html: str = ""
        self._is_indexed: bool = False
        self._last_update_time: float = 0.0

    @property
    def simplified_html(self) -> str:
        return self._simplified_html

    @property
    def selector_map(self) -> DOMSelectorMap:
        return self._selector_map

    async def _get_page(self):
        return await self._session.must_get_current_page()

    # ── State queries ──────────────────────────────────────────

    async def get_browser_state(self) -> BrowserState:
        page = await self._get_page()
        url = await page.get_url()
        title = await page.get_title()
        pi = await self._get_page_info(page)

        await self.update_tree()
        content = self._simplified_html

        header = self._build_header(title, url, pi)
        footer = self._build_footer(pi)

        return BrowserState(
            url=url, title=title, header=header, content=content, footer=footer
        )

    async def update_tree(self) -> str:
        import time

        from browser_use.dom.service import DomService

        page = await self._get_page()

        await self._apply_dom_patches(page)

        dom_service = DomService(self._session)
        dom_state, _tree, _timing = await dom_service.get_serialized_dom_tree()
        self._simplified_html = dom_state.llm_representation()
        self._selector_map = dom_state.selector_map
        self._element_text_map = _build_element_text_map(self._selector_map)
        self._is_indexed = True
        self._last_update_time = time.time()
        return self._simplified_html

    async def get_last_update_time(self) -> float:
        return self._last_update_time

    async def _apply_dom_patches(self, page) -> None:
        """Apply lightweight DOM patches before serialized tree extraction."""
        for script in (_PATCH_REACT_JS, _PATCH_ANTD_JS):
            try:
                await page.evaluate(script)
            except Exception:
                continue

    async def extract_interactive_elements_with_top(self) -> list[dict[str, object]]:
        """Extract top-layer metadata for currently indexed interactive elements."""
        self._assert_indexed()
        out: list[dict[str, object]] = []
        for idx in sorted(k for k in self._selector_map.keys() if isinstance(k, int)):
            node = self._selector_map.get(idx)
            if node is None:
                continue
            attrs = getattr(node, "attributes", {}) or {}
            tag = str(getattr(node, "tag_name", "") or "").lower()
            text = str(getattr(node, "node_value", "") or "")
            role = str(attrs.get("role") or "").lower()
            is_menu_container = role in {"menu", "menubar", "listbox"}

            is_visible = True
            is_top = False
            hit_source = "document"
            bbox: dict[str, object] = {}
            try:
                element = await self._get_element(idx)
                if element is not None:
                    raw = await element.evaluate(_TOP_LAYER_INFO_JS)
                    parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    is_visible = bool(parsed.get("is_visible", True))
                    is_top = bool(parsed.get("is_top", False))
                    hit_source = str(parsed.get("hit_source") or "document")
                    bbox_raw = parsed.get("bbox", {})
                    if isinstance(bbox_raw, dict):
                        bbox = bbox_raw
            except Exception:
                # Keep conservative default: visible but not top.
                is_visible = True
                is_top = False

            out.append(
                {
                    "index": idx,
                    "tag": tag,
                    "text": text,
                    "attributes": attrs,
                    "is_visible": is_visible,
                    "is_top": is_top,
                    "hit_source": hit_source,
                    "is_menu_container": is_menu_container,
                    "bbox": bbox,
                }
            )
        return out

    async def render_llm_dom_with_top(self, only_top: bool = True) -> str:
        """Render simplified DOM and annotate each interactive line with is_top."""
        self._assert_indexed()
        simplified = self._simplified_html or ""
        if not simplified:
            return simplified

        top_info = await self.extract_interactive_elements_with_top()
        top_map = {int(item["index"]): bool(item.get("is_top")) for item in top_info}
        menu_map = {
            int(item["index"]): bool(item.get("is_menu_container")) for item in top_info
        }

        lines = simplified.splitlines()
        rendered: list[str] = []
        for line in lines:
            m = re.match(r"^(\*?)\[(\d+)\](.*)$", line)
            if not m:
                rendered.append(line)
                continue
            idx = int(m.group(2))
            is_top = top_map.get(idx, False)
            is_menu = menu_map.get(idx, False)
            if only_top and not is_top and not is_menu:
                continue
            suffix = " is_top=true" if is_top else " is_top=false"
            rendered.append(f"{line}{suffix}")
        return "\n".join(rendered)

    # ── Element actions ──────────────────────────────────────────

    def _assert_indexed(self) -> None:
        if not self._is_indexed:
            raise RuntimeError("DOM tree not indexed. Call update_tree() first.")

    async def _get_element(self, index: int):
        """Get a browser-use Element by index from the selector map."""
        from browser_use.actor.element import Element

        node = self._selector_map.get(index)
        if not node:
            return None
        return Element(self._session, node.backend_node_id, node.session_id)

    async def click_element(self, index: int) -> ActionResult:
        """Click element with W3C pointer event simulation.

        Mirrors page-agent: scrollIntoView → hit-test → full pointer/mouse
        event sequence → click activation.
        """
        node_xpath = ""
        try:
            node = self._selector_map.get(index)
            node_xpath = str(getattr(node, "xpath", "") or "")
        except Exception:
            node_xpath = ""
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                keys = [k for k in self._selector_map.keys() if isinstance(k, int)]
                # Try one forced tree refresh to recover from stale index maps.
                try:
                    await self.update_tree()
                except Exception:
                    pass
                refreshed = await self._get_element(index)
                refreshed_keys = [k for k in self._selector_map.keys() if isinstance(k, int)]
                if refreshed:
                    try:
                        await refreshed.click()
                        elem_text = self._element_text_map.get(index, str(index))
                        return ActionResult(
                            success=True,
                            message=f"Clicked [{index}] ({elem_text}) via refresh recovery.",
                        )
                    except Exception:
                        pass
                sample = sorted(refreshed_keys)[:8] if refreshed_keys else []
                return ActionResult(
                    success=False,
                    message=(
                        f"No element at index {index}. Current valid index range: "
                        f"[{min(refreshed_keys) if refreshed_keys else 'N/A'}, "
                        f"{max(refreshed_keys) if refreshed_keys else 'N/A'}], "
                        f"sample={sample}"
                    ),
                )

            elem_text = self._element_text_map.get(index, str(index))

            # Top-layer preflight: if occluded, try one scroll/recheck before clicking.
            top_probe_raw = await element.evaluate(_TOP_LAYER_INFO_JS)
            top_probe = (
                json.loads(top_probe_raw)
                if isinstance(top_probe_raw, str)
                else (top_probe_raw or {})
            )
            is_top = bool(top_probe.get("is_top", False))
            top_hit_source = str(top_probe.get("hit_source") or "document")
            if not is_top:
                try:
                    await element.evaluate(
                        """(el) => { try { el.scrollIntoView({block:'center', inline:'nearest'}); } catch (e) {} }"""
                    )
                    await asyncio.sleep(0.1)
                    top_probe_retry_raw = await element.evaluate(_TOP_LAYER_INFO_JS)
                    top_probe_retry = (
                        json.loads(top_probe_retry_raw)
                        if isinstance(top_probe_retry_raw, str)
                        else (top_probe_retry_raw or {})
                    )
                    is_top = bool(top_probe_retry.get("is_top", False))
                    top_hit_source = str(top_probe_retry.get("hit_source") or top_hit_source)
                except Exception:
                    pass
            logger.info(
                "[TOP_LAYER] click index=%s is_top=%s hit_source=%s",
                index,
                is_top,
                top_hit_source,
            )

            # W3C event simulation via JS injection
            raw = await element.evaluate(_CLICK_ELEMENT_JS)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})

            if result.get("isBlank"):
                return ActionResult(
                    success=True,
                    message=f"Clicked [{index}] ({elem_text}). Link opened in a new tab.",
                )
            return ActionResult(
                success=True,
                message=(
                    f"Clicked [{index}] ({elem_text})."
                    if is_top
                    else f"Clicked [{index}] ({elem_text}) with top-layer fallback."
                ),
            )
        except Exception as e:
            # Fallback to browser-use's native click
            element = await self._get_element(index)
            if element:
                try:
                    await element.click()
                    elem_text = self._element_text_map.get(index, str(index))
                    return ActionResult(
                        success=True,
                        message=f"Clicked [{index}] ({elem_text}) via CDP fallback.",
                    )
                except Exception:
                    pass
            # Try DOM-level xpath fallback when backend-node click path is stale.
            if node_xpath:
                try:
                    page = await self._get_page()
                    raw = await page.evaluate(_CLICK_BY_XPATH_JS, node_xpath)
                    result = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    if result.get("success"):
                        elem_text = self._element_text_map.get(index, str(index))
                        return ActionResult(
                            success=True,
                            message=f"Clicked [{index}] ({elem_text}) via xpath recovery.",
                        )
                except Exception:
                    pass
            return ActionResult(success=False, message=f"Click failed: {e}")

    async def input_text(self, index: int, text: str) -> ActionResult:
        """Input text with contenteditable multi-strategy support.

        Mirrors page-agent:
        - contenteditable: synthetic InputEvent → execCommand fallback
        - input/textarea: native value setter for React compatibility
        """
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                return ActionResult(success=False, message=f"No element at index {index}")

            elem_text = self._element_text_map.get(index, str(index))

            # First click to focus (mirrors page-agent: clickElement before input)
            try:
                await element.evaluate(_CLICK_ELEMENT_JS)
            except Exception:
                pass
            await asyncio.sleep(0.1)

            # Multi-strategy input via JS
            raw = await element.evaluate(_INPUT_TEXT_JS, text)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})

            if result.get("success"):
                method = result.get("method", "")
                return ActionResult(
                    success=True,
                    message=f"Input '{text}' into [{index}] ({elem_text}) via {method}.",
                )

            # Fallback to browser-use fill()
            await element.fill(text)
            return ActionResult(
                success=True,
                message=f"Input '{text}' into [{index}] ({elem_text}) via CDP fill.",
            )
        except Exception as e:
            return ActionResult(success=False, message=f"Input failed: {e}")

    async def select_option(self, index: int, option_text: str) -> ActionResult:
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                return ActionResult(success=False, message=f"No element at index {index}")
            elem_text = self._element_text_map.get(index, str(index))
            await element.select_option(option_text)
            return ActionResult(
                success=True,
                message=f"Selected '{option_text}' in [{index}] ({elem_text}).",
            )
        except Exception as e:
            return ActionResult(success=False, message=f"Select failed: {e}")

    async def scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        index: int | None = None,
    ) -> ActionResult:
        """Smart vertical scroll with container detection.

        Mirrors page-agent scrollVertically(): searches up the DOM for the
        nearest scrollable ancestor, falls back to page-level scroll.
        """
        try:
            self._assert_indexed()
            page = await self._get_page()
            delta = amount if direction == "down" else -amount

            if index is not None:
                element = await self._get_element(index)
                if element:
                    raw = await element.evaluate(_SCROLL_VERTICAL_JS, delta)
                    result = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    return ActionResult(
                        success=result.get("success", False),
                        message=result.get("message", "Scroll completed"),
                    )

            # Page-level scroll
            raw = await page.evaluate(
                f"() => {{ {_SCROLL_VERTICAL_JS.replace('(el, dy)', '(null, ' + str(delta) + ')')} }}"
            )
            # Simpler: just call window.scrollBy and report
            raw = await page.evaluate(f"""() => {{
                const before = window.scrollY;
                const maxS = document.documentElement.scrollHeight - window.innerHeight;
                window.scrollBy(0, {delta});
                const after = window.scrollY;
                const scrolled = after - before;
                if (Math.abs(scrolled) < 1) {{
                    return JSON.stringify({{success: false, message: {delta} > 0
                        ? 'Already at the bottom of the page'
                        : 'Already at the top of the page'}});
                }}
                const atEnd = ({delta} > 0 && after >= maxS - 1) || ({delta} < 0 && after <= 1);
                const edge = {delta} > 0 ? 'bottom' : 'top';
                const msg = atEnd
                    ? 'Scrolled page by ' + scrolled + 'px. Reached the ' + edge + '.'
                    : 'Scrolled page by ' + scrolled + 'px.';
                return JSON.stringify({{success: true, message: msg}});
            }}""")
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
            return ActionResult(
                success=result.get("success", True),
                message=result.get("message", f"Scrolled page {direction} {amount}px"),
            )
        except Exception as e:
            return ActionResult(success=False, message=f"Scroll failed: {e}")

    async def scroll_horizontally(
        self,
        direction: str = "right",
        amount: int = 300,
        index: int | None = None,
    ) -> ActionResult:
        """Smart horizontal scroll with container detection.

        Mirrors page-agent scrollHorizontally(): searches up the DOM for the
        nearest horizontally scrollable ancestor, falls back to page-level scroll.
        """
        try:
            self._assert_indexed()
            page = await self._get_page()
            delta = amount if direction == "right" else -amount

            if index is not None:
                element = await self._get_element(index)
                if element:
                    raw = await element.evaluate(_SCROLL_HORIZONTAL_JS, delta)
                    result = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    return ActionResult(
                        success=result.get("success", False),
                        message=result.get("message", "Scroll completed"),
                    )

            raw = await page.evaluate(f"""() => {{
                const before = window.scrollX;
                const maxS = document.documentElement.scrollWidth - window.innerWidth;
                window.scrollBy({delta}, 0);
                const after = window.scrollX;
                const scrolled = after - before;
                if (Math.abs(scrolled) < 1) {{
                    return JSON.stringify({{success: false, message: {delta} > 0
                        ? 'Already at the right edge of the page'
                        : 'Already at the left edge of the page'}});
                }}
                const atEnd = ({delta} > 0 && after >= maxS - 1) || ({delta} < 0 && after <= 1);
                const edge = {delta} > 0 ? 'right edge' : 'left edge';
                const msg = atEnd
                    ? 'Scrolled page horizontally by ' + scrolled + 'px. Reached the ' + edge + '.'
                    : 'Scrolled page horizontally by ' + scrolled + 'px.';
                return JSON.stringify({{success: true, message: msg}});
            }}""")
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
            return ActionResult(
                success=result.get("success", True),
                message=result.get("message", f"Scrolled page {direction} {amount}px"),
            )
        except Exception as e:
            return ActionResult(success=False, message=f"Scroll horizontally failed: {e}")

    async def execute_javascript(self, script: str) -> ActionResult:
        try:
            page = await self._get_page()
            result = await page.evaluate(script)
            return ActionResult(success=True, message=f"JS result: {result}")
        except Exception as e:
            return ActionResult(success=False, message=f"JS error: {e}")

    # ── Captcha helpers ─────────────────────────────────────────

    async def extract_captcha_image(self, index: int) -> ActionResult:
        """Extract captcha image bytes from an <img> element.

        Supports both base64 data URI and standalone image URLs.
        For data URI sources, returns the original `data:image/...` string directly
        to preserve MIME type and avoid redundant re-encoding.
        For standalone URLs, returns raw image bytes in the message as base64.
        """
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                return ActionResult(
                    success=False, message=f"No element at index {index}"
                )

            src = await element.evaluate("el => el.src")
            if not src or not isinstance(src, str):
                return ActionResult(
                    success=False, message="Captcha image has no src attribute"
                )
            if src.startswith("data:image"):
                return ActionResult(
                    success=True,
                    message=src,
                )

            page = await self._get_page()
            b64 = await extract_image_base64_from_src(src, page)
            return ActionResult(
                success=True,
                message=b64,
            )
        except Exception as e:
            return ActionResult(
                success=False, message=f"Captcha extraction failed: {e}"
            )

    async def refresh_captcha(self, index: int) -> ActionResult:
        """Click a captcha image to refresh it."""
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                return ActionResult(
                    success=False, message=f"No element at index {index}"
                )
            await element.click()
            await asyncio.sleep(0.5)
            return ActionResult(
                success=True, message=f"Refreshed captcha at index {index}"
            )
        except Exception as e:
            return ActionResult(
                success=False, message=f"Captcha refresh failed: {e}"
            )

    def dispose(self) -> None:
        """Clean up resources. Mirrors page-agent PageController.dispose()."""
        self._selector_map = {}
        self._element_text_map = {}
        self._simplified_html = ""
        self._is_indexed = False

    # ── Page info helpers ────────────────────────────────────────

    async def _get_page_info(self, page) -> PageInfo:
        raw_str = await page.evaluate(_PAGE_INFO_JS)
        raw = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
        return PageInfo(**raw)

    @staticmethod
    def _build_header(title: str, url: str, pi: PageInfo) -> str:
        title_line = f"Current Page: [{title}]({url})"
        page_info_line = (
            f"Page info: {pi.viewport_width}x{pi.viewport_height}px viewport, "
            f"{pi.page_width}x{pi.page_height}px total, "
            f"{pi.pages_above:.1f} pages above, {pi.pages_below:.1f} pages below, "
            f"{pi.total_pages:.1f} total pages, at {pi.current_page_position * 100:.0f}%"
        )
        elements_label = "Interactive elements from top layer of the current page:"
        has_above = pi.pixels_above > 4
        scroll_above = (
            f"... {pi.pixels_above} pixels above ({pi.pages_above:.1f} pages) - scroll to see more ..."
            if has_above
            else "[Start of page]"
        )
        return f"{title_line}\n{page_info_line}\n\n{elements_label}\n\n{scroll_above}"

    @staticmethod
    def _build_footer(pi: PageInfo) -> str:
        has_below = pi.pixels_below > 4
        return (
            f"... {pi.pixels_below} pixels below ({pi.pages_below:.1f} pages) - scroll to see more ..."
            if has_below
            else "[End of page]"
        )


# ── Module-level helpers ──────────────────────────────────────

def _build_element_text_map(selector_map: DOMSelectorMap) -> dict[int, str]:
    """Build index→text description map from the selector map.

    Mirrors page-agent getElementTextMap().
    """
    text_map: dict[int, str] = {}
    for idx, node in selector_map.items():
        text_map[idx] = _node_short_desc(node)
    return text_map


async def extract_image_base64_from_src(src: str, page) -> str:
    """Convert an image src (data URI or standalone URL) to a base64 string.

    Shared by PageController.extract_captcha_image and
    CartographyAgent._resolve_captcha_from_page to avoid duplication.

    Args:
        src: The image src attribute (data:image/... or http://...)
        page: A Playwright page object with .evaluate() available.

    Returns:
        Base64-encoded image bytes.

    Raises:
        ValueError: If src format is unsupported.
        Exception: If fetch or decode fails.
    """
    import base64

    if src.startswith("data:image"):
        # Source is already a data URI; return the base64 payload directly.
        return src.split(",", 1)[1]
    else:
        # Fetch external URL via page JS context (bypasses CORS restrictions)
        resp = await page.evaluate(
            f"""async () => {{
                const r = await fetch({json.dumps(src)});
                const buf = await r.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {{
                    binary += String.fromCharCode(bytes[i]);
                }}
                return btoa(binary);
            }}"""
        )
        img_bytes = base64.b64decode(resp)

    return base64.b64encode(img_bytes).decode()


def _node_short_desc(node: "EnhancedDOMTreeNode") -> str:
    tag = node.tag_name
    text = (node.node_value or "")[:30]
    attrs = node.attributes or {}
    label = (
        attrs.get("aria-label")
        or attrs.get("title")
        or attrs.get("placeholder")
        or attrs.get("name")
        or text
    )
    return f"<{tag}> {label}".strip()

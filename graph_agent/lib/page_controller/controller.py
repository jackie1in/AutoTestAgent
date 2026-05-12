from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession

from graph_agent.lib.types import ActionResult, BrowserState, PageInfo
from graph_agent.lib.page_controller.dom_utils import (
    DOMSelectorMap,
    _build_element_text_map,
)
from graph_agent.lib.page_controller.image_utils import extract_image_base64_from_src
from graph_agent.lib.page_controller.js_snippets import (
    _CLICK_BY_XPATH_JS,
    _CLICK_ELEMENT_JS,
    _EXTRACT_MENU_JS,
    _INPUT_TEXT_JS,
    _PAGE_INFO_JS,
    _PATCH_ANTD_JS,
    _PATCH_REACT_JS,
    _SCROLL_HORIZONTAL_JS,
    _SCROLL_VERTICAL_JS,
    _TOP_LAYER_INFO_JS,
)

logger = logging.getLogger(__name__)


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
        from browser_use.actor.element import Element

        node = self._selector_map.get(index)
        if not node:
            return None
        return Element(self._session, node.backend_node_id, node.session_id)

    async def click_element(self, index: int) -> ActionResult:
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
                [k for k in self._selector_map.keys() if isinstance(k, int)]
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
        try:
            self._assert_indexed()
            element = await self._get_element(index)
            if not element:
                return ActionResult(success=False, message=f"No element at index {index}")

            elem_text = self._element_text_map.get(index, str(index))

            try:
                await element.evaluate(_CLICK_ELEMENT_JS)
            except Exception:
                pass
            await asyncio.sleep(0.1)

            raw = await element.evaluate(_INPUT_TEXT_JS, text)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})

            if result.get("success"):
                method = result.get("method", "")
                return ActionResult(
                    success=True,
                    message=f"Input '{text}' into [{index}] ({elem_text}) via {method}.",
                )

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

    async def extract_menu_structure(self) -> str:
        """Extract hierarchical menu structure from the page.

        Returns a JSON string with nested menu items:
        {"items": [{"text": "...", "href": "...", "level": 1, "tag": "a", "children": [...]}]}
        """
        try:
            page = await self._get_page()
            raw = await page.evaluate(_EXTRACT_MENU_JS)
            return raw if isinstance(raw, str) else json.dumps(raw or {})
        except Exception as e:
            logger.warning("extract_menu_structure failed: %s", e)
            return json.dumps({"items": [], "error": str(e)})

    def dispose(self) -> None:
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

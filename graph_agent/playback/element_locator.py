"""Element relocation using multiple fallback strategies.

This module provides utilities to relocate elements when the original
selector fails, using element characteristics like text content,
attributes, and tag name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_agent.models import ElementSnapshot

if TYPE_CHECKING:
    from playwright.async_api import Page


class ElementRelocator:
    """Relocates elements using multiple fallback strategies."""

    def __init__(self, page: "Page"):
        self._page = page

    async def locate(self, snapshot: ElementSnapshot | None) -> str | None:
        """Try to locate element using multiple strategies.

        Args:
            snapshot: Element snapshot with stored characteristics

        Returns:
            Working selector or None if all strategies fail
        """
        if not snapshot:
            return None

        strategies = [
            self._try_original_selector,
            self._try_text_content,
            self._try_aria_label,
            self._try_id,
            self._try_name,
            self._try_composite,
            self._try_partial_text,
        ]

        for strategy in strategies:
            result = await strategy(snapshot)
            if result:
                return result

        return None

    async def _try_original_selector(self, snapshot: ElementSnapshot) -> str | None:
        """Try the original selector."""
        selector = snapshot.css_selector or snapshot.selector
        if not selector:
            return None

        try:
            count = await self._page.locator(selector).count()
            if count > 0:
                return selector
        except Exception:
            pass
        return None

    async def _try_text_content(self, snapshot: ElementSnapshot) -> str | None:
        """Try locating by text content."""
        text = snapshot.text_content or snapshot.inner_text
        tag = snapshot.tag_name

        if not text or len(text) < 2:  # Avoid single character matches
            return None

        # Escape quotes in text
        escaped_text = text.replace('"', '\\"')

        # Try exact match with tag
        if tag:
            selector = f'{tag}:has-text("{escaped_text}")'
            try:
                count = await self._page.locator(selector).count()
                if count == 1:  # Unique match
                    return selector
            except Exception:
                pass

        # Try without tag
        selector = f':has-text("{escaped_text}")'
        try:
            count = await self._page.locator(selector).count()
            if count == 1:
                return selector
        except Exception:
            pass

        return None

    async def _try_aria_label(self, snapshot: ElementSnapshot) -> str | None:
        """Try locating by aria-label."""
        if not snapshot.aria_label:
            return None

        escaped = snapshot.aria_label.replace('"', '\\"')
        selector = f'[aria-label="{escaped}"]'

        try:
            count = await self._page.locator(selector).count()
            if count > 0:
                return selector
        except Exception:
            pass

        return None

    async def _try_id(self, snapshot: ElementSnapshot) -> str | None:
        """Try locating by ID."""
        if not snapshot.id:
            return None

        selector = f'#{snapshot.id}'
        try:
            count = await self._page.locator(selector).count()
            if count > 0:
                return selector
        except Exception:
            pass

        return None

    async def _try_name(self, snapshot: ElementSnapshot) -> str | None:
        """Try locating by name attribute."""
        if not snapshot.name:
            return None

        escaped = snapshot.name.replace('"', '\\"')
        selector = f'[name="{escaped}"]'

        try:
            count = await self._page.locator(selector).count()
            if count > 0:
                return selector
        except Exception:
            pass

        return None

    async def _try_composite(self, snapshot: ElementSnapshot) -> str | None:
        """Try composite selector using multiple attributes."""
        tag = snapshot.tag_name or "*"
        conditions = []

        if snapshot.class_name:
            # Try first class
            first_class = snapshot.class_name.split()[0]
            conditions.append(f'.{first_class}')

        if snapshot.type:
            conditions.append(f'[type="{snapshot.type}"]')

        if not conditions:
            return None

        selector = tag + "".join(conditions)

        try:
            count = await self._page.locator(selector).count()
            if count == 1:  # Unique match
                return selector
        except Exception:
            pass

        return None

    async def _try_partial_text(self, snapshot: ElementSnapshot) -> str | None:
        """Try partial text match as last resort."""
        text = snapshot.text_content or snapshot.inner_text
        tag = snapshot.tag_name

        if not text or len(text) < 5:
            return None

        # Use first 20 chars for partial match
        partial = text[:20].replace('"', '\\"')

        if tag:
            selector = f'{tag}:has-text("{partial}")'
        else:
            selector = f':has-text("{partial}")'

        try:
            count = await self._page.locator(selector).count()
            if count >= 1:  # Accept even if multiple (best effort)
                return selector
        except Exception:
            pass

        return None


async def relocate_element(
    page: "Page",
    snapshot: ElementSnapshot | None,
    original_selector: str | None = None,
) -> str | None:
    """Convenience function to relocate an element.

    Args:
        page: Playwright page
        snapshot: Element snapshot with stored characteristics
        original_selector: Fallback original selector

    Returns:
        Working selector or None
    """
    relocator = ElementRelocator(page)

    # Try snapshot-based relocation first
    if snapshot:
        result = await relocator.locate(snapshot)
        if result:
            return result

    # Try original selector as last resort
    if original_selector:
        try:
            count = await page.locator(original_selector).count()
            if count > 0:
                return original_selector
        except Exception:
            pass

    return None

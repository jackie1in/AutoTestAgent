"""Checkpoint validation runner using browser-use Page API."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from browser_use.actor.page import Page

from graph_agent.models import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointRunner:
    """Executes checkpoint validations against the current page state."""

    async def run(self, checkpoint: Checkpoint, page: "Page") -> dict[str, Any]:
        """Execute a single checkpoint and return result.

        Returns:
            {"passed": bool, "message": str}
        """
        rule = checkpoint.get_rule()

        match checkpoint.rule_type:
            case "url_match":
                return await self._check_url(rule, page)
            case "element_exists":
                return await self._check_element(rule, page)
            case "toast":
                return await self._check_toast(rule, page)
            case _:
                return {"passed": False, "message": f"Unknown rule_type: {checkpoint.rule_type}"}

    async def _check_url(self, rule: dict[str, Any], page: "Page") -> dict[str, Any]:
        url = await page.get_url()
        expected = rule.get("expected_url_contains", "")
        passed = expected in url
        return {"passed": passed, "message": f"URL {'contains' if passed else 'missing'} {expected}"}

    async def _check_element(self, rule: dict[str, Any], page: "Page") -> dict[str, Any]:
        selector = rule.get("selector", "")
        if not selector:
            return {"passed": False, "message": "No selector in rule"}
        raw = await page.evaluate(
            """(sel) => document.querySelectorAll(sel).length""",
            selector,
        )
        count = int(raw) if raw else 0
        passed = count > 0
        return {"passed": passed, "message": f"Element {selector} {'found' if passed else 'not found'}"}

    async def _check_toast(self, rule: dict[str, Any], page: "Page") -> dict[str, Any]:
        msg_contains = rule.get("message_contains", "")
        raw = await page.evaluate(
            """(msgContains) => {
                const sels = ['.ant-message', '.el-message', "[role='alert']", '.toast'];
                for (const sel of sels) {
                    const els = document.querySelectorAll(sel);
                    for (const el of els) {
                        const text = (el.textContent || '').trim();
                        if (text && text.includes(msgContains)) {
                            return JSON.stringify({ found: true, text: text });
                        }
                    }
                }
                return JSON.stringify({ found: false });
            }""",
            msg_contains,
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        if result.get("found"):
            return {"passed": True, "message": f"Toast found: {result.get('text', '')}"}
        return {"passed": False, "message": f"Toast with '{msg_contains}' not found"}

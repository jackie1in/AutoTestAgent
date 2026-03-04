"""Scout: list all interactive elements on a page (no click), persist as inventory JSON."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from graph_agent.llm import get_llm

LOG = logging.getLogger(__name__)

SCOUT_TASK_TEMPLATE = (
    "Navigate to {url} using the navigate action, then identify ALL interactive elements on the page. "
    "Do NOT click anything. List buttons, links, forms, input fields, menus, dropdowns, and any other clickable elements. "
    "Provide a comprehensive inventory."
)

EXTRACT_PROMPT = """From the following scout report (list of interactive elements on a web page), extract a JSON array of elements.
Each item must have: "selector" (Playwright selector, e.g. #id or [name="x"] or xpath=...), "type" (one of: button, input, link, other), "label" (optional short description or null).
Output only the JSON array, no markdown or explanation.

Scout report:
---
{report}
---
"""


def _parse_elements_from_llm_response(response_text: str) -> list[dict]:
    """Parse JSON array from LLM response; return [] on failure."""
    if not (response_text or "").strip():
        return []
    text = response_text.strip()
    # Allow JSON inside code block
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return []
    try:
        raw = json.loads(match.group())
        if not isinstance(raw, list):
            return []
        elements = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sel = item.get("selector")
            if sel is None:
                continue
            elements.append({
                "selector": str(sel).strip(),
                "type": str(item.get("type", "other")).strip() or "other",
                "label": item.get("label") if item.get("label") is None else str(item.get("label")).strip() or None,
            })
        return elements
    except (json.JSONDecodeError, TypeError) as e:
        LOG.warning("Failed to parse scout LLM response as JSON: %s", e)
        return []


async def run_scout(
    url: str,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Run scout agent: navigate to url, list all interactive elements (no click), return structured inventory.

    - url: Page URL to scout.
    - output_path: If set, write inventory to JSON file { "url": url, "elements": [...] }. Parent dir is created.

    Returns list of { "selector", "type", "label" }. On Agent or LLM failure returns [].
    """
    from browser_use import Agent, Browser

    browser = Browser(headless=True)
    llm = get_llm()
    initial_actions = [{"navigate": {"url": url, "new_tab": False}}]
    task = SCOUT_TASK_TEMPLATE.format(url=url)

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        initial_actions=initial_actions,
    )

    try:
        history = await agent.run(max_steps=10)
    except Exception as e:
        LOG.warning("Scout agent run failed: %s", e)
        return []
    finally:
        if hasattr(browser, "stop"):
            await browser.stop()
        elif hasattr(browser, "close"):
            await browser.close()

    raw_report = ""
    try:
        if history and hasattr(history, "final_result"):
            raw_report = (history.final_result() or "") or ""
        else:
            raw_report = str(history) if history else ""
    except Exception:
        raw_report = ""

    if not raw_report:
        LOG.warning("Scout produced empty report")
        elements = []
    else:
        from langchain_core.messages import HumanMessage
        prompt = EXTRACT_PROMPT.format(report=raw_report)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            text = getattr(response, "content", None)
            if text is None:
                text = str(response)
            elif not isinstance(text, str):
                text = "".join(getattr(c, "content", str(c)) for c in text)
            elements = _parse_elements_from_llm_response(text)
        except Exception as e:
            LOG.warning("Scout LLM extraction failed: %s", e)
            elements = []

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"url": url, "elements": elements}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return elements

"""Scout: list all interactive elements on a page (no click), persist as inventory JSON."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from graph_agent.llm import get_llm, ainvoke_prompt

LOG = logging.getLogger(__name__)

SCOUT_TASK_TEMPLATE = (
    "Navigate to {url} using the navigate action, then identify ALL interactive elements on the page. "
    "Do NOT click anything. Keep output short and deterministic. "
    "Prefer returning a compact JSON object with an `elements` array; "
    "if strict JSON cannot be guaranteed, return a concise plain-text list of selectors and element types only. "
    "No markdown code fences."
)

EXTRACT_PROMPT = """From the following scout report (list of interactive elements on a web page), extract a JSON array of elements.
Each item must have: "selector" (Playwright selector, e.g. #id or [name="x"] or xpath=...), "type" (one of: button, input, link, other), "label" (optional short description or null).
Output only the JSON array, no markdown or explanation.

Scout report:
---
{report}
---
"""

_TYPE_ALIASES: dict[str, str] = {
    "button": "button",
    "btn": "button",
    "submit": "button",
    "input": "input",
    "textbox": "input",
    "text": "input",
    "field": "input",
    "form": "input",
    "select": "input",
    "dropdown": "input",
    "link": "link",
    "anchor": "link",
    "a": "link",
    "other": "other",
}


def _llm_response_to_text(response: Any) -> str:
    """Normalize completion/content variants into plain text."""
    if hasattr(response, "completion"):
        return str(response.completion)
    text = getattr(response, "content", None)
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: list[str] = []
        for item in text:
            parts.append(str(getattr(item, "content", getattr(item, "text", item))))
        return "".join(parts)
    return str(response)


def _parse_elements_from_llm_response(response_text: str) -> list[dict]:
    """Parse JSON array from LLM response; return [] on failure."""
    if not (response_text or "").strip():
        return []
    text = response_text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Fast path: explicit JSON array in text.
    match = re.search(r"\[[\s\S]*\]", text)
    raw_text = match.group() if match else text

    try:
        raw = json.loads(raw_text)
        if isinstance(raw, dict) and isinstance(raw.get("elements"), list):
            raw = raw["elements"]
        if not isinstance(raw, list):
            return []
        elements = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sel = item.get("selector")
            if sel is None:
                continue
            elements.append(
                {
                    "selector": str(sel).strip(),
                    "type": str(item.get("type", "other")).strip() or "other",
                    "label": item.get("label") if item.get("label") is None else str(item.get("label")).strip() or None,
                }
            )
        return elements
    except (json.JSONDecodeError, TypeError) as e:
        LOG.warning("Failed to parse scout LLM response as JSON: %s", e)
        return []


def _infer_type_from_selector_or_label(selector: str, label: str | None) -> str:
    """Infer element type from selector and label when model type is weak."""
    s = (selector or "").lower()
    l = (label or "").lower()
    text = f"{s} {l}"
    if any(token in text for token in ("#btn", ".btn", "button", "submit", "login", "logout")):
        return "button"
    if any(token in text for token in ("input", "name=", "username", "password", "email", "search", "select", "dropdown")):
        return "input"
    if any(token in text for token in ("text=", "xpath=(//a)", "/a", "href", "link")):
        return "link"
    return "other"


def _normalize_type(raw_type: str, selector: str, label: str | None) -> str:
    """Normalize model/heuristic type values into {button,input,link,other}."""
    candidate = (raw_type or "").strip().lower()
    if candidate in _TYPE_ALIASES:
        normalized = _TYPE_ALIASES[candidate]
        if normalized != "other":
            return normalized
    inferred = _infer_type_from_selector_or_label(selector, label)
    return inferred


def normalize_elements(elements: list[dict]) -> list[dict]:
    """Clean and normalize extracted elements.

    - drop empty selector
    - de-duplicate by selector
    - normalize type and label
    """
    seen: set[str] = set()
    normalized: list[dict] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        selector = str(item.get("selector", "")).strip()
        if not selector:
            continue
        dedupe_key = selector.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        label_raw = item.get("label")
        label = None if label_raw is None else (str(label_raw).strip() or None)
        normalized.append(
            {
                "selector": selector,
                "type": _normalize_type(str(item.get("type", "other")), selector, label),
                "label": label,
            }
        )
    return normalized


def _build_scout_metadata(
    extraction_path: str,
    raw_report: str,
    extraction_text: str,
    elements: list[dict],
) -> dict[str, Any]:
    """Build scout metadata with stable quality counters."""
    type_counts = dict(Counter(item.get("type", "other") for item in elements))
    return {
        "extraction_path": extraction_path,
        "raw_report_length": len(raw_report),
        "raw_extraction_text_length": len(extraction_text),
        "element_count": len(elements),
        "type_counts": type_counts,
    }


def _heuristic_extract_elements_from_text(text: str) -> list[dict]:
    """Best-effort selector extraction when JSON parsing fails."""
    if not text.strip():
        return []
    candidates = set(re.findall(r"(#[-_a-zA-Z0-9]+|\[name=['\"][^'\"]+['\"]\]|xpath=[^\s,;]+)", text))
    out: list[dict] = []
    for sel in sorted(candidates):
        kind = "other"
        lowered = sel.lower()
        if "input" in lowered or "name=" in lowered:
            kind = "input"
        elif "button" in lowered or "btn" in lowered:
            kind = "button"
        out.append({"selector": sel, "type": kind, "label": None})
    return out


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

    agent: Any = Agent(
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

    extraction_path = "none"
    extraction_text = ""
    if not raw_report:
        LOG.warning("Scout produced empty report")
        elements = []
    else:
        # Structured-first: parse agent final result directly if it is JSON.
        elements = _parse_elements_from_llm_response(raw_report)
        if elements:
            extraction_path = "direct_structured"
        else:
            prompt = EXTRACT_PROMPT.format(report=raw_report)
            try:
                response = await ainvoke_prompt(llm, prompt)
                extraction_text = _llm_response_to_text(response)
                elements = _parse_elements_from_llm_response(extraction_text)
                if elements:
                    extraction_path = "llm_extract"
                else:
                    heuristic = _heuristic_extract_elements_from_text(raw_report)
                    elements = heuristic
                    extraction_path = "heuristic_fallback"
                    LOG.warning("Scout extraction empty; used heuristic fallback with %s elements", len(elements))
            except Exception as e:
                LOG.warning("Scout LLM extraction failed: %s", e)
                elements = _heuristic_extract_elements_from_text(raw_report)
                extraction_path = "heuristic_after_llm_error"

    elements = normalize_elements(elements)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"url": url, "elements": elements}
        if raw_report:
            payload["metadata"] = _build_scout_metadata(
                extraction_path=extraction_path,
                raw_report=raw_report,
                extraction_text=extraction_text,
                elements=elements,
            )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return elements


def _is_http_url(value: str) -> bool:
    """Return True if value looks like a stable http(s) URL."""
    v = (value or "").strip()
    return v.startswith("http://") or v.startswith("https://")


def _clean_url_for_derived(url: str, base_url: str) -> str:
    """Strip query/fragment and resolve relative URLs to absolute."""
    if not url or not base_url:
        return ""
    raw = (url or "").strip()
    if not _is_http_url(raw):
        raw = urljoin(base_url.rstrip("/") + "/", raw)
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(raw)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return raw


def extract_derived_urls(
    urls: list[str],
    start_url: str,
    *,
    exclude_start: bool = True,
) -> list[str]:
    """Extract unique derived URLs from a URL list (e.g. mapping history).

    - urls: Raw URL list from history.
    - start_url: Base URL for resolving relative paths; used to exclude start.
    - exclude_start: If True, exclude start_url and same-path URLs from result.

    Returns sorted unique http(s) URLs, excluding start when requested.
    """
    base_clean = _clean_url_for_derived(start_url, start_url) if start_url else ""
    seen: set[str] = set()
    result: list[str] = []
    for raw in urls or []:
        cleaned = _clean_url_for_derived(raw, start_url or raw)
        if not cleaned or not _is_http_url(cleaned):
            continue
        if cleaned in seen:
            continue
        if exclude_start and base_clean and cleaned == base_clean:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    result.sort()
    return result


def extract_derived_urls_from_elements(
    elements: list[dict],
    base_url: str,
) -> list[str]:
    """Extract derived URLs from link elements (selector with href).

    Parses selectors like a[href='/path'], xpath=//a[@href='/foo'] to get paths.
    Returns absolute unique URLs for use as scout page hints.
    """
    import re
    derived: set[str] = set()
    for item in elements or []:
        if not isinstance(item, dict):
            continue
        sel = str(item.get("selector", "")).strip()
        if not sel or str(item.get("type", "other")).lower() != "link":
            continue
        # a[href='/path'] or a[href="/path"]
        m = re.search(r'\[href\s*=\s*["\']([^"\']+)["\']\]', sel, re.IGNORECASE)
        if m:
            path = m.group(1).strip()
            if path and not path.startswith("#") and not path.startswith("javascript:"):
                abs_url = path if _is_http_url(path) else urljoin(base_url.rstrip("/") + "/", path)
                if _is_http_url(abs_url):
                    derived.add(_clean_url_for_derived(abs_url, base_url))
        # xpath=//a[@href='/path']
        m2 = re.search(r'@href\s*=\s*["\']([^"\']+)["\']', sel, re.IGNORECASE)
        if m2:
            path = m2.group(1).strip()
            if path and not path.startswith("#") and not path.startswith("javascript:"):
                abs_url = path if _is_http_url(path) else urljoin(base_url.rstrip("/") + "/", path)
                if _is_http_url(abs_url):
                    derived.add(_clean_url_for_derived(abs_url, base_url))
    return sorted(derived)


def _resolve_multi_page_urls(start_url: str, page_hints: list[str] | None = None) -> list[str]:
    """Resolve multi-page scout targets to absolute unique URLs."""
    resolved: list[str] = []
    seen: set[str] = set()
    candidates = [start_url, *(page_hints or [])]
    for candidate in candidates:
        raw = (candidate or "").strip()
        if not raw:
            continue
        absolute = raw if raw.startswith(("http://", "https://")) else urljoin(start_url.rstrip("/") + "/", raw)
        if absolute in seen:
            continue
        seen.add(absolute)
        resolved.append(absolute)
    return resolved


def _aggregate_elements_with_sources(per_page_elements: dict[str, list[dict]]) -> list[dict]:
    """Merge per-page elements and keep source URL trace for each selector/type/label."""
    merged: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for page_url, elements in per_page_elements.items():
        for item in elements:
            if not isinstance(item, dict):
                continue
            selector = str(item.get("selector", "")).strip()
            if not selector:
                continue
            item_type = str(item.get("type", "other")).strip() or "other"
            label_raw = item.get("label")
            label = None if label_raw is None else (str(label_raw).strip() or None)
            key = (selector, item_type, label)
            bucket = merged.get(key)
            if bucket is None:
                merged[key] = {
                    "selector": selector,
                    "type": item_type,
                    "label": label,
                    "source_urls": [page_url],
                }
            else:
                source_urls = bucket["source_urls"]
                if page_url not in source_urls:
                    source_urls.append(page_url)
    out = list(merged.values())
    out.sort(key=lambda x: (x.get("type", "other"), x.get("selector", "")))
    return out


async def run_scout_multi(
    start_url: str,
    page_hints: list[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Run scout on multiple pages and aggregate element inventory."""
    target_urls = _resolve_multi_page_urls(start_url=start_url, page_hints=page_hints)
    per_page_elements: dict[str, list[dict]] = {}
    page_summaries: list[dict[str, Any]] = []
    for target_url in target_urls:
        elements = await run_scout(url=target_url, output_path=None)
        per_page_elements[target_url] = elements
        page_summaries.append(
            {
                "url": target_url,
                "element_count": len(elements),
                "type_counts": dict(Counter(item.get("type", "other") for item in elements)),
            }
        )

    aggregated = _aggregate_elements_with_sources(per_page_elements)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "url": start_url,
            "mode": "multi_page",
            "pages": page_summaries,
            "elements": aggregated,
            "metadata": {
                "page_count": len(target_urls),
                "aggregated_element_count": len(aggregated),
                "type_counts": dict(Counter(item.get("type", "other") for item in aggregated)),
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return aggregated

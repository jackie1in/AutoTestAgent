from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.actor.page import Page
from graph_agent.cartography.types import BBox, LayoutSnapshot


def _parse_eval_payload(raw: object) -> LayoutSnapshot:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw  # type: ignore[return-value]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}  # type: ignore[return-value]
    return {}


def _normalize_bbox(raw_bbox: object) -> BBox:
    if not isinstance(raw_bbox, Mapping):
        return {"x": 0, "y": 0, "width": 0, "height": 0}

    def _as_int(key: str) -> int:
        try:
            return int(raw_bbox.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "x": _as_int("x"),
        "y": _as_int("y"),
        "width": _as_int("width"),
        "height": _as_int("height"),
    }


async def capture_layout_snapshot(page: "Page", limit: int = 200) -> LayoutSnapshot:
    """Capture visible layout anchors and interactive geometry."""
    safe_limit = max(20, min(1000, int(limit or 200)))
    raw = await page.evaluate(
        """
        (...args) => {
            const [limit] = args;
            const vp = {
                width: window.innerWidth || 0,
                height: window.innerHeight || 0,
                scrollX: window.scrollX || 0,
                scrollY: window.scrollY || 0,
            };
            const selectors = [
                "header", "nav", "aside", "main", "section", "form", "table",
                "[role='dialog']", ".modal", ".drawer",
                "[role='tablist']", "[role='tab']", "button", "a[href]",
                "input", "select", "textarea"
            ];
            const nodes = [];
            const seen = new Set();
            const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
            const sampleText = (el) => {
                const t = norm(el.innerText || el.textContent || "");
                return t.slice(0, 60);
            };
            const isVisible = (el, rect) => {
                if (!rect || rect.width < 2 || rect.height < 2) return false;
                if (rect.bottom < 0 || rect.right < 0) return false;
                if (rect.top > vp.height || rect.left > vp.width) return false;
                const style = window.getComputedStyle(el);
                return !(style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0);
            };
            const inScrollContainer = (el) => {
                let p = el.parentElement;
                while (p) {
                    const s = window.getComputedStyle(p);
                    const ov = (s.overflow || "") + (s.overflowY || "") + (s.overflowX || "");
                    if (/auto|scroll/i.test(ov) && (p.scrollHeight > p.clientHeight || p.scrollWidth > p.clientWidth)) {
                        return true;
                    }
                    p = p.parentElement;
                }
                return false;
            };
            const collect = (el) => {
                if (!el || seen.has(el)) return;
                seen.add(el);
                const rect = el.getBoundingClientRect();
                if (!isVisible(el, rect)) return;
                const style = window.getComputedStyle(el);
                const role = el.getAttribute("role") || "";
                const tag = (el.tagName || "").toLowerCase();
                const selector = el.id ? `#${el.id}` : (
                    el.className && typeof el.className === "string"
                        ? `${tag}.${el.className.trim().split(/\\s+/).slice(0, 2).join(".")}`
                        : tag
                );
                nodes.push({
                    tag,
                    role,
                    selector,
                    text_hint: sampleText(el),
                    aria_label: el.getAttribute("aria-label") || "",
                    bbox: {
                        x: Math.round(rect.left),
                        y: Math.round(rect.top),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                    },
                    z_index: style.zIndex || "",
                    is_fixed: style.position === "fixed",
                    is_sticky: style.position === "sticky",
                    in_scroll_container: inScrollContainer(el),
                });
            };
            for (const sel of selectors) {
                const list = Array.from(document.querySelectorAll(sel));
                for (const el of list) {
                    collect(el);
                    if (nodes.length >= limit) break;
                }
                if (nodes.length >= limit) break;
            }
            return JSON.stringify({
                viewport: vp,
                element_count: nodes.length,
                elements: nodes.slice(0, limit),
            });
        }
        """,
        safe_limit,
    )
    return _parse_eval_payload(raw)


def build_layout_summary(snapshot: LayoutSnapshot) -> str:
    """Build compact, stable layout summary for LLM prompt."""
    elements = snapshot.get("elements") if isinstance(snapshot, dict) else []
    if not isinstance(elements, list) or not elements:
        return "No layout anchors detected."

    sections: dict[str, list[str]] = {
        "top_nav": [],
        "sidebar": [],
        "main_content": [],
        "modals": [],
        "forms": [],
        "tables": [],
    }
    for item in elements:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "").lower()
        role = str(item.get("role") or "").lower()
        sel = str(item.get("selector") or tag or "node")
        bbox = _normalize_bbox(item.get("bbox"))
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]
        lower_sel = sel.lower()

        if role == "dialog" or "modal" in lower_sel or "drawer" in lower_sel:
            sections["modals"].append(sel)
            continue
        if tag in {"form", "input", "select", "textarea"}:
            sections["forms"].append(sel)
        if tag in {"table"} or "table" in lower_sel:
            sections["tables"].append(sel)
        if tag in {"header", "nav"} or (y <= 120 and w >= 280):
            sections["top_nav"].append(sel)
        elif tag == "aside" or x <= 280:
            sections["sidebar"].append(sel)
        elif h > 80:
            sections["main_content"].append(sel)

    lines: list[str] = []
    for key in ("top_nav", "sidebar", "main_content", "modals", "forms", "tables"):
        values = sections[key]
        uniq: list[str] = []
        seen: set[str] = set()
        for v in values:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        samples = ", ".join(uniq[:3]) if uniq else "-"
        lines.append(f"{key}: count={len(uniq)} sample={samples}")
    return "\n".join(lines)


def compute_layout_fingerprint(snapshot: LayoutSnapshot) -> str:
    """Compute stable fingerprint from visible layout geometry."""
    elements = snapshot.get("elements") if isinstance(snapshot, dict) else []
    if not isinstance(elements, list) or not elements:
        return ""

    normalized: list[str] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "")
        role = str(item.get("role") or "")
        bbox = _normalize_bbox(item.get("bbox"))
        x = bbox["x"] // 120
        y = bbox["y"] // 120
        w = bbox["width"] // 120
        h = bbox["height"] // 120
        fixed = "1" if item.get("is_fixed") else "0"
        sticky = "1" if item.get("is_sticky") else "0"
        z = str(item.get("z_index") or "auto")
        normalized.append(f"{tag}|{role}|{x},{y},{w},{h}|{fixed}{sticky}|{z}")

    raw = "|".join(sorted(normalized))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def estimate_layout_confidence(snapshot: LayoutSnapshot) -> float:
    """Estimate confidence of sampled layout quality (0.0-1.0)."""
    elements = snapshot.get("elements") if isinstance(snapshot, dict) else []
    if not isinstance(elements, list) or not elements:
        return 0.1

    count = min(1.0, len(elements) / 40.0)
    has_nav = 0.0
    has_main = 0.0
    has_actionables = 0.0
    has_bbox = 0.0

    bbox_ok = 0
    actionable = 0
    for item in elements:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "").lower()
        role = str(item.get("role") or "").lower()
        if tag in {"header", "nav", "aside"}:
            has_nav = 1.0
        if tag in {"main", "section", "table", "form"}:
            has_main = 1.0
        if tag in {"button", "a", "input", "select", "textarea"} or role in {
            "button",
            "link",
            "tab",
            "textbox",
        }:
            actionable += 1
        bbox = _normalize_bbox(item.get("bbox"))
        if bbox["width"] > 1 and bbox["height"] > 1:
            bbox_ok += 1

    if actionable >= 3:
        has_actionables = 1.0
    if bbox_ok >= max(3, len(elements) // 3):
        has_bbox = 1.0

    score = 0.35 * count + 0.2 * has_nav + 0.2 * has_main + 0.15 * has_actionables + 0.1 * has_bbox
    return max(0.1, min(1.0, round(score, 3)))

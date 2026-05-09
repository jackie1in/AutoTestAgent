from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.dom.views import EnhancedDOMTreeNode

DOMSelectorMap = dict[int, "EnhancedDOMTreeNode"]


def _build_element_text_map(selector_map: DOMSelectorMap) -> dict[int, str]:
    text_map: dict[int, str] = {}
    for idx, node in selector_map.items():
        text_map[idx] = _node_short_desc(node)
    return text_map


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

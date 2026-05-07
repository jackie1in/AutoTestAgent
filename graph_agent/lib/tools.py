"""Tool definitions for the ReAct agent.

Aligned with page-agent's tools/index.ts:
- All tools from page-agent included
- Additional cartography-specific tools (go_back, close_overlay)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]

    def format_for_prompt(self) -> str:
        """Format tool description for system prompt."""
        param_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        
        for name, spec in props.items():
            ptype = spec.get("type", "any")
            enum = spec.get("enum")
            default = spec.get("default")
            
            parts = [name]
            if ptype != "object":
                parts.append(f": {ptype}")
            if enum:
                parts.append(f" ({'|'.join(enum)})")
            if default is not None:
                parts.append(f" = {default}")
            elif name not in required:
                parts.append("?")
                
            param_desc.append("".join(parts))
        
        params_str = ", ".join(param_desc) if param_desc else "none"
        return f"- {self.name}: {self.description} ({params_str})"


TOOLS: dict[str, ToolDef] = {
    "done": ToolDef(
        name="done",
        description=(
            "Complete task. Text is your final response — keep it concise "
            "unless the user explicitly asks for detail."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "success": {"type": "boolean", "default": True},
            },
            "required": ["text"],
        },
    ),
    "wait": ToolDef(
        name="wait",
        description=(
            "Wait for x seconds. Can be used to wait until the page or data "
            "is fully loaded."
        ),
        parameters={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 1, "maximum": 10, "default": 1},
            },
            "required": [],
        },
    ),
    "click_element_by_index": ToolDef(
        name="click_element_by_index",
        description="Click element by index",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 0}},
            "required": ["index"],
        },
    ),
    "input_text": ToolDef(
        name="input_text",
        description="Click and type text into an interactive input element",
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "text": {"type": "string"},
            },
            "required": ["index", "text"],
        },
    ),
    "select_dropdown_option": ToolDef(
        name="select_dropdown_option",
        description=(
            "Select dropdown option for interactive element index by the text "
            "of the option you want to select"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "option_text": {"type": "string"},
            },
            "required": ["index", "option_text"],
        },
    ),
    "scroll": ToolDef(
        name="scroll",
        description=(
            "Scroll vertically. Without index: scrolls the document. "
            "With index: scrolls the container at that index "
            "(or its nearest scrollable ancestor)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "default": "down",
                },
                "amount": {"type": "integer", "default": 500},
                "index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional element index to scroll within",
                },
            },
            "required": [],
        },
    ),
    "scroll_horizontally": ToolDef(
        name="scroll_horizontally",
        description=(
            "Scroll horizontally. Without index: scrolls the document. "
            "With index: scrolls the container at that index "
            "(or its nearest scrollable ancestor)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "default": "right",
                },
                "amount": {"type": "integer", "default": 300},
                "index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional element index to scroll within",
                },
            },
            "required": [],
        },
    ),
    "execute_javascript": ToolDef(
        name="execute_javascript",
        description=(
            "Execute JavaScript code on the current page. "
            "Supports async/await syntax. Use with caution!"
        ),
        parameters={
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
        },
    ),
    "go_back": ToolDef(
        name="go_back",
        description="Navigate back to the previous page",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "close_overlay": ToolDef(
        name="close_overlay",
        description=(
            "Close any visible overlay (modal, drawer, dialog) by clicking "
            "its close button or pressing Escape"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
}


def format_tools_for_prompt() -> str:
    """Generate the Available actions section for system prompt.
    
    Usage:
        system_prompt = CARTOGRAPHY_SYSTEM_PROMPT.format(
            max_steps=500,
            available_actions=format_tools_for_prompt()
        )
    """
    lines = ["Available actions (set action_type to the name):"]
    for tool in TOOLS.values():
        lines.append(tool.format_for_prompt())
    return "\n".join(lines)


# Example output verification
if __name__ == "__main__":
    sys.stdout.write(format_tools_for_prompt() + "\n")

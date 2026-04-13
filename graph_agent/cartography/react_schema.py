"""Pydantic schemas for ReAct agent structured output.

Equivalent to page-agent's Zod macro tool schema (AgentOutput).
Used with browser-use ChatOpenAI's ``output_format`` parameter for
strict JSON schema validation via OpenAI ``response_format: json_schema``.

The action field is a discriminated union keyed on ``action_type``,
mirroring page-agent's ``z.union(actionSchemas)`` pattern.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Action parameter models ──────────────────────────────────


class ClickElementAction(BaseModel):
    """Click an interactive element by its index number."""

    action_type: Literal["click_element_by_index"] = "click_element_by_index"
    index: int = Field(description="Element index to click", ge=0)


class InputTextAction(BaseModel):
    """Click and type text into an interactive input element."""

    action_type: Literal["input_text"] = "input_text"
    index: int = Field(description="Element index to input into", ge=0)
    text: str = Field(description="Text to input")


class SelectDropdownAction(BaseModel):
    """Select dropdown option by index and option text."""

    action_type: Literal["select_dropdown_option"] = "select_dropdown_option"
    index: int = Field(description="Element index of the dropdown", ge=0)
    option_text: str = Field(description="Text of the option to select")


class ScrollAction(BaseModel):
    """Scroll vertically. Without index: scrolls the document.
    With index: scrolls the container at that index."""

    action_type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = Field(default="down")
    amount: int = Field(default=500, description="Pixels to scroll")
    index: Optional[int] = Field(
        default=None, description="Optional element index to scroll within"
    )


class ScrollHorizontallyAction(BaseModel):
    """Scroll horizontally. Without index: scrolls the document.
    With index: scrolls the container at that index."""

    action_type: Literal["scroll_horizontally"] = "scroll_horizontally"
    direction: Literal["left", "right"] = Field(default="right")
    amount: int = Field(default=300, description="Pixels to scroll")
    index: Optional[int] = Field(
        default=None, description="Optional element index to scroll within"
    )


class WaitAction(BaseModel):
    """Wait for x seconds until the page or data is fully loaded."""

    action_type: Literal["wait"] = "wait"
    seconds: int = Field(default=1, ge=1, le=10)


class GoBackAction(BaseModel):
    """Navigate back to the previous page."""

    action_type: Literal["go_back"] = "go_back"


class CloseOverlayAction(BaseModel):
    """Close any visible overlay (modal, drawer, dialog)."""

    action_type: Literal["close_overlay"] = "close_overlay"


class ExecuteJavascriptAction(BaseModel):
    """Execute JavaScript code on the current page."""

    action_type: Literal["execute_javascript"] = "execute_javascript"
    script: str = Field(description="JavaScript code to execute")


class DoneAction(BaseModel):
    """Complete the exploration task."""

    action_type: Literal["done"] = "done"
    text: str = Field(
        default="", description="Summary of all discovered transitions"
    )
    success: bool = Field(default=True)


# ── Discriminated union ──────────────────────────────────────

AgentAction = Annotated[
    Union[
        ClickElementAction,
        InputTextAction,
        SelectDropdownAction,
        ScrollAction,
        ScrollHorizontallyAction,
        WaitAction,
        GoBackAction,
        CloseOverlayAction,
        ExecuteJavascriptAction,
        DoneAction,
    ],
    Field(discriminator="action_type"),
]


# ── Macro tool output model ─────────────────────────────────


class AgentOutput(BaseModel):
    """Structured output for each step of the ReAct exploration loop.

    Mirrors page-agent's ``AgentOutput`` Zod macro tool:
    - evaluation + memory + next_goal = reflection
    - action = one of the discriminated action types
    """

    evaluation_previous_goal: str = Field(
        default="",
        description="评估上一步的结果：成功/失败/不确定",
    )
    memory: str = Field(
        default="",
        description="进度记录：已探索哪些元素，哪些待探索",
    )
    next_goal: str = Field(
        default="",
        description="下一步计划及原因",
    )
    action: AgentAction = Field(
        description="要执行的动作",
    )


# ── Conversion helpers ───────────────────────────────────────


def agent_output_to_dict(output: AgentOutput) -> dict:
    """Convert structured AgentOutput to the legacy dict format
    ``{"action": {"action_name": {params}}}`` for backward compatibility.
    """
    action_model = output.action
    action_name = action_model.action_type
    action_params = action_model.model_dump(exclude={"action_type"})

    return {
        "evaluation_previous_goal": output.evaluation_previous_goal,
        "memory": output.memory,
        "next_goal": output.next_goal,
        "action": {action_name: action_params},
    }

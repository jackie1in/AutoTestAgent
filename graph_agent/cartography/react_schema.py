"""Pydantic schemas for custom ReAct agent structured output.

Adapted from the historical react_schema.py to align with current browser-use
action registry names and param models.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ClickElementAction(BaseModel):
    """Click an interactive element by its index number."""

    action_type: Literal["click"] = "click"
    index: int = Field(description="Element index from browser_state", ge=1)


class InputTextAction(BaseModel):
    """Click and type text into an interactive input element."""

    action_type: Literal["input"] = "input"
    index: int = Field(description="Element index from browser_state", ge=0)
    text: str = Field(description="Text to input")


class SelectDropdownAction(BaseModel):
    """Select dropdown option by index and option text."""

    action_type: Literal["select_dropdown"] = "select_dropdown"
    index: int = Field(description="Element index of the dropdown")
    option_text: str = Field(description="Exact text/value of the option to select")


class ScrollAction(BaseModel):
    """Scroll vertically. Without index: scrolls the document.
    With index: scrolls the container at that index."""

    action_type: Literal["scroll"] = "scroll"
    down: bool = Field(default=True, description="True=scroll down, False=scroll up")
    pages: float = Field(default=1.0, description="0.5=half page, 1=full page, 10=to bottom/top")
    index: int | None = Field(
        default=None, description="Optional element index to scroll within specific element"
    )


class WaitAction(BaseModel):
    """Wait for the page or data to fully load."""

    action_type: Literal["wait"] = "wait"
    seconds: int = Field(default=1, ge=1, le=10)


class GoBackAction(BaseModel):
    """Navigate back to the previous page."""

    action_type: Literal["go_back"] = "go_back"


class DoneAction(BaseModel):
    """Complete the exploration task."""

    action_type: Literal["done"] = "done"
    text: str = Field(
        default="", description="Summary of all discovered transitions and states"
    )
    success: bool = Field(default=True)


class DiscoverZonesAction(BaseModel):
    """Discover functional zones on the current page."""

    action_type: Literal["discover_zones"] = "discover_zones"


class ExtractMenuAction(BaseModel):
    """Extract navigation menu from the current page."""

    action_type: Literal["extract_menu"] = "extract_menu"


class QueryKnowledgeAction(BaseModel):
    """Query previously explored knowledge via GraphRAG vector search.

    Use this when you need to recall historical states, intents, or zones
    before deciding the next interaction.
    """

    action_type: Literal["query_knowledge"] = "query_knowledge"
    query_text: str = Field(
        description="Natural language query about previously explored knowledge",
    )
    target_type: Literal["state", "intent", "all"] = Field(
        default="all",
        description="Which index to search: state (pages), intent (actions), or all",
    )


class SolveCaptchaAction(BaseModel):
    """Detect and solve a captcha on the current page using LLM vision.

    Call this when you see a captcha image (e.g. next to a '验证码' input).
    The tool automatically finds the captcha image, extracts it, and returns
    the recognized text. If an input_index is provided, it also fills the
    captcha value into that field.
    """

    action_type: Literal["solve_captcha"] = "solve_captcha"
    input_index: int | None = Field(
        default=None,
        description="Optional index of the captcha input field to fill after solving",
    )


AgentAction = Annotated[
    Union[
        ClickElementAction,
        InputTextAction,
        SelectDropdownAction,
        ScrollAction,
        WaitAction,
        GoBackAction,
        DoneAction,
        DiscoverZonesAction,
        ExtractMenuAction,
        QueryKnowledgeAction,
        SolveCaptchaAction,
    ],
    Field(discriminator="action_type"),
]


class AgentOutput(BaseModel):
    """Structured output for each step of the ReAct exploration loop."""

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


def agent_output_to_dict(output: AgentOutput) -> dict:
    """Convert structured AgentOutput to legacy dict format for backward compat."""
    action_model = output.action
    action_name = action_model.action_type
    action_params = action_model.model_dump(exclude={"action_type"})

    return {
        "evaluation_previous_goal": output.evaluation_previous_goal,
        "memory": output.memory,
        "next_goal": output.next_goal,
        "action": {action_name: action_params},
    }

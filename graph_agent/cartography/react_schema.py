"""Pydantic schemas for ReAct agent structured output.

Action classes are aligned with browser-use's registry names and param models.
Project-specific actions (solve_captcha, etc.) extend the union alongside
browser-use-native actions.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ── Browser-use aligned actions ──────────────────────────────────────


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
    index: int = Field(description="Element index of the dropdown", ge=0)
    text: str = Field(description="Exact text/value of the option to select")


class ScrollAction(BaseModel):
    """Scroll vertically. Without index: scrolls the document.
    With index: scrolls the container at that index."""

    action_type: Literal["scroll"] = "scroll"
    down: bool = Field(default=True, description="True=scroll down, False=scroll up")
    pages: float = Field(
        default=1.0, description="0.5=half page, 1=full page, 10=to bottom/top"
    )
    index: int | None = Field(
        default=None,
        description="Optional element index to scroll within specific element",
    )


class ScrollHorizontallyAction(BaseModel):
    """Scroll horizontally (e.g. wide tables, carousels)."""

    action_type: Literal["scroll_horizontally"] = "scroll_horizontally"
    direction: Literal["left", "right"] = Field(
        default="right", description="Scroll direction"
    )
    amount: int = Field(default=300, description="Pixels to scroll")
    index: int | None = Field(
        default=None,
        description="Optional element index to scroll within",
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


class SendKeysAction(BaseModel):
    """Send keyboard keys or shortcuts to the page. Use for hotkeys like
    Alt+Z to open menus, Escape to close modals, PageDown to scroll, etc.
    Format: 'Alt+Z', 'Control+S', 'Escape', 'Enter', 'PageDown', etc."""

    action_type: Literal["send_keys"] = "send_keys"
    keys: str = Field(
        description="Keys or shortcut (e.g. 'Escape', 'Enter', 'PageDown', 'Control+o', 'Alt+Z')"
    )


class EvaluateAction(BaseModel):
    """Execute JavaScript on the page. Best practice: wrap in IIFE
    (function(){...})() with try-catch. Use ONLY browser APIs (document, window).
    Avoid comments. Use for hover, drag, zoom, custom selectors, or analysing
    page structure."""

    action_type: Literal["evaluate"] = "evaluate"
    code: str = Field(description="JavaScript code to execute")


class DropdownOptionsAction(BaseModel):
    """Get all options from a native dropdown or ARIA menu at the given index."""

    action_type: Literal["dropdown_options"] = "dropdown_options"
    index: int = Field(description="Element index of the dropdown", ge=0)


class FindElementsAction(BaseModel):
    """Query DOM elements by CSS selector (like find). Zero LLM cost, instant.
    Returns matching elements with tag, text, and attributes."""

    action_type: Literal["find_elements"] = "find_elements"
    selector: str = Field(description="CSS selector (e.g. 'table tr', 'a.link')")
    attributes: list[str] | None = Field(
        default=None,
        description="Specific attributes to extract (e.g. ['href', 'src'])",
    )
    max_results: int = Field(default=50, description="Maximum elements to return")
    include_text: bool = Field(default=True, description="Include text content")


class SearchPageAction(BaseModel):
    """Search page text for a pattern (like grep). Zero LLM cost, instant.
    Returns matches with surrounding context."""

    action_type: Literal["search_page"] = "search_page"
    pattern: str = Field(description="Text or regex pattern to search for")
    regex: bool = Field(default=False, description="Treat pattern as regex")
    case_sensitive: bool = Field(default=False)
    max_results: int = Field(default=25, description="Maximum matches to return")


class CloseOverlayAction(BaseModel):
    """Close any visible overlay (modal, drawer, dialog) by clicking
    its close button or pressing Escape."""

    action_type: Literal["close_overlay"] = "close_overlay"


# ── Project-specific actions ─────────────────────────────────────────


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
    """Detect and solve an image-based verification code on the current page using LLM vision.

    ## 何时调用
    当页面上存在需要"看图填码"的 input 时调用，判断依据是 input 的 label、
    placeholder、name 或 id 中含有以下任意特征：
    - 验证码 / captcha / 图形码 / 图片验证码
    - 验证码图片旁边有刷新/换一张按钮
    - 二维码扫码后需要填入的确认码（非扫码本身）

    ## 不调用的情况
    - 手机短信验证码（placeholder 含"手机"/"短信"/"SMS"）→ 直接用 input_text 填入
    - 邮箱验证码 → 直接用 input_text 填入
    - 二维码图片本身（用户需用手机扫描）→ 不需要调用此工具，跳过或等待扫码
    - 普通密码框 → 直接用 input_text 填入

    ## 工作方式
    工具自动定位 input_index 对应输入框附近的图像/canvas 元素，
    用 LLM Vision 识别验证码文字，并自动将结果填入 input_index 指定的输入框。
    """

    action_type: Literal["solve_captcha"] = "solve_captcha"
    input_index: int | None = Field(
        default=None,
        description=(
            "验证码输入框的 index（从 browser_state 的 [N] 编号中取）。"
            "工具会以此 input 为锚点定位旁边的验证码图像，识别后自动填入。"
            "如果不确定 index，传 null，工具会用启发式方法寻找验证码图像。"
        ),
    )
    input_hint: str = Field(
        default="",
        description=(
            "可选。验证码输入框的语义特征，如 placeholder 或 label 文本，"
            "例如 '图形验证码'、'验证码'、'captcha'。用于辅助定位验证码图像。"
        ),
    )


# ── Agent action union ───────────────────────────────────────────────

AgentAction = Annotated[
    Union[
        # Browser-use aligned
        ClickElementAction,
        InputTextAction,
        SelectDropdownAction,
        ScrollAction,
        ScrollHorizontallyAction,
        WaitAction,
        GoBackAction,
        DoneAction,
        SendKeysAction,
        EvaluateAction,
        DropdownOptionsAction,
        FindElementsAction,
        SearchPageAction,
        CloseOverlayAction,
        # Project-specific
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

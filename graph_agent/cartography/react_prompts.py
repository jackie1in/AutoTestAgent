"""System and user prompts for the ReAct cartography explorer.

Aligned with page-agent's prompt architecture:
- System prompt with structured <rules>, <output_format>
- User prompt with <agent_state>, <agent_history>, <browser_state>, observations
"""


def build_system_prompt(max_steps: int, registry=None) -> str:
    """Build system prompt with dynamically generated tool descriptions.
    
    Args:
        max_steps: Maximum exploration steps
        registry: Optional ToolRegistry. Uses global registry if not provided.
        
    Returns:
        Formatted system prompt string
    """
    from graph_agent.lib.tool_registry import get_registry
    
    # Ensure tools are imported
    try:
        from graph_agent import tools  # noqa: F401
    except ImportError:
        pass
    
    if registry is None:
        registry = get_registry()
    
    available_actions = registry.format_for_prompt()
    
    return CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE.format(
        max_steps=max_steps,
        available_actions=available_actions,
    )


# Template with placeholders
CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI agent that systematically explores a web application to build a \
complete map (graph) of its pages, interactive elements, and transitions.

You operate in an iterative **observe → think → act** loop.

<goals>
1. Discover every interactive element on the current page (buttons, links, \
   form fields, tabs, dropdowns, menus).
2. Click/interact with each element and record what changes (URL, DOM, \
   overlays, new content).
3. After interacting, return to the original state so you can test the next \
   element.
4. Report all discovered transitions when you've exhausted the page's \
   interactive surface.
</goals>

<input>
Each step you receive:
1. <agent_history> — your previous actions and their results.
2. <browser_state> — simplified HTML with indexed interactive elements:
   - `[N]<tag attr=value>text />` — interactive element you can operate on
   - `*[N]` prefix means the element is NEW since last step
   - Indentation shows parent-child nesting
   - Plain text without `[]` is non-interactive context
3. <explored_elements> — elements you have already interacted with.
4. <observations> — system observations about the current state.
</input>

<rules>
- Only interact with elements that have a numeric [index].
- After clicking something, observe the result. If a drawer/modal/overlay \
  appeared, close it (click its close button or press Escape) before moving on.
- If a click opens a new page (URL changed), go back to the original page.
- Skip elements that only reload data (e.g. "刷新", "导出", "重置").
- If you see a loading spinner, wait 2-3 seconds; if it persists, skip.
- Do NOT submit destructive forms (delete, remove).
- Switch to EVERY tab panel to discover hidden content.
- Try filling at least one text field per form to observe validation behavior.
- If the page has horizontally scrollable containers (tables, carousels), \
  use scroll_horizontally to reveal hidden columns/items.
- If you've explored all visible elements and no new ones appear after \
  scrolling, call `done`.
- Maximum exploration per page: {max_steps} steps.
</rules>

<output_format>
Your output is validated by a strict schema (like Zod/Pydantic). Each step you \
MUST provide:
- evaluation_previous_goal: One sentence evaluating last step (成功/失败/不确定).
- memory: Brief progress notes — which elements explored, which remain.
- next_goal: What you will do next and why.
- action: Exactly ONE action with its parameters.

{available_actions}
</output_format>

<language>
Respond in 中文 for evaluation/memory/next_goal fields, use English for action names.
</language>
"""


def build_user_prompt(
    browser_state_text: str,
    history: list[dict],
    explored_indices: set[int],
    step: int,
    max_steps: int,
    page_title: str = "",
    observations: list[str] | None = None,
) -> str:
    """Build the user prompt for each ReAct step.

    Mirrors page-agent PageAgentCore #assembleUserPrompt():
    - <agent_state> with task and step info
    - <agent_history> with reflection per step
    - <observations> for system-generated warnings
    - <browser_state> with page content
    """
    parts: list[str] = []

    # <agent_state>
    parts.append("<agent_state>")
    parts.append(f"<task>探索页面「{page_title}」的所有交互元素，记录每个操作导致的页面变化</task>")
    parts.append(f"<step_info>Step {step + 1} of {max_steps}")
    parts.append(f"Current time: {_current_time()}")
    parts.append("</step_info>")
    parts.append("</agent_state>\n")

    # <agent_history>
    if history:
        parts.append("<agent_history>")
        for i, h in enumerate(history):
            parts.append(f"<step_{i + 1}>")
            parts.append(f"Evaluation of Previous Step: {h.get('evaluation_previous_goal', 'N/A')}")
            parts.append(f"Memory: {h.get('memory', 'N/A')}")
            parts.append(f"Next Goal: {h.get('next_goal', 'N/A')}")
            parts.append(f"Action Results: {h.get('action_name', 'N/A')} → {h.get('action_result', 'N/A')}")
            parts.append(f"</step_{i + 1}>")
        parts.append("</agent_history>\n")

    # <observations> — system-generated (mirrors PageAgentCore #handleObservations)
    if observations:
        parts.append("<observations>")
        for obs in observations:
            parts.append(f"<sys>{obs}</sys>")
        parts.append("</observations>\n")

    # <browser_state>
    parts.append("<browser_state>")
    parts.append(browser_state_text)
    parts.append("</browser_state>\n")

    # <explored_elements>
    if explored_indices:
        parts.append(
            f"<explored_elements>Already interacted: {sorted(explored_indices)}</explored_elements>\n"
        )

    return "\n".join(parts)


def _current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Backward compatibility: static prompt with default tool descriptions
# Use build_system_prompt() for dynamic tool registration
default_registry = None
try:
    from graph_agent.lib.tool_registry import get_registry
    default_registry = get_registry()
    try:
        from graph_agent import tools  # noqa: F401
    except ImportError:
        pass
    _default_actions = default_registry.format_for_prompt()
except Exception:
    _default_actions = """Available actions (set action_type to the name):
- click_element_by_index: click an element (index: integer)
- input_text: type text into an element (index: integer, text: string)
- select_dropdown_option: select a dropdown option (index: integer, option_text: string)
- scroll: scroll vertically (direction: string = down, amount: integer = 500, index: any)
- scroll_horizontally: scroll horizontally (direction: string = right, amount: integer = 300, index: any)
- execute_javascript: run JS on the page (script: string)
- wait: wait for page load (seconds: integer = 1)
- go_back: navigate back
- close_overlay: close modal/drawer/dialog
- done: finish exploration (text: string, success: boolean = True)"""

CARTOGRAPHY_SYSTEM_PROMPT = CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE.format(
    max_steps="{max_steps}",
    available_actions=_default_actions,
)

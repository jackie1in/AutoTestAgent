"""System and user prompts for the ReAct cartography explorer.

Aligned with page-agent's prompt architecture:
- System prompt with structured <rules>, <output_format>
- User prompt with <agent_state>, <agent_history>, <browser_state>, observations
"""

from __future__ import annotations

from browser_use.tools.registry.service import Registry


def build_system_prompt(
    max_steps: int,
    registry: "Registry | None" = None,
    supported_actions: set[str] | frozenset[str] | None = None,
) -> str:
    """Build system prompt with dynamically generated tool descriptions.

    Args:
        max_steps: Maximum exploration steps
        registry: Browser-use Registry instance. Uses default Tools registry if None.
        supported_actions: Optional whitelist of action names to include in the prompt.
            If None, all registered actions are shown.

    Returns:
        Formatted system prompt string
    """
    # Build available actions text from browser-use registry
    description_lines = _build_action_descriptions(registry, supported_actions)

    return CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE.format(
        max_steps=max_steps,
        available_actions=description_lines,
    )


def _build_action_descriptions(
    registry: "Registry | None" = None,
    supported_actions: set[str] | frozenset[str] | None = None,
) -> str:
    """Build action descriptions from browser-use registry."""
    if registry is not None:
        # Use browser-use Registry.get_prompt_description()
        base_description = registry.get_prompt_description(page_url=None)
        if supported_actions is not None:
            # Filter to only supported actions
            filtered = []
            for line in base_description.split("\n"):
                if not line.strip():
                    continue
                action_name = line.split(":")[0].strip()
                if action_name in supported_actions:
                    filtered.append(line)
            # Add project-specific actions not in browser-use
            for name in sorted(supported_actions):
                if _is_custom_action(name):
                    filtered.append(_custom_action_description(name))
            return "\n".join(filtered) if filtered else _fallback_descriptions()
        return base_description if base_description.strip() else _fallback_descriptions()
    return _fallback_descriptions()


def _is_custom_action(name: str) -> bool:
    """Check if an action name is project-specific (not in browser-use)."""
    return name in {
        "scroll_horizontally",
        "close_overlay",
        "query_knowledge",
        "discover_zones",
        "extract_menu",
        "solve_captcha",
    }


def _custom_action_description(name: str) -> str:
    """Generate prompt description for project-specific actions."""
    descriptions = {
        "scroll_horizontally": (
            "scroll_horizontally: Scroll horizontally (e.g. wide tables, carousels). "
            "(direction: string = right, amount: integer = 300, index: integer?)"
        ),
        "close_overlay": (
            "close_overlay: Close any visible overlay (modal, drawer, dialog) "
            "by clicking its close button or pressing Escape. (none)"
        ),
        "query_knowledge": (
            "query_knowledge: Query previously explored knowledge via vector search. "
            "(query_text: string, target_type: string = all)"
        ),
        "discover_zones": (
            "discover_zones: Discover functional zones on the current page. (none)"
        ),
        "extract_menu": (
            "extract_menu: Extract navigation menu structure. Call AFTER opening"
            " the menu (e.g. send_keys('Alt+Z')). Returns JSON with items,"
            " each having text/href/level/children. Empty items if menu not found. (none)"
        ),
        "solve_captcha": (
            "solve_captcha: Detect and solve image-based captcha on the current page "
            "using LLM vision. (input_index: integer?, input_hint: string?)"
        ),
    }
    return descriptions.get(name, f"{name}: Custom action")


def _fallback_descriptions() -> str:
    """Fallback when no registry is available."""
    return """Available actions (set action_type to the name):
- click: Click element by index (index: integer)
- input: Click and type text into an input element (index: integer, text: string)
- select_dropdown: Select dropdown option by index and text (index: integer, text: string)
- scroll: Scroll vertically (down: boolean = true, pages: float = 1.0, index: integer?)
- scroll_horizontally: Scroll horizontally (direction: string = right, amount: integer = 300, index: integer?)
- send_keys: Send keyboard keys/shortcuts like Alt+Z, Escape (keys: string)
- evaluate: Execute JavaScript on the page (code: string)
- dropdown_options: Get all options from a dropdown (index: integer)
- find_elements: Query DOM elements by CSS selector (selector: string, attributes: string[]?, max_results: integer = 50)
- search_page: Search page text for a pattern (pattern: string, regex: boolean = false)
- wait: Wait for x seconds (seconds: integer = 1)
- close_overlay: Close overlays/modal/drawer (none)
- query_knowledge: Query historical knowledge (query_text: string, target_type: string = all)
- discover_zones: Discover page functional zones (none)
- extract_menu: Extract navigation menu structure — call AFTER opening menu (none)
- solve_captcha: Detect and solve image captcha (input_index: integer?, input_hint: string?)
- done: Complete exploration task (text: string, success: boolean = true)"""


# Template with placeholders
CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI agent that systematically explores a web application to build a \
complete map (graph) of its pages, interactive elements, and transitions.

You operate in an iterative **observe → think → act** loop.

<goals>
1. Explore the current page: discover interactive elements, click them, observe changes.
2. Extract the navigation menu structure so you can navigate to other pages.
3. Follow menu items in depth-first order: explore one branch fully, then the next.
4. Report all discovered transitions when you've exhausted the page.
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

<menu_discovery>
To discover the navigation menu structure:
  1. First, try to open the menu (e.g. send_keys('Alt+Z') if a hotkey is needed, or click a menu trigger element).
  2. Wait for the menu to appear, then call extract_menu.
  3. If extract_menu returns items with text/href/level/children, you have the full tree.
     Navigate by clicking menu items directly — no need to go back.
  4. If extract_menu returns empty items (no menu found):
     a. The menu may not be visible. Try send_keys with different shortcuts.
     b. Visually scan the DOM for menu-like elements (nav bars, sidebars).
     c. Click each visible top-level menu item one by one, observe what changes:
        - If a submenu appears → record its items, click them to discover pages.
        - If the page navigates → you've found a leaf node, explore that page.
     d. Build up the menu tree manually through interaction.
  5. Once you have the menu tree, follow it depth-first:
     go menu-item-1 → explore its page → use the menu to navigate to menu-item-2 → ...
</menu_discovery>

<rules>
- Only interact with elements that have a numeric [index].
- After clicking something, observe the result. If a drawer/modal/overlay \
  appeared, close it (click its close button, press Escape, or use send_keys) before moving on.
- You are exploring DEPTH-FIRST: follow one navigation path to its end, \
  then use the menu to go to the next sibling branch. Do NOT go back.
- Skip elements that only reload data (e.g. "刷新", "导出", "重置").
- If you see a loading spinner, wait 2-3 seconds; if it persists, skip.
- Do NOT submit destructive forms (delete, remove).
- Switch to EVERY tab panel to discover hidden content.
- Try filling at least one text field per form to observe validation behavior.
- If the page has horizontally scrollable containers (tables, carousels), \
  use scroll_horizontally to reveal hidden columns/items.
- Use send_keys to trigger keyboard shortcuts (e.g. Alt+Z, Ctrl+S) if menus \
  or features are hidden behind icons or hotkeys.
- Use evaluate to run JavaScript for hover, drag, zoom, or analyzing page structure.
- Use dropdown_options before select_dropdown to inspect available options.
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
    target_zone_selectors: list[str] | None = None,
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

    # <zone_filter> — SkipAdvisor 决策为 EXPLORE_ZONES_ONLY 时注入；软约束
    if target_zone_selectors:
        zone_lines = "\n".join(f"  - {sel}" for sel in target_zone_selectors[:20])
        parts.append(
            "<zone_filter>\n"
            "本页面已被部分探索过，仅以下区域 (root selector) 仍待补充覆盖：\n"
            f"{zone_lines}\n"
            "请优先且只与位于这些 selector 子树内的元素交互；"
            "若全部已尽则立即调用 done。"
            "如果当前 DOM 中找不到这些 selector，可退化为常规探索。\n"
            "</zone_filter>\n"
        )

    return "\n".join(parts)


def _current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

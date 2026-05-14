"""System and user prompts for the ReAct cartography explorer.

Aligned with page-agent's prompt architecture:
- System prompt with structured <rules>, <output_format>
- User prompt with <agent_state>, <agent_history>, <browser_state>, observations
"""

from __future__ import annotations

from browser_use.tools.registry.service import Registry


def build_system_prompt(
    registry: "Registry | None" = None,
    supported_actions: set[str] | frozenset[str] | None = None,
) -> str:
    """Build system prompt with dynamically generated tool descriptions."""
    description_lines = _build_action_descriptions(registry, supported_actions)

    return CARTOGRAPHY_SYSTEM_PROMPT_TEMPLATE.format(
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
            "discover_zones: Analyze page with LLM to discover functional zones "
            "(form, table, action_bar, filter, modal, tabs, etc.) and their CSS "
            "selectors. Use as the FIRST action when encountering a new page to "
            "understand its structure and focus exploration on interactive areas. (none)"
        ),
        "extract_menu": (
            "extract_menu: Extract navigation menu structure. Call AFTER opening"
            " the menu (e.g. send_keys('Alt+Z')). Returns JSON with items,"
            " each having text/href/level/children. Uses JS for known frameworks"
            " (Ant Design, Element UI), falls back to LLM vision for custom menus."
            " Empty items if menu truly not found. (none)"
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
- discover_zones: Analyze page structure via LLM — call FIRST on new pages to identify functional zones by type+selector (none)
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
Navigation menus in SPAs often use internal routing — clicking a menu item
changes content without changing the full URL.  Explore ALL menu branches
in one session using this depth-first pattern:

  1. Open the menu (send_keys shortcut or click menu trigger).
  2. Call extract_menu to get the full tree (text/href/level/children).
  3. If extract_menu returns empty: manually discover by clicking each visible
     top-level menu item and observing what appears.
  4. For each menu item in order:
     a. Click the menu item to navigate to its page.
     b. Explore that page fully (elements, tabs, forms, transitions).
     c. When done with that page, return to the HOME PAGE:
        - Click the logo/home link, OR
        - Re-open the menu and click the first "首页/Home" item
     d. Re-open the menu (shortcut or click trigger).
     e. Click the NEXT sibling menu item, repeat step 4.
  5. Only call done when ALL menu branches have been explored.
     Do NOT stop early — you have enough steps to complete the full tree.
</menu_discovery>

<zone_discovery>
Before exploring interactive elements, call `discover_zones` to understand the
page's functional layout.  It identifies zones by type (form, table, action_bar,
filter, modal, tabs, card, chart, list, content) with their CSS selectors.
Use these zones to:
  - Prioritize exploration: forms > action_bars > tables > tabs > content.
  - Scope interactions to specific zone selectors when the LLM knows them.
  - Re-call after major page changes (tab switch, modal open, navigation).
Call `discover_zones` as the FIRST action on every new page to avoid missing
hidden interactive areas.
</zone_discovery>

<form_exploration>
Forms (CREATE/EDIT dialogs) are HIGH-VALUE exploration targets — each input
field is a transition that will be used to generate test scripts.

When you open a form:
  IMPORTANT: Filling and submitting the form IS the exploration. Do NOT
  skip a form because "this is just exploration" or "the form has too many
  fields." Every field you fill and every submit result is a recorded
  transition used to generate real test scripts later.
  1. Systematically fill EVERY visible input, select, checkbox, and radio
     field ONE AT A TIME (each `input` action is one transition).
  2. After filling ALL fields, click the submit/save/confirm button.
     → If save succeeds: observe the result page (list updated, toast message).
     → If save fails (page stays, no network request, or validation errors):
       a. Scan for unfilled required fields: red border-color, red asterisk *,
          aria-required, HTML5 required. Fill every one you find.
       b. Click submit again. Repeat this retry loop up to 3 times.
       c. Only after 3 failed submit attempts, give up — click back/cancel to
          return to the list page (do NOT skip the form without trying to save).
  3. EXCEPTION: skip the submit button if the form is clearly destructive
     (delete, remove, reset password, batch delete).
  4. Fill ALL required fields (marked with red asterisk *, aria-required,
     class="required", HTML5 required attribute, or red border-color).
     Do not skip any required field regardless of how many fields the form has.
  5. Use `dropdown_options` before `select_dropdown` to discover available
     options for each select field.  If `dropdown_options` says the element
     is NOT a native <select>, do NOT use `select_dropdown` — instead click
     the option directly.
  6. For inputs whose placeholder contains "搜索": click the input first.
     If NO dropdown appears, look for a "设置" or "创建" button in the same
     row. Click it to open a creation dialog, fill and submit, then type
     the created record's name into the original search input.
</form_exploration>

<dropdown_handling>
Before using `dropdown_options` or `select_dropdown`, understand the dropdown type:

1. The system checks the element's HTML tag. If it's NOT <select>, it returns
   a warning. Pay attention to this warning!

2. STANDARD (<select> tag): `dropdown_options` lists options, `select_dropdown`
   selects one. Works as expected.

3. NON-STANDARD (<div>, <span>, <input>, or anything not <select>):
   a. Click the trigger element to open the dropdown panel
   b. Identify the dropdown panel that appeared — look for the popup/dropdown
      container directly adjacent to or below the trigger you clicked.
      IMPORTANT: scope to THIS dropdown only, NOT all dropdowns on the page.
      If using evaluate/JS, first locate the parent dropdown container
      (the popup/panel adjacent to the trigger you clicked), then query
      options only within that parent — never query options globally.
   c. Use regular `click` on the specific option you want within that panel
   d. For tree: expand parent nodes first, then click the leaf option
   e. For searchable selects: input text to filter, then click the match
</dropdown_handling>

<data_safety>
Before performing destructive operations (delete/remove/reset):

STEP 1 — SCAN: Use `extract` tool with query="list all rows or items that
  contain [AUTO]" to scan the data display area (table, list, card grid).
  → Found [AUTO] records: proceed to STEP 4
  → No [AUTO] records found: go to STEP 2

STEP 2 — SEARCH: Check if the page has a search/filter input.
  → Found a search box: input "[AUTO]" and submit the search.
  → No search box: try scrolling or switching tabs to find [AUTO] records.
  → Still none found after search: STOP — no agent data to operate on.
    Report "No [AUTO] records found; destructive test skipped."

STEP 3 — VERIFY: After search results appear, confirm at least one row
  contains "[AUTO]". If the search returned unrelated results, STOP.

STEP 4 — OPERATE: Now click the delete/remove button on the [AUTO] row.
  Only operate on ONE record at a time.

STEP 5 — CONFIRM: After deletion, verify the [AUTO] record is gone.
  If the page has pagination, check that the record count decreased.

Other rules:
- When filling forms to CREATE records: name/title fields are AUTO-prefixed
  with "[AUTO]" by the system. You don't need to add it yourself.
- When MODIFYING: add "[Agent操作]" to remark/description fields before saving.
</data_safety>

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
- When exploring forms: fill EVERY field systematically, then submit. Each input=one transition.
- If the page has horizontally scrollable containers (tables, carousels), \
  use scroll_horizontally to reveal hidden columns/items.
- Use send_keys to trigger keyboard shortcuts (e.g. Alt+Z, Ctrl+S) if menus \
  or features are hidden behind icons or hotkeys.
- Use evaluate to run JavaScript for hover, drag, zoom, or analyzing page structure.
- Use dropdown_options before select_dropdown to inspect available options.
- If you've explored ALL menu branches and all visible elements on every page, \
  and no new items appear, call `done`.
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
    history: list[dict[str, object]],
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

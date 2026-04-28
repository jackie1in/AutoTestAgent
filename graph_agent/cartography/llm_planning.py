from __future__ import annotations

import os

from browser_use.llm.base import BaseChatModel
from pydantic import BaseModel, Field

from graph_agent.cartography.types import LLMTransitionHint
from graph_agent.llm.utils import ainvoke_structured


class LLMMenuItem(BaseModel):
    text: str = Field(description="Menu item display text")
    href: str = Field(default="", description="Link URL or route path")
    level: int = Field(default=0, description="Hierarchy level (0 = top level)")


class LLMFunctionalZone(BaseModel):
    zone_type: str = Field(
        description="Zone type: form, table, nav, action_bar, filter, modal, card, tabs, chart, list, content"
    )
    selector: str = Field(description="Best CSS selector to locate the zone")
    description: str = Field(default="", description="Brief description of what this zone does")


class LLMPageAnalysis(BaseModel):
    page_type: str = Field(
        description="Page type: login, dashboard, list, detail, form, settings, welcome, unknown"
    )
    menu_items: list[LLMMenuItem] = Field(default_factory=list)
    functional_zones: list[LLMFunctionalZone] = Field(default_factory=list)
    is_login_page: bool = Field(default=False)
    reasoning: str = Field(default="")


class LLMExplorationTask(BaseModel):
    task_type: str = Field(
        description="Type: explore_page, explore_zone, click_menu, validate_transition, stop"
    )
    target_url: str = Field(default="")
    target_selector: str = Field(default="")
    description: str = Field(default="")
    expected_outcome: str = Field(default="")


class LLMExplorationPlan(BaseModel):
    tasks: list[LLMExplorationTask] = Field(default_factory=list)
    strategy: str = Field(default="continue")
    coverage_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


PAGE_TYPE_ACTION_POLICY: dict[str, str] = {
    "login": "ACTION POLICY: Login page. If credentials are available via the auto-login hint, use them.",
    "form": "ACTION POLICY: Form page. Prefer `input` over `click`, then submit.",
    "list": "ACTION POLICY: List/table page. Prefer row clicks and row-level actions.",
    "detail": "ACTION POLICY: Detail page. Click sub-tabs and primary actions.",
    "dashboard": "ACTION POLICY: Dashboard page. Avoid navigation menus; inspect in-page widgets only.",
    "settings": "ACTION POLICY: Settings page. Toggle and save settings.",
    "welcome": "ACTION POLICY: Welcome page. Click primary CTA.",
}

UNIVERSAL_EXPLORER_RULES = (
    "GENERAL RULES:\n"
    "  - Thoroughly exercise the CURRENT page then call `done`.\n"
    "  - Do not repeatedly click global navigation menus.\n"
    "  - On cross-domain redirects, call `done` immediately.\n"
    "  - If page unchanged, continue other elements on same page."
)


def build_login_hint_from_env() -> str:
    username = (os.getenv("MAPPING_USERNAME") or "").strip()
    password = (os.getenv("MAPPING_PASSWORD") or "").strip()
    if not username and not password:
        return ""
    parts: list[str] = []
    if username:
        parts.append(f"username={username}")
    if password:
        parts.append(f"password={password}")
    credentials = ", ".join(parts)
    return (
        "若页面包含登录表单，优先使用以下测试账号完成登录："
        f"{credentials}。如字段名不同，请根据语义匹配对应输入框。"
    )


def build_exploration_guidance(page_analysis: LLMPageAnalysis) -> str:
    blocks: list[str] = [UNIVERSAL_EXPLORER_RULES]
    policy_text = PAGE_TYPE_ACTION_POLICY.get(page_analysis.page_type)
    if policy_text:
        blocks.append(policy_text)
    if page_analysis.functional_zones:
        zone_lines = [
            f"  {i+1}. [{z.zone_type}] {z.selector}" + (f" — {z.description}" if z.description else "")
            for i, z in enumerate(page_analysis.functional_zones[:5])
        ]
        blocks.append(
            "ZONE ORDER (explore in this sequence, as they appear on the page):\n"
            + "\n".join(zone_lines)
        )
    return "\n\n".join(blocks)


async def analyze_page_with_llm(
    llm: BaseChatModel,
    dom_text: str,
    current_url: str,
    page_title: str = "",
    layout_summary: str = "",
) -> LLMPageAnalysis:
    system_prompt = (
        "You are an expert web UI analyzer. Analyze page structure and return menu_items, "
        "functional_zones, login flag, and page_type."
    )
    layout_block = (
        f"\nLayout summary (visual structure):\n{layout_summary[:2000]}\n"
        if layout_summary.strip()
        else "\nLayout summary (visual structure):\n(none)\n"
    )
    user_prompt = (
        f"Current URL: {current_url}\n"
        f"Page Title: {page_title}\n\n"
        f"{layout_block}\n"
        f"DOM representation (first 12000 chars):\n{dom_text[:12000]}\n\n"
        "Preserve menu and zone order based on DOM."
    )
    try:
        return await ainvoke_structured(
            llm,
            system_prompt,
            user_prompt,
            LLMPageAnalysis,
            max_retries=2,
            timeout_ms=45_000,
        )
    except Exception as e:
        print(f"[LLM-ORCH] Page analysis failed: {e}. Falling back to empty analysis.")
        return LLMPageAnalysis(page_type="unknown", reasoning=f"Analysis failed: {e}")


async def plan_next_exploration_with_llm(
    llm: BaseChatModel,
    current_url: str,
    page_title: str,
    page_analysis: LLMPageAnalysis,
    explored_urls: list[str],
    discovered_transitions: list[LLMTransitionHint],
    time_budget_remaining_ms: float,
    max_steps_remaining: int,
) -> LLMExplorationPlan:
    explored_summary = "\n".join(f"  - {url}" for url in explored_urls[-20:]) or "  (none yet)"
    transition_summary = "\n".join(
        f"  - {t.get('action', '?')} on {t.get('selector', '?')} -> {t.get('to_url', '?')[:60]}"
        for t in discovered_transitions[-15:]
    ) or "  (none yet)"
    menu_summary = "\n".join(
        f"  - [{m.level}] {m.text} ({m.href or 'no href'})"
        for m in page_analysis.menu_items[:15]
    ) or "  (none detected)"
    zone_summary = "\n".join(
        f"  - [{z.zone_type}] {z.selector}"
        for z in page_analysis.functional_zones[:15]
    ) or "  (none detected)"

    system_prompt = (
        "You are an expert exploration planner for web application cartography."
    )
    user_prompt = (
        f"Current URL: {current_url}\n"
        f"Page Title: {page_title}\n"
        f"Page Type: {page_analysis.page_type}\n\n"
        f"Time budget remaining: {time_budget_remaining_ms / 1000:.0f}s\n"
        f"Max steps remaining: {max_steps_remaining}\n\n"
        f"Menu items detected:\n{menu_summary}\n\n"
        f"Functional zones detected:\n{zone_summary}\n\n"
        f"Already explored URLs ({len(explored_urls)} total):\n{explored_summary}\n\n"
        f"Recent transitions:\n{transition_summary}\n\n"
        "Plan the next exploration tasks. Return an ordered list of tasks."
    )
    try:
        return await ainvoke_structured(
            llm,
            system_prompt,
            user_prompt,
            LLMExplorationPlan,
            max_retries=2,
            timeout_ms=45_000,
        )
    except Exception as e:
        print(f"[LLM-ORCH] Exploration planning failed: {e}. Falling back to stop.")
        return LLMExplorationPlan(
            tasks=[LLMExplorationTask(task_type="stop", description=f"Planning failed: {e}")],
            strategy="stop",
            coverage_estimate=0.0,
            reasoning=f"Planning failed: {e}",
        )

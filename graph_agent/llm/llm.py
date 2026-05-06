"""LLM factory and configuration.

Creates LLM instances based on environment configuration.
Similar to browser-use's llm/models.py pattern.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel

    from graph_agent.llm.zhipu.chat import ChatZhiPu


def _track_llm(llm: "BaseChatModel") -> "BaseChatModel":
    """Monkey-patch llm.ainvoke to record token usage globally.

    Similar to browser-use's TokenCost.register_llm().
    """
    from graph_agent.lib.token_tracker import get_global_tracker

    if getattr(llm, "_graph_agent_token_tracking", False):
        return llm

    original_ainvoke = llm.ainvoke
    model_name = getattr(llm, "model", "") or getattr(llm, "name", "")

    async def tracked_ainvoke(messages, output_format=None, **kwargs):
        result = await original_ainvoke(messages, output_format, **kwargs)
        tracker = get_global_tracker()
        usage = tracker.extract_usage(result, model=model_name)
        if usage:
            tracker.record(usage)
        return result

    # Use object.__setattr__ to bypass Pydantic model protection
    object.__setattr__(llm, "ainvoke", tracked_ainvoke)
    object.__setattr__(llm, "_graph_agent_token_tracking", True)
    return llm


def get_llm(use_thinking: bool = False) -> "BaseChatModel":
    """Create LLM instance from environment configuration.

    Args:
        use_thinking: Enable thinking mode for GLM (default: false)
                        This parameter directly controls the GLM thinking mode,
                        replacing the previous LLM_ENABLE_THINKING env var.

    Environment Variables:
        LLM_API_KEY: API key
        LLM_BASE_URL: Base URL
        LLM_MODEL: Model name
        OPENAI_API_KEY: OpenAI API key (fallback)

    Returns:
        BaseChatModel instance

    Raises:
        RuntimeError: If no valid configuration found
    """
    # Try generic LLM config first
    llm_key = os.getenv("LLM_API_KEY")
    llm_base = os.getenv("LLM_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "")

    if llm_key and llm_base and llm_model:
        return _create_from_config(llm_key, llm_base, llm_model, use_thinking)

    # Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from browser_use import ChatOpenAI

        return _track_llm(
            ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=openai_key,
                temperature=0.2,
            )
        )

    raise RuntimeError(
        "No valid LLM configuration found. "
        "Set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL or OPENAI_API_KEY."
    )


def _create_from_config(
    api_key: str,
    base_url: str,
    model: str,
    use_thinking: bool = False,
) -> "BaseChatModel":
    """Create LLM from configuration."""
    # Detect provider from base_url and model
    is_glm = "glm" in model.lower()
    is_bigmodel = "bigmodel.cn" in base_url.lower()
    is_openrouter = "openrouter" in base_url.lower()
    is_kimi = "kimi" in model.lower()

    if is_glm or is_bigmodel:
        # Use native GLM provider with thinking support
        from graph_agent.llm.zhipu.chat import ChatZhiPu

        return _track_llm(
            ChatZhiPu(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.2,
                thinking="enabled" if use_thinking else "disabled",
                clear_thinking=False,  # Preserve thinking across turns by default
                stream=False,  # Structured output (AgentOutput) must be non-stream for reliability
            )
        )
    if is_kimi:
        from graph_agent.llm.kimi.chat import ChatKimi

        return _track_llm(
            ChatKimi(
                model=model,
                api_key=api_key,
                base_url=base_url,
                thinking="enabled" if use_thinking else "disabled",
                stream=False,  # Structured output (AgentOutput) must be non-stream for reliability
            )
        )
    if is_openrouter:
        # Use browser-use's built-in ChatOpenRouter
        from browser_use.llm.openrouter.chat import ChatOpenRouter

        return _track_llm(
            ChatOpenRouter(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.2,
            )
        )

    # Generic OpenAI-compatible provider
    from browser_use import ChatOpenAI

    return _track_llm(
        ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )
    )
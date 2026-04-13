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


def get_llm(use_thinking: bool = True) -> "BaseChatModel":
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
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=0.6,
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
    
    if is_glm and is_bigmodel:
        # Use native GLM provider with thinking support
        from graph_agent.llm.zhipu.chat import ChatZhiPu
        return ChatZhiPu(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            thinking="enabled" if use_thinking else "disabled",
            clear_thinking=False,  # Preserve thinking across turns by default
        )
    
    if is_openrouter:
        # Use browser-use's built-in ChatOpenRouter
        from browser_use.llm.openrouter.chat import ChatOpenRouter
        return ChatOpenRouter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
        )
    
    # Generic OpenAI-compatible provider
    from browser_use import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.6,
    )


def create_zhipu_llm(
    model: str = "glm-5",
    api_key: str | None = None,
    enable_thinking: bool = False,
    preserve_thinking: bool = True,
    **kwargs,
) -> ChatZhiPu:
    """Create a ZhiPu LLM instance.
    
    Args:
        model: Model name (default: "glm-5")
        api_key: API key (reads from GLM_API_KEY or LLM_API_KEY env var)
        enable_thinking: Enable thinking mode
        preserve_thinking: Preserve thinking across turns (if enable_thinking=True)
        **kwargs: Additional arguments for ChatZhiPu
        
    Returns:
        ChatZhiPu instance
    """
    from graph_agent.llm.zhipu.chat import ChatZhiPu
    
    api_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
    return ChatZhiPu(
        model=model,
        api_key=api_key,
        thinking="enabled" if enable_thinking else "disabled",
        clear_thinking=not preserve_thinking,
        **kwargs,
    )

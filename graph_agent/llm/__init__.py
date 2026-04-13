"""LLM providers and factory for AutoTestAgent.

Structure mirrors browser-use's llm module:
- llm.py: Factory function (get_llm)
- glm/: GLM provider
- utils.py: Utility functions (ainvoke_structured)

OpenRouter is provided by browser-use directly.
"""

# Factory functions
from graph_agent.llm.llm import create_zhipu_llm, get_llm

# Providers
from graph_agent.llm.zhipu.chat import ChatZhiPu as ChatGLM

# Re-export browser-use's ChatOpenRouter
from browser_use.llm.openrouter.chat import ChatOpenRouter

# Utility functions
from graph_agent.llm.utils import (
    ainvoke_prompt,
    ainvoke_structured,
    get_format_instructions,
)

# View models (from browser-use for type compatibility)
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

__all__ = [
    # Factory
    "get_llm",
    "create_zhipu_llm",
    # Providers
    "ChatGLM",
    "ChatOpenRouter",
    # Utils
    "ainvoke_structured",
    "ainvoke_prompt",
    "get_format_instructions",
    # Views
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
]

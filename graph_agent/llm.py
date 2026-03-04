"""LLM configuration and factory."""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Build LLM from env: OPENAI_API_KEY (ChatOpenAI) or ANTHROPIC_API_KEY.
    Also supports LLM_API_KEY/LLM_BASE_URL/LLM_MODEL for OpenRouter/Gemini compatibility.
    """
    # 1. Try generic LLM_* config (e.g. OpenRouter)
    llm_key = os.getenv("LLM_API_KEY")
    llm_base = os.getenv("LLM_BASE_URL")
    llm_model = os.getenv("LLM_MODEL")

    if llm_key and llm_base and llm_model:
        try:
            # Try to use browser_use's ChatOpenAI first as it might have specific attributes
            from browser_use import ChatOpenAI
            return ChatOpenAI(
                model=llm_model,
                api_key=llm_key,
                base_url=llm_base,
                temperature=0.0,
            )
        except ImportError:
            # Fallback to langchain_openai if browser_use doesn't export it
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=llm_model,
                    api_key=llm_key,
                    base_url=llm_base,
                    temperature=0.0,
                )
            except ImportError:
                pass

    # 2. Try Standard OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from browser_use import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=api_key,
                temperature=0.0,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=api_key,
                temperature=0.0,
            )
    
    # 3. Try Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                api_key=anthropic_key,
                temperature=0.0,
            )
        except ImportError:
            raise RuntimeError(
                "ANTHROPIC_API_KEY set but langchain-anthropic not installed. "
                "Install it or set OPENAI_API_KEY."
            )
            
    raise RuntimeError(
        "No valid LLM configuration found. Set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL, "
        "or OPENAI_API_KEY, or ANTHROPIC_API_KEY in environment."
    )

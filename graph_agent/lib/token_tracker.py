"""Token usage tracking for LLM calls.

Tracks prompt tokens, completion tokens, and total tokens across all LLM invocations.
Supports both per-call tracking and cumulative session tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}, model={self.model})"
        )


@dataclass
class UsageStats:
    """Cumulative token usage statistics."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_embedding_calls: int = 0
    total_embedding_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_tokens_per_call(self) -> float:
        return self.total_tokens / max(1, self.total_calls)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / max(1, self.elapsed_seconds)

    def __str__(self) -> str:
        elapsed = self.elapsed_seconds
        return (
            f"UsageStats(calls={self.total_calls}, "
            f"prompt={self.total_prompt_tokens}, "
            f"completion={self.total_completion_tokens}, "
            f"total={self.total_tokens}, "
            f"embed_calls={self.total_embedding_calls}, "
            f"embed_tokens={self.total_embedding_tokens}, "
            f"avg={self.avg_tokens_per_call:.0f}/call, "
            f"elapsed={elapsed:.1f}s, "
            f"rate={self.tokens_per_second:.1f}tok/s)"
        )


class TokenUsageTracker:
    """Tracks token usage across multiple LLM calls.
    
    Usage:
        tracker = TokenUsageTracker()
        
        # In LLM call:
        result = await llm.ainvoke(...)
        usage = tracker.extract_usage(result)
        tracker.record(usage)
        
        # Get stats:
        stats = tracker.get_stats()
        logger.info("Token usage: %s", stats)
    """
    
    def __init__(self):
        self._usages: list[TokenUsage] = []
        self._stats = UsageStats()
        self._call_counter = 0
    
    def extract_usage(self, llm_result: Any, model: str = "") -> TokenUsage | None:
        """Extract token usage from LLM result.
        
        Supports browser-use ChatOpenAI and langchain OpenAI results.
        """
        usage = TokenUsage(model=model)
        
        try:
            # Try browser-use ChatInvokeCompletion structure
            if hasattr(llm_result, 'usage'):
                u = llm_result.usage
                if hasattr(u, 'prompt_tokens'):
                    usage.prompt_tokens = u.prompt_tokens
                if hasattr(u, 'completion_tokens'):
                    usage.completion_tokens = u.completion_tokens
                if hasattr(u, 'total_tokens'):
                    usage.total_tokens = u.total_tokens
            
            # Try raw response structure
            elif hasattr(llm_result, 'response'):
                response = llm_result.response
                if hasattr(response, 'usage'):
                    u = response.usage
                    usage.prompt_tokens = getattr(u, 'prompt_tokens', 0)
                    usage.completion_tokens = getattr(u, 'completion_tokens', 0)
                    usage.total_tokens = getattr(u, 'total_tokens', 0)
            
            # Try dict-like structure
            elif isinstance(llm_result, dict):
                if 'usage' in llm_result:
                    u = llm_result['usage']
                    usage.prompt_tokens = u.get('prompt_tokens', 0)
                    usage.completion_tokens = u.get('completion_tokens', 0)
                    usage.total_tokens = u.get('total_tokens', 0)
            
            # Try raw_completion structure (OpenAI compatible)
            if usage.total_tokens == 0 and hasattr(llm_result, 'raw_completion'):
                raw = llm_result.raw_completion  # type: ignore[union-attr]
                if hasattr(raw, 'usage'):
                    u = raw.usage
                    usage.prompt_tokens = getattr(u, 'prompt_tokens', 0)
                    usage.completion_tokens = getattr(u, 'completion_tokens', 0)
                    usage.total_tokens = getattr(u, 'total_tokens', 0)
            
            # Calculate total if not provided
            if usage.total_tokens == 0 and (usage.prompt_tokens or usage.completion_tokens):
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            
            if usage.total_tokens > 0:
                self._call_counter += 1
                usage.call_id = f"call_{self._call_counter}"
                return usage
            
            return None
            
        except Exception as e:
            logger.debug("Failed to extract token usage: %s", e)
            return None
    
    def record(self, usage: TokenUsage | None) -> None:
        """Record a token usage entry."""
        if usage is None or usage.total_tokens == 0:
            return

        self._usages.append(usage)
        self._stats.total_calls += 1
        self._stats.total_prompt_tokens += usage.prompt_tokens
        self._stats.total_completion_tokens += usage.completion_tokens
        self._stats.total_tokens += usage.total_tokens

        # Log each call
        logger.info(
            "[TokenUsage] #%d: prompt=%d, completion=%d, total=%d, model=%s",
            self._stats.total_calls,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.model or "unknown"
        )

    def record_embedding(self, total_tokens: int, model: str = "") -> None:
        """Record embedding token usage."""
        if total_tokens <= 0:
            return

        self._stats.total_embedding_calls += 1
        self._stats.total_embedding_tokens += total_tokens

        logger.info(
            "[TokenUsage] embedding #%d: total=%d, model=%s",
            self._stats.total_embedding_calls,
            total_tokens,
            model or "unknown"
        )
    
    def get_stats(self) -> UsageStats:
        """Get current usage statistics."""
        return self._stats
    
    def log_summary(self) -> None:
        """Log a summary of all token usage."""
        stats = self.get_stats()

        if stats.total_calls == 0 and stats.total_embedding_calls == 0:
            logger.info("[TokenUsage] No LLM calls recorded")
            return

        # Calculate cost (approximate, based on OpenAI pricing)
        # GPT-4: $0.03/1K prompt, $0.06/1K completion
        # GPT-3.5: $0.0015/1K prompt, $0.002/1K completion
        prompt_cost = stats.total_prompt_tokens * 0.00003  # Assume GPT-4 pricing
        completion_cost = stats.total_completion_tokens * 0.00006
        total_cost = prompt_cost + completion_cost

        logger.info(
            "[TokenUsage] Summary: LLM=%d calls (%d prompt + %d completion = %d tokens), "
            "Embedding=%d calls (%d tokens), ~$%.4f estimated cost, %.1fs elapsed",
            stats.total_calls,
            stats.total_prompt_tokens,
            stats.total_completion_tokens,
            stats.total_tokens,
            stats.total_embedding_calls,
            stats.total_embedding_tokens,
            total_cost,
            stats.elapsed_seconds
        )
    
    def reset(self) -> None:
        """Reset all tracking."""
        self._usages.clear()
        self._stats = UsageStats()
        self._call_counter = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        stats = self.get_stats()
        return {
            "total_calls": stats.total_calls,
            "total_prompt_tokens": stats.total_prompt_tokens,
            "total_completion_tokens": stats.total_completion_tokens,
            "total_tokens": stats.total_tokens,
            "total_embedding_calls": stats.total_embedding_calls,
            "total_embedding_tokens": stats.total_embedding_tokens,
            "avg_tokens_per_call": stats.avg_tokens_per_call,
            "elapsed_seconds": stats.elapsed_seconds,
            "tokens_per_second": stats.tokens_per_second,
        }


# Global tracker instance for session-wide tracking
_global_tracker: TokenUsageTracker | None = None


def get_global_tracker() -> TokenUsageTracker:
    """Get or create the global token usage tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenUsageTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global token usage tracker."""
    global _global_tracker
    _global_tracker = TokenUsageTracker()

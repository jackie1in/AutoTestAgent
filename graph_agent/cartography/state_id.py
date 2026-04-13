"""
State ID generation strategies for cartography.

This module provides different strategies for generating State IDs,
with different trade-offs for handling website changes.
"""

from __future__ import annotations

import hashlib
from enum import Enum


class StateIdStrategy(str, Enum):
    """Available state ID generation strategies."""
    
    # 仅基于菜单文本（当前默认）- 合并相同菜单的页面
    MENU_TEXT = "menu_text"
    
    # 基于 URL - 每个唯一 URL 一个 State
    URL_ONLY = "url_only"
    
    # 基于 URL + Fingerprint - 页面结构变化时创建新 State
    URL_FINGERPRINT = "url_fingerprint"
    
    # 基于完整 URL（含查询参数）- 最细粒度
    FULL_URL = "full_url"


def generate_state_id(
    url: str,
    fingerprint: str = "",
    menu_text: str = "",
    strategy: StateIdStrategy = StateIdStrategy.URL_FINGERPRINT,
) -> str:
    """Generate a deterministic State ID.
    
    Args:
        url: Page URL
        fingerprint: DOM fingerprint
        menu_text: Menu item text (if navigated via menu)
        strategy: ID generation strategy
        
    Returns:
        Unique state identifier
        
    Examples:
        >>> generate_state_id("/users", "abc123", "用户管理", StateIdStrategy.MENU_TEXT)
        'state:用户管理'
        
        >>> generate_state_id("/users", "abc123", strategy=StateIdStrategy.URL_FINGERPRINT)
        'state:a3f5e2b8c1d4e7f9'
    """
    match strategy:
        case StateIdStrategy.MENU_TEXT:
            if not menu_text:
                raise ValueError("menu_text required for MENU_TEXT strategy")
            return f"state:{menu_text.replace(' ', '-').lower()}"
            
        case StateIdStrategy.URL_ONLY:
            # Normalize URL
            normalized = url.rstrip('/').lower()
            return f"state:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"
            
        case StateIdStrategy.URL_FINGERPRINT:
            if not fingerprint:
                # Fallback to URL only if no fingerprint
                return generate_state_id(url, "", "", StateIdStrategy.URL_ONLY)
            key = f"{url}:{fingerprint}"
            return f"state:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            
        case StateIdStrategy.FULL_URL:
            # Include query parameters
            return f"state:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
            
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")


def is_same_state(
    state_id_a: str,
    state_id_b: str,
    strategy: StateIdStrategy = StateIdStrategy.URL_FINGERPRINT,
) -> bool:
    """Check if two state IDs represent the same state.
    
    For MENU_TEXT strategy, exact match.
    For URL_FINGERPRINT, exact match.
    """
    return state_id_a == state_id_b


def get_state_history_query(state_id: str) -> str:
    """Generate Cypher query to find historical versions of a state.
    
    For URL_FINGERPRINT strategy, find states with same URL prefix.
    """
    # Extract URL prefix from state_id (implementation depends on strategy)
    # This is a placeholder for future implementation
    return """
    MATCH (s:State)
    WHERE s.id STARTS WITH $url_prefix
    RETURN s ORDER BY s.first_discovered DESC
    """

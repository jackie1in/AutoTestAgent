"""
Tests for State ID generation strategies.

These tests verify that state IDs are generated correctly for different scenarios,
ensuring proper deduplication and change detection.
"""

import pytest
from graph_agent.cartography.state_id import (
    generate_state_id,
    StateIdStrategy,
)


class TestStateIdStrategies:
    """Test different state ID generation strategies."""
    
    def test_menu_text_strategy(self):
        """MENU_TEXT strategy generates ID based on menu text."""
        state_id = generate_state_id(
            url="/users",
            fingerprint="abc123",
            menu_text="用户管理",
            strategy=StateIdStrategy.MENU_TEXT,
        )
        assert state_id == "state:用户管理"
        
        # Same text should produce same ID regardless of URL/fingerprint
        state_id2 = generate_state_id(
            url="/users/v2",  # Different URL
            fingerprint="xyz789",  # Different fingerprint
            menu_text="用户管理",  # Same text
            strategy=StateIdStrategy.MENU_TEXT,
        )
        assert state_id == state_id2
    
    def test_menu_text_requires_text(self):
        """MENU_TEXT strategy requires menu_text parameter."""
        with pytest.raises(ValueError, match="menu_text required"):
            generate_state_id(
                url="/users",
                strategy=StateIdStrategy.MENU_TEXT,
            )
    
    def test_url_only_strategy(self):
        """URL_ONLY strategy generates ID based on URL."""
        state_id = generate_state_id(
            url="https://example.com/users",
            strategy=StateIdStrategy.URL_ONLY,
        )
        
        # Same URL should produce same ID
        state_id2 = generate_state_id(
            url="https://example.com/users",
            strategy=StateIdStrategy.URL_ONLY,
        )
        assert state_id == state_id2
        
        # Different URL should produce different ID
        state_id3 = generate_state_id(
            url="https://example.com/orders",
            strategy=StateIdStrategy.URL_ONLY,
        )
        assert state_id != state_id3
    
    def test_url_fingerprint_strategy(self):
        """URL_FINGERPRINT strategy generates ID based on both URL and fingerprint."""
        state_id = generate_state_id(
            url="/users",
            fingerprint="abc123",
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        
        # Same URL + fingerprint should produce same ID
        state_id2 = generate_state_id(
            url="/users",
            fingerprint="abc123",
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        assert state_id == state_id2
        
        # Different fingerprint should produce different ID (page changed)
        state_id3 = generate_state_id(
            url="/users",  # Same URL
            fingerprint="xyz789",  # Different fingerprint (page structure changed)
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        assert state_id != state_id3
        
        # Different URL should produce different ID
        state_id4 = generate_state_id(
            url="/orders",
            fingerprint="abc123",  # Same fingerprint
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        assert state_id != state_id4
    
    def test_url_fingerprint_fallback(self):
        """URL_FINGERPRINT falls back to URL_ONLY when no fingerprint."""
        state_id = generate_state_id(
            url="/users",
            fingerprint="",  # Empty fingerprint
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        
        # Should use URL_ONLY strategy
        expected_id = generate_state_id(
            url="/users",
            strategy=StateIdStrategy.URL_ONLY,
        )
        assert state_id == expected_id
    
    def test_full_url_strategy(self):
        """FULL_URL strategy includes query parameters."""
        state_id = generate_state_id(
            url="/users?page=1&filter=active",
            strategy=StateIdStrategy.FULL_URL,
        )
        
        # Different query params should produce different ID
        state_id2 = generate_state_id(
            url="/users?page=2&filter=active",
            strategy=StateIdStrategy.FULL_URL,
        )
        assert state_id != state_id2


class TestMenuRestructuringScenario:
    """Test the menu restructuring scenario."""
    
    def test_old_behavior_menu_restructure_same_text(self):
        """
        OLD BEHAVIOR: Menu text same → Same State ID → Data overwritten
        
        This test documents the problematic behavior where menu restructuring
        doesn't create new states because the text is the same.
        """
        # Version 1: Original menu structure
        v1_state_id = generate_state_id(
            url="/users/v1/list",
            fingerprint="layout-v1-abc123",
            menu_text="用户管理",
            strategy=StateIdStrategy.MENU_TEXT,
        )
        
        # Version 2: Menu restructured but text kept same
        v2_state_id = generate_state_id(
            url="/users/v2/dashboard",  # Different URL!
            fingerprint="layout-v2-xyz789",  # Different fingerprint!
            menu_text="用户管理",  # Same text
            strategy=StateIdStrategy.MENU_TEXT,
        )
        
        # PROBLEM: Same ID even though it's effectively a different page
        assert v1_state_id == v2_state_id
        assert v1_state_id == "state:用户管理"
    
    def test_new_behavior_menu_restructure_detection(self):
        """
        NEW BEHAVIOR: URL + Fingerprint → Different State IDs → Historical data preserved
        
        With URL_FINGERPRINT strategy, menu restructuring creates new states,
        preserving the old state history.
        """
        # Version 1: Original menu structure
        v1_state_id = generate_state_id(
            url="/users/v1/list",
            fingerprint="layout-v1-abc123",
            menu_text="用户管理",
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        
        # Version 2: Menu restructured but text kept same
        v2_state_id = generate_state_id(
            url="/users/v2/dashboard",  # Different URL
            fingerprint="layout-v2-xyz789",  # Different fingerprint
            menu_text="用户管理",  # Same text
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        
        # SOLUTION: Different IDs for different pages
        assert v1_state_id != v2_state_id
        
        # Both IDs are deterministic and will be consistent across runs
        v1_state_id_again = generate_state_id(
            url="/users/v1/list",
            fingerprint="layout-v1-abc123",
            strategy=StateIdStrategy.URL_FINGERPRINT,
        )
        assert v1_state_id == v1_state_id_again


class TestIdempotentGeneration:
    """Test that ID generation is deterministic (same input → same output)."""
    
    @pytest.mark.parametrize("strategy", [
        StateIdStrategy.MENU_TEXT,
        StateIdStrategy.URL_ONLY,
        StateIdStrategy.URL_FINGERPRINT,
        StateIdStrategy.FULL_URL,
    ])
    def test_deterministic_generation(self, strategy):
        """Same inputs should always produce same ID."""
        kwargs = {
            "url": "https://example.com/test",
            "fingerprint": "fp123",
            "menu_text": "Test Menu",
            "strategy": strategy,
        }
        
        # Remove irrelevant params for some strategies
        if strategy == StateIdStrategy.MENU_TEXT:
            del kwargs["fingerprint"]
        elif strategy == StateIdStrategy.URL_ONLY:
            del kwargs["fingerprint"]
            del kwargs["menu_text"]
        elif strategy == StateIdStrategy.FULL_URL:
            del kwargs["fingerprint"]
            del kwargs["menu_text"]
        
        id1 = generate_state_id(**kwargs)
        id2 = generate_state_id(**kwargs)
        id3 = generate_state_id(**kwargs)
        
        assert id1 == id2 == id3

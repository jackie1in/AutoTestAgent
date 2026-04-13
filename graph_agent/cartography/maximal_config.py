"""Maximal Exploration configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    """Configuration for maximal exploration.
    
    Attributes:
        min_completion_ratio: Target completion ratio (0.0-1.0)
        max_steps_per_zone: Maximum steps per zone (soft limit)
        max_time_per_zone_ms: Maximum time per zone in milliseconds
        max_parallel_zones: Number of zones to explore in parallel
        enable_smart_strategy: Enable intelligent strategy detection
        enable_recursive_menu: Enable recursive menu exploration
        enable_form_combination: Enable form field combination testing
        checkpoint_interval_s: Checkpoint save interval in seconds
        stale_threshold_steps: Steps without new state to consider stale
    """
    
    # Completion targets
    min_completion_ratio: float = 0.95
    
    # Limits (soft, for safety)
    max_steps_per_zone: int = 1000
    max_time_per_zone_ms: int = 600_000  # 10 minutes
    
    # Parallelism
    max_parallel_zones: int = 3
    
    # Feature flags
    enable_smart_strategy: bool = True
    enable_recursive_menu: bool = True
    enable_form_combination: bool = True
    
    # Checkpointing
    checkpoint_interval_s: int = 60
    
    # Stale detection
    stale_threshold_steps: int = 50
    
    # Completion weights
    element_click_weight: float = 0.4
    input_fill_weight: float = 0.2
    select_test_weight: float = 0.1
    form_submit_weight: float = 0.2
    modal_test_weight: float = 0.1
    
    @classmethod
    def small_site(cls) -> "ExplorationConfig":
        """Config for small sites (<20 pages)."""
        return cls(
            min_completion_ratio=0.98,
            max_parallel_zones=2,
            max_steps_per_zone=500,
        )
    
    @classmethod
    def medium_site(cls) -> "ExplorationConfig":
        """Config for medium sites (20-100 pages)."""
        return cls(
            min_completion_ratio=0.95,
            max_parallel_zones=4,
            max_steps_per_zone=1000,
        )
    
    @classmethod
    def large_site(cls) -> "ExplorationConfig":
        """Config for large sites (>100 pages)."""
        return cls(
            min_completion_ratio=0.90,
            max_parallel_zones=6,
            max_steps_per_zone=2000,
        )

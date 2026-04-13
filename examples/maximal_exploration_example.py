"""Maximal Exploration usage example.

This example demonstrates how to use the maximal exploration
features for comprehensive site coverage.
"""

from __future__ import annotations

import asyncio

from graph_agent.cartography import (
    ExplorationConfig,
)


async def example_basic():
    """Basic maximal exploration example."""
    print("=== Example 1: Basic Maximal Exploration ===\n")
    
    # Use default configuration
    config = ExplorationConfig(
        min_completion_ratio=0.95,  # Target 95% completion
        max_parallel_zones=3,
        enable_smart_strategy=True,
        enable_recursive_menu=True,
    )
    
    print("Configuration:")
    print(f"  - Target completion: {config.min_completion_ratio:.0%}")
    print(f"  - Max parallel zones: {config.max_parallel_zones}")
    print(f"  - Max steps per zone: {config.max_steps_per_zone}")
    print(f"  - Smart strategy: {config.enable_smart_strategy}")
    print()
    
    # Create orchestrator (would need actual Neo4j driver)
    # orchestrator = MaximalOrchestrator(driver, config)
    # result = await orchestrator.run_maximal_exploration(
    #     start_url="https://example.com",
    #     app_id="my_app",
    #     time_budget_hours=2,
    # )
    
    print("✅ Configuration ready (driver required for actual execution)")


async def example_site_configs():
    """Example using predefined site configurations."""
    print("\n=== Example 2: Site-Specific Configurations ===\n")
    
    configs = {
        "small": ExplorationConfig.small_site(),
        "medium": ExplorationConfig.medium_site(),
        "large": ExplorationConfig.large_site(),
    }
    
    for site_type, cfg in configs.items():
        print(f"{site_type.capitalize()} site:")
        print(f"  - Completion target: {cfg.min_completion_ratio:.0%}")
        print(f"  - Parallel zones: {cfg.max_parallel_zones}")
        print(f"  - Max steps/zone: {cfg.max_steps_per_zone}")
        print()


async def example_custom_weights():
    """Example with custom completion weights."""
    print("\n=== Example 3: Custom Completion Weights ===\n")
    
    config = ExplorationConfig(
        min_completion_ratio=0.90,
        # Customize completion calculation weights
        element_click_weight=0.5,    # More emphasis on clicking
        input_fill_weight=0.15,
        form_submit_weight=0.25,
        select_test_weight=0.05,
        modal_test_weight=0.05,
    )
    
    print("Custom weights:")
    print(f"  - Element clicks: {config.element_click_weight}")
    print(f"  - Input fills: {config.input_fill_weight}")
    print(f"  - Form submits: {config.form_submit_weight}")
    print(f"  - Select tests: {config.select_test_weight}")
    print(f"  - Modal tests: {config.modal_test_weight}")
    print()
    
    # Verify weights sum to 1.0
    total = (
        config.element_click_weight +
        config.input_fill_weight +
        config.form_submit_weight +
        config.select_test_weight +
        config.modal_test_weight
    )
    print(f"  Total weight: {total} (should be ~1.0)")


async def main():
    """Run all examples."""
    print("Maximal Exploration Examples")
    print("=" * 50)
    print()
    
    await example_basic()
    await example_site_configs()
    await example_custom_weights()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print()
    print("To run actual exploration:")
    print("  1. Ensure Neo4j is running: docker compose up -d neo4j")
    print("  2. Set up environment variables in .env")
    print("  3. Run with actual driver and browser session")


if __name__ == "__main__":
    asyncio.run(main())

"""Maximal exploration for comprehensive site coverage."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from graph_agent.cartography.completion_tracker import CompletionTracker
from graph_agent.cartography.maximal_config import ExplorationConfig
from graph_agent.cartography.react_explorer import ReActExplorer
from graph_agent.cartography.strategy_detector import StrategyDetector
from graph_agent.models import State, Zone

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession

logger = logging.getLogger(__name__)


@dataclass
class ZoneExplorationResult:
    """Result of exploring a single zone."""
    zone_id: str
    state: State | None
    steps_taken: int
    completion_ratio: float
    new_transitions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class MaximalExplorationResult:
    """Overall result of maximal exploration."""
    app_id: str
    states_discovered: int = 0
    zones_explored: int = 0
    total_steps: int = 0
    total_completion_ratio: float = 0.0
    zone_results: list[ZoneExplorationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class MaximalZoneExplorer:
    """Explores a single zone with completion-driven termination.
    
    Unlike standard ReActExplorer which stops at max_steps,
    this explorer continues until completion target is met.
    """
    
    def __init__(
        self,
        config: ExplorationConfig,
        browser_session: "BrowserSession",
        tracker: CompletionTracker,
    ):
        self._config = config
        self._session = browser_session
        self._tracker = tracker
        self._detector = StrategyDetector()
        self._base_explorer = ReActExplorer(
            max_steps=config.max_steps_per_zone,
            browser_session=browser_session,
        )
    
    async def explore(
        self,
        zone: Zone,
        start_state: State,
        page: Any,
    ) -> ZoneExplorationResult:
        """Explore a zone until completion target is met.
        
        Args:
            zone: Zone to explore
            start_state: Starting state
            page: Browser page
            
        Returns:
            Zone exploration result
        """
        zone_id = zone.id
        start_time = time.monotonic()
        steps = 0
        errors = []
        
        logger.info(f"[Maximal] Starting exploration of zone {zone_id}")
        
        try:
            # Detect strategy for this zone
            page_info = self._detector.analyze_page(page)
            strategy = self._detector.detect(page_info)
            task_sequence = self._detector.get_task_sequence(strategy)
            
            logger.info(f"[Maximal] Detected strategy: {strategy.name}, tasks: {task_sequence}")
            
            # Execute task sequence
            for task in task_sequence:
                if await self._should_stop(zone_id, steps):
                    logger.info(f"[Maximal] Stopping zone {zone_id} at step {steps}")
                    break
                
                task_steps = await self._execute_task(
                    task, zone, start_state, page, zone_id
                )
                steps += task_steps
                
                # Check completion after each task
                if self._tracker.is_complete(zone_id):
                    logger.info(f"[Maximal] Zone {zone_id} reached completion target")
                    break
            
            # Final exploration with base explorer to catch anything missed
            remaining_steps = self._config.max_steps_per_zone - steps
            if remaining_steps > 0 and not self._tracker.is_complete(zone_id):
                logger.info(f"[Maximal] Running fallback exploration for {remaining_steps} steps")
                result = await self._base_explorer.explore_page(
                    start_url=start_state.url,
                    max_steps=min(remaining_steps, 100),
                )
                steps += result.get('steps', 0)
            
        except Exception as e:
            logger.error(f"[Maximal] Error exploring zone {zone_id}: {e}")
            errors.append(str(e))
        
        duration = time.monotonic() - start_time
        completion = self._tracker.calculate_completion(zone_id)
        
        logger.info(
            f"[Maximal] Zone {zone_id} complete: "
            f"steps={steps}, completion={completion['overall_ratio']:.1%}, "
            f"duration={duration:.1f}s"
        )
        
        return ZoneExplorationResult(
            zone_id=zone_id,
            state=start_state,
            steps_taken=steps,
            completion_ratio=completion['overall_ratio'],
            errors=errors,
        )
    
    async def _should_stop(self, zone_id: str, current_steps: int) -> bool:
        """Check if exploration should stop."""
        # Check step limit
        if current_steps >= self._config.max_steps_per_zone:
            return True
        
        # Check completion target
        if self._tracker.is_complete(zone_id):
            return True
        
        return False
    
    async def _execute_task(
        self,
        task: str,
        zone: Zone,
        state: State,
        page: Any,
        zone_id: str,
    ) -> int:
        """Execute a specific exploration task.
        
        Returns:
            Number of steps taken
        """
        task_handlers = {
            'scan_elements': self._task_scan_elements,
            'click_all_interactive': self._task_click_interactive,
            'test_inputs': self._task_test_inputs,
            'scan_form_fields': self._task_scan_forms,
            'fill_required_fields': self._task_fill_required,
            'submit_form': self._task_submit_form,
            'scan_menu_items': self._task_scan_menu,
            'expand_all_levels': self._task_expand_menu,
            'scan_triggers': self._task_scan_modals,
            'open_each_modal': self._task_open_modals,
        }
        
        handler = task_handlers.get(task)
        if handler:
            try:
                return await handler(zone, state, page, zone_id)
            except Exception as e:
                logger.warning(f"[Maximal] Task {task} failed: {e}")
                return 0
        
        return 0
    
    # Task handlers (placeholders - would be implemented with actual browser interactions)
    async def _task_scan_elements(self, zone, state, page, zone_id: str) -> int:
        """Scan and count all interactive elements."""
        # Would query page for buttons, links, inputs, etc.
        self._tracker.register_elements(zone_id, {'clickable': 10})
        return 1
    
    async def _task_click_interactive(self, zone, state, page, zone_id: str) -> int:
        """Click all interactive elements."""
        # Would iterate through elements and click each
        for i in range(3):  # Placeholder
            self._tracker.record_click(zone_id, f"button-{i}")
        return 3
    
    async def _task_test_inputs(self, zone, state, page, zone_id: str) -> int:
        """Test all input fields."""
        for i in range(2):  # Placeholder
            self._tracker.record_fill(zone_id, f"input-{i}")
        return 2
    
    async def _task_scan_forms(self, zone, state, page, zone_id: str) -> int:
        """Scan and register all forms."""
        self._tracker.register_forms(zone_id, 2)
        return 1
    
    async def _task_fill_required(self, zone, state, page, zone_id: str) -> int:
        """Fill all required form fields."""
        return 2
    
    async def _task_submit_form(self, zone, state, page, zone_id: str) -> int:
        """Submit forms."""
        self._tracker.record_form_submit(zone_id, "form-1")
        return 1
    
    async def _task_scan_menu(self, zone, state, page, zone_id: str) -> int:
        """Scan menu structure."""
        return 1
    
    async def _task_expand_menu(self, zone, state, page, zone_id: str) -> int:
        """Expand all menu levels."""
        return 3
    
    async def _task_scan_modals(self, zone, state, page, zone_id: str) -> int:
        """Scan for modal triggers."""
        self._tracker.register_modals(zone_id, 2)
        return 1
    
    async def _task_open_modals(self, zone, state, page, zone_id: str) -> int:
        """Open and test each modal."""
        self._tracker.record_modal_open(zone_id, "modal-1")
        return 2


class MaximalOrchestrator:
    """Orchestrates maximal exploration across multiple zones."""
    
    def __init__(
        self,
        driver: Any,
        config: ExplorationConfig | None = None,
    ):
        self._driver = driver
        self._config = config or ExplorationConfig()
        self._tracker = CompletionTracker(self._config)
        self._detector = StrategyDetector()
    
    async def run_maximal_exploration(
        self,
        start_url: str,
        app_id: str,
        time_budget_hours: float = 24.0,
        browser_session: "BrowserSession" | None = None,
    ) -> MaximalExplorationResult:
        """Run maximal exploration.
        
        Args:
            start_url: Starting URL
            app_id: Application ID
            time_budget_hours: Total time budget in hours
            browser_session: Optional browser session
            
        Returns:
            Exploration result
        """
        start_time = time.monotonic()
        time_budget_seconds = time_budget_hours * 3600
        
        result = MaximalExplorationResult(app_id=app_id)
        
        logger.info(
            f"[Maximal] Starting maximal exploration for {app_id} "
            f"from {start_url}, budget={time_budget_hours}h"
        )
        
        try:
            # Phase 1: Initial discovery (menu + zones)
            zones = await self._discover_zones(start_url, browser_session)
            logger.info(f"[Maximal] Discovered {len(zones)} zones")
            
            # Phase 2: Parallel zone exploration
            await self._explore_zones_parallel(
                zones, browser_session, result, time_budget_seconds
            )
            
        except Exception as e:
            logger.error(f"[Maximal] Exploration failed: {e}")
            result.errors.append(str(e))
        
        # Calculate final metrics
        result.duration_seconds = time.monotonic() - start_time
        result.total_completion_ratio = self._calculate_overall_completion()
        
        logger.info(
            f"[Maximal] Exploration complete: "
            f"zones={result.zones_explored}, "
            f"states={result.states_discovered}, "
            f"steps={result.total_steps}, "
            f"completion={result.total_completion_ratio:.1%}, "
            f"duration={result.duration_seconds:.1f}s"
        )
        
        return result
    
    async def _discover_zones(
        self,
        start_url: str,
        browser_session: "BrowserSession" | None,
    ) -> list[Zone]:
        """Discover all zones in the application."""
        # Would use ZoneDiscoverer and MenuExtractor
        # For now, return placeholder
        return []
    
    async def _explore_zones_parallel(
        self,
        zones: list[Zone],
        browser_session: "BrowserSession" | None,
        result: MaximalExplorationResult,
        time_budget_seconds: float,
    ) -> None:
        """Explore zones in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self._config.max_parallel_zones)
        
        async def explore_with_limit(zone: Zone) -> ZoneExplorationResult | None:
            async with semaphore:
                # Check time budget
                if time.monotonic() - result.duration_seconds >= time_budget_seconds:
                    return None
                
                explorer = MaximalZoneExplorer(
                    self._config,
                    browser_session,  # type: ignore
                    self._tracker,
                )
                
                # Would get actual page and state
                zone_result = await explorer.explore(
                    zone=zone,
                    start_state=State(id=f"state-{zone.id}", url=""),
                    page=None,  # type: ignore
                )
                
                return zone_result
        
        # Run all zone explorations
        tasks = [explore_with_limit(zone) for zone in zones]
        zone_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for zone_result in zone_results:
            if isinstance(zone_result, Exception):
                result.errors.append(str(zone_result))
            elif zone_result:
                result.zone_results.append(zone_result)
                result.zones_explored += 1
                result.total_steps += zone_result.steps_taken
                if zone_result.state:
                    result.states_discovered += 1
    
    def _calculate_overall_completion(self) -> float:
        """Calculate overall completion across all zones."""
        all_metrics = self._tracker.get_all_metrics()
        if not all_metrics:
            return 0.0
        
        total = sum(m['overall_ratio'] for m in all_metrics.values())
        return total / len(all_metrics)

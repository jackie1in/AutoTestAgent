"""Strategy detection for maximal exploration."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any


class ExplorationStrategy(Enum):
    """Exploration strategies based on page characteristics."""
    DEFAULT = auto()
    FORM = auto()      # Form-heavy pages
    TABLE = auto()     # Table/grid pages
    MENU = auto()      # Menu/navigation pages
    MODAL = auto()     # Modal-heavy pages
    WIZARD = auto()    # Multi-step wizards


class StrategyDetector:
    """Detects the best exploration strategy for a page/zone."""
    
    # Thresholds for strategy detection
    FORM_FIELD_THRESHOLD = 5
    TABLE_ROW_THRESHOLD = 10
    MENU_ITEM_THRESHOLD = 5
    MODAL_COUNT_THRESHOLD = 3
    WIZARD_STEP_THRESHOLD = 3
    
    def detect(self, page_info: dict[str, Any]) -> ExplorationStrategy:
        """Detect strategy based on page characteristics.
        
        Args:
            page_info: Dict with page analysis results
            
        Returns:
            Detected exploration strategy
        """
        # Count elements by type
        form_fields = page_info.get('form_fields', 0)
        table_rows = page_info.get('table_rows', 0)
        menu_items = page_info.get('menu_items', 0)
        modal_triggers = page_info.get('modal_triggers', 0)
        wizard_steps = page_info.get('wizard_steps', 0)
        
        # Priority order matters
        if wizard_steps >= self.WIZARD_STEP_THRESHOLD:
            return ExplorationStrategy.WIZARD
        
        if form_fields >= self.FORM_FIELD_THRESHOLD:
            return ExplorationStrategy.FORM
        
        if table_rows >= self.TABLE_ROW_THRESHOLD:
            return ExplorationStrategy.TABLE
        
        if menu_items >= self.MENU_ITEM_THRESHOLD:
            return ExplorationStrategy.MENU
        
        if modal_triggers >= self.MODAL_COUNT_THRESHOLD:
            return ExplorationStrategy.MODAL
        
        return ExplorationStrategy.DEFAULT
    
    def get_task_sequence(self, strategy: ExplorationStrategy) -> list[str]:
        """Get recommended task sequence for a strategy.
        
        Args:
            strategy: The detected strategy
            
        Returns:
            List of task names in order
        """
        sequences = {
            ExplorationStrategy.DEFAULT: [
                'scan_elements',
                'click_all_interactive',
                'test_inputs',
            ],
            ExplorationStrategy.FORM: [
                'scan_form_fields',
                'fill_required_fields',
                'fill_optional_fields',
                'test_validation',
                'submit_form',
                'test_reset',
            ],
            ExplorationStrategy.TABLE: [
                'scan_table',
                'test_pagination',
                'test_sorting',
                'test_filtering',
                'test_row_actions',
                'test_bulk_actions',
            ],
            ExplorationStrategy.MENU: [
                'scan_menu_items',
                'expand_all_levels',
                'click_leaf_items',
                'test_back_navigation',
            ],
            ExplorationStrategy.MODAL: [
                'scan_triggers',
                'open_each_modal',
                'test_modal_actions',
                'test_close_methods',
            ],
            ExplorationStrategy.WIZARD: [
                'identify_steps',
                'complete_step_by_step',
                'test_back_navigation',
                'test_skip_options',
                'complete_final_step',
            ],
        }
        return sequences.get(strategy, sequences[ExplorationStrategy.DEFAULT])
    
    def analyze_page(self, page: Any) -> dict[str, Any]:
        """Analyze page to extract characteristics.
        
        Args:
            page: Browser page object
            
        Returns:
            Dict with page analysis
        """
        # This would normally query the page DOM
        # For now, return placeholder structure
        return {
            'form_fields': 0,
            'table_rows': 0,
            'menu_items': 0,
            'modal_triggers': 0,
            'wizard_steps': 0,
            'total_interactive': 0,
        }

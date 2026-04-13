"""Completion tracking for maximal exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_agent.cartography.maximal_config import ExplorationConfig


@dataclass
class ElementMetrics:
    """Metrics for element interactions."""
    total: int = 0
    clicked: int = 0
    filled: int = 0
    selected: int = 0


@dataclass
class FormMetrics:
    """Metrics for form interactions."""
    total: int = 0
    submitted: int = 0
    reset_tested: int = 0
    validation_tested: int = 0


@dataclass
class ModalMetrics:
    """Metrics for modal interactions."""
    total: int = 0
    opened: int = 0
    closed: int = 0
    actions_tested: int = 0


@dataclass
class CompletionMetrics:
    """Complete metrics for a zone."""
    elements: ElementMetrics = field(default_factory=ElementMetrics)
    forms: FormMetrics = field(default_factory=FormMetrics)
    modals: ModalMetrics = field(default_factory=ModalMetrics)
    
    # Track specific interactions
    clicked_selectors: set[str] = field(default_factory=set)
    filled_inputs: set[str] = field(default_factory=set)
    tested_selects: set[str] = field(default_factory=set)
    submitted_forms: set[str] = field(default_factory=set)
    tested_modals: set[str] = field(default_factory=set)


class CompletionTracker:
    """Tracks exploration completion metrics."""
    
    def __init__(self, config: "ExplorationConfig") -> None:
        self._config = config
        self._metrics: dict[str, CompletionMetrics] = {}
    
    def get_or_create(self, zone_id: str) -> CompletionMetrics:
        """Get or create metrics for a zone."""
        if zone_id not in self._metrics:
            self._metrics[zone_id] = CompletionMetrics()
        return self._metrics[zone_id]
    
    def record_click(self, zone_id: str, selector: str) -> None:
        """Record an element click."""
        m = self.get_or_create(zone_id)
        m.elements.clicked += 1
        m.clicked_selectors.add(selector)
    
    def record_fill(self, zone_id: str, selector: str) -> None:
        """Record an input fill."""
        m = self.get_or_create(zone_id)
        m.elements.filled += 1
        m.filled_inputs.add(selector)
    
    def record_select(self, zone_id: str, selector: str) -> None:
        """Record a select/dropdown test."""
        m = self.get_or_create(zone_id)
        m.elements.selected += 1
        m.tested_selects.add(selector)
    
    def record_form_submit(self, zone_id: str, form_selector: str) -> None:
        """Record a form submission."""
        m = self.get_or_create(zone_id)
        m.forms.submitted += 1
        m.submitted_forms.add(form_selector)
    
    def record_form_reset(self, zone_id: str, form_selector: str) -> None:
        """Record a form reset test."""
        m = self.get_or_create(zone_id)
        m.forms.reset_tested += 1
    
    def record_modal_open(self, zone_id: str, modal_selector: str) -> None:
        """Record a modal open."""
        m = self.get_or_create(zone_id)
        m.modals.opened += 1
        m.tested_modals.add(modal_selector)
    
    def record_modal_close(self, zone_id: str) -> None:
        """Record a modal close."""
        m = self.get_or_create(zone_id)
        m.modals.closed += 1
    
    def record_modal_action(self, zone_id: str) -> None:
        """Record a modal action test."""
        m = self.get_or_create(zone_id)
        m.modals.actions_tested += 1
    
    def register_elements(self, zone_id: str, element_counts: dict[str, int]) -> None:
        """Register total element counts for a zone."""
        m = self.get_or_create(zone_id)
        m.elements.total = element_counts.get('clickable', 0)
    
    def register_forms(self, zone_id: str, form_count: int) -> None:
        """Register total form count."""
        m = self.get_or_create(zone_id)
        m.forms.total = form_count
    
    def register_modals(self, zone_id: str, modal_count: int) -> None:
        """Register total modal count."""
        m = self.get_or_create(zone_id)
        m.modals.total = modal_count
    
    def calculate_completion(self, zone_id: str) -> dict[str, Any]:
        """Calculate completion ratio for a zone.
        
        Returns:
            Dict with completion metrics and overall ratio.
        """
        m = self.get_or_create(zone_id)
        cfg = self._config
        
        # Calculate individual ratios
        click_ratio = min(1.0, len(m.clicked_selectors) / max(1, m.elements.total))
        fill_ratio = min(1.0, len(m.filled_inputs) / max(1, m.elements.total * 0.3))  # Assume 30% are inputs
        select_ratio = min(1.0, len(m.tested_selects) / max(1, m.elements.total * 0.1))  # Assume 10% are selects
        submit_ratio = min(1.0, m.forms.submitted / max(1, m.forms.total))
        modal_ratio = min(1.0, m.modals.opened / max(1, m.modals.total))
        
        # Weighted completion
        total = (
            cfg.element_click_weight * click_ratio +
            cfg.input_fill_weight * fill_ratio +
            cfg.select_test_weight * select_ratio +
            cfg.form_submit_weight * submit_ratio +
            cfg.modal_test_weight * modal_ratio
        )
        
        return {
            'zone_id': zone_id,
            'overall_ratio': total,
            'click_ratio': click_ratio,
            'fill_ratio': fill_ratio,
            'select_ratio': select_ratio,
            'submit_ratio': submit_ratio,
            'modal_ratio': modal_ratio,
            'details': {
                'elements_clicked': len(m.clicked_selectors),
                'elements_total': m.elements.total,
                'forms_submitted': m.forms.submitted,
                'forms_total': m.forms.total,
                'modals_tested': len(m.tested_modals),
                'modals_total': m.modals.total,
            }
        }
    
    def is_complete(self, zone_id: str) -> bool:
        """Check if zone has reached target completion."""
        result = self.calculate_completion(zone_id)
        return result['overall_ratio'] >= self._config.min_completion_ratio
    
    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get completion metrics for all zones."""
        return {
            zone_id: self.calculate_completion(zone_id)
            for zone_id in self._metrics.keys()
        }

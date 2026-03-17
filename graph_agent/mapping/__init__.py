"""Mapping: parse browser-use history into graph step (selector, action, semantic_label, param_name, action_value, element)."""

from graph_agent.mapping.parser import parse_browser_use_step, parse_browser_use_step_lite

__all__ = ["parse_browser_use_step", "parse_browser_use_step_lite"]

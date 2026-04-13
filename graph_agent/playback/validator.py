"""
Playback Validator: Validates playback results against checkpoints.

This module provides validation capabilities to verify that playback operations
achieved their intended business purpose, not just that they didn't error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from playwright.async_api import Page, expect

from graph_agent.models import (
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointTiming,
)

logger = logging.getLogger(__name__)


class ValidationResultType(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class CheckpointValidationResult:
    """Result of validating a single checkpoint."""
    checkpoint_id: str
    checkpoint_layer: CheckpointLayer
    result: ValidationResultType
    message: str = ""
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaybackValidationReport:
    """Complete validation report for a playback session."""
    success: bool
    total_checkpoints: int
    passed: int
    failed: int
    skipped: int
    errors: int
    results: list[CheckpointValidationResult]
    summary: str = ""


class CheckpointValidator:
    """Validates checkpoints during and after playback."""

    def __init__(self, page: Page):
        self._page = page

    async def validate(
        self,
        checkpoint: Checkpoint,
        context: dict[str, Any] | None = None,
    ) -> CheckpointValidationResult:
        """Validate a single checkpoint."""
        import time
        start = time.monotonic()
        
        try:
            rule = checkpoint.get_rule()
            
            match checkpoint.rule_type:
                case "url_match":
                    result = await self._validate_url_match(rule)
                case "element_exists":
                    result = await self._validate_element_exists(rule)
                case "validation_error":
                    result = await self._validate_validation_error(rule)
                case "field_value":
                    result = await self._validate_field_value(rule)
                case "toast":
                    result = await self._validate_toast(rule)
                case "entity_exists":
                    result = await self._validate_entity_exists(rule, context)
                case "entity_created":
                    result = await self._validate_entity_created(rule, context)
                case "page_matches_intent":
                    result = await self._validate_page_matches_intent(rule)
                case _:
                    result = CheckpointValidationResult(
                        checkpoint_id=checkpoint.id,
                        checkpoint_layer=checkpoint.layer,
                        result=ValidationResultType.SKIP,
                        message=f"Unknown rule_type: {checkpoint.rule_type}",
                    )
            
            duration = (time.monotonic() - start) * 1000
            result.duration_ms = duration
            
            # Apply expect logic: should_pass vs should_fail
            if checkpoint.expect == CheckpointExpect.SHOULD_FAIL:
                # Invert the result for negative tests
                if result.result == ValidationResultType.PASS:
                    result.result = ValidationResultType.FAIL
                    result.message = f"Expected to fail but passed: {result.message}"
                elif result.result == ValidationResultType.FAIL:
                    result.result = ValidationResultType.PASS
                    result.message = f"Correctly failed as expected: {result.message}"
            
            return result
            
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return CheckpointValidationResult(
                checkpoint_id=checkpoint.id,
                checkpoint_layer=checkpoint.layer,
                result=ValidationResultType.ERROR,
                message=f"Validation error: {e}",
                duration_ms=duration,
            )

    async def _validate_url_match(self, rule: dict) -> CheckpointValidationResult:
        """Validate current URL matches expected pattern."""
        expected_url = rule.get("url", "")
        current_url = self._page.url
        
        if rule.get("exact", False):
            matches = current_url == expected_url
        else:
            # Allow partial match
            matches = expected_url in current_url
        
        if matches:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.STRUCTURAL,
                result=ValidationResultType.PASS,
                message=f"URL matches: {current_url}",
            )
        else:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.STRUCTURAL,
                result=ValidationResultType.FAIL,
                message=f"URL mismatch: expected {expected_url}, got {current_url}",
            )

    async def _validate_element_exists(self, rule: dict) -> CheckpointValidationResult:
        """Validate element exists (or not) on page."""
        selector = rule.get("selector", "")
        should_exist = rule.get("exists", True)
        
        try:
            locator = self._page.locator(selector)
            count = await locator.count()
            exists = count > 0
            
            if exists == should_exist:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.STRUCTURAL,
                    result=ValidationResultType.PASS,
                    message=f"Element {selector} exists={exists} as expected",
                )
            else:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.STRUCTURAL,
                    result=ValidationResultType.FAIL,
                    message=f"Element {selector}: expected exists={should_exist}, got {exists}",
                )
        except Exception as e:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.STRUCTURAL,
                result=ValidationResultType.ERROR,
                message=f"Error checking element {selector}: {e}",
            )

    async def _validate_validation_error(self, rule: dict) -> CheckpointValidationResult:
        """Validate validation error message visibility/content."""
        selector = rule.get("selector", "")
        should_be_visible = rule.get("visible", True)
        message_contains = rule.get("message_contains", "")
        
        try:
            locator = self._page.locator(selector)
            is_visible = await locator.is_visible()
            
            if is_visible != should_be_visible:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.DATA,
                    result=ValidationResultType.FAIL,
                    message=f"Validation error visibility mismatch: expected {should_be_visible}, got {is_visible}",
                )
            
            if should_be_visible and message_contains:
                text = await locator.inner_text()
                if message_contains not in text:
                    return CheckpointValidationResult(
                        checkpoint_id="",
                        checkpoint_layer=CheckpointLayer.DATA,
                        result=ValidationResultType.FAIL,
                        message=f"Validation error message doesn't contain '{message_contains}': {text}",
                    )
            
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.DATA,
                result=ValidationResultType.PASS,
                message="Validation error check passed",
            )
        except Exception as e:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.DATA,
                result=ValidationResultType.ERROR,
                message=f"Error checking validation error: {e}",
            )

    async def _validate_field_value(self, rule: dict) -> CheckpointValidationResult:
        """Validate input field has expected value."""
        selector = rule.get("selector", "")
        expected_value = rule.get("value", "")
        
        try:
            locator = self._page.locator(selector)
            actual_value = await locator.input_value()
            
            if actual_value == expected_value:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.DATA,
                    result=ValidationResultType.PASS,
                    message=f"Field value matches: {expected_value}",
                )
            else:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.DATA,
                    result=ValidationResultType.FAIL,
                    message=f"Field value mismatch: expected '{expected_value}', got '{actual_value}'",
                )
        except Exception as e:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.DATA,
                result=ValidationResultType.ERROR,
                message=f"Error checking field value: {e}",
            )

    async def _validate_toast(self, rule: dict) -> CheckpointValidationResult:
        """Validate toast/notification message appears."""
        message_contains = rule.get("message_contains", "")
        timeout_ms = rule.get("timeout_ms", 5000)
        
        # Common toast selectors
        toast_selectors = [
            ".ant-message .ant-message-notice-content",
            ".el-message .el-message__content",
            ".toast",
            ".notification",
            "[role='alert']",
            ".snackbar",
        ]
        
        try:
            found = False
            matched_text = ""
            
            for selector in toast_selectors:
                try:
                    locator = self._page.locator(selector)
                    await expect(locator).to_be_visible(timeout=timeout_ms)
                    text = await locator.inner_text()
                    if message_contains in text:
                        found = True
                        matched_text = text
                        break
                except Exception:
                    continue
            
            if found:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.BEHAVIORAL,
                    result=ValidationResultType.PASS,
                    message=f"Toast found with message: {matched_text}",
                )
            else:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.BEHAVIORAL,
                    result=ValidationResultType.FAIL,
                    message=f"Toast with message containing '{message_contains}' not found",
                )
        except Exception as e:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.BEHAVIORAL,
                result=ValidationResultType.ERROR,
                message=f"Error checking toast: {e}",
            )

    async def _validate_entity_exists(
        self, rule: dict, context: dict[str, Any] | None
    ) -> CheckpointValidationResult:
        """Validate entity exists (requires external verification)."""
        entity_id = rule.get("entity_id", "")
        match_fields = rule.get("match_fields", {})
        
        # This would typically query the database or API
        # For now, we mark it as requiring external validation
        return CheckpointValidationResult(
            checkpoint_id="",
            checkpoint_layer=CheckpointLayer.ENTITY,
            result=ValidationResultType.SKIP,
            message=f"Entity existence check requires external validation: {entity_id}",
            details={"entity_id": entity_id, "match_fields": match_fields},
        )

    async def _validate_entity_created(
        self, rule: dict, context: dict[str, Any] | None
    ) -> CheckpointValidationResult:
        """Validate entity was created (requires external verification)."""
        entity_id = rule.get("entity_id", "")
        created_fields = rule.get("created_fields", [])
        
        # This would typically query the database or API
        return CheckpointValidationResult(
            checkpoint_id="",
            checkpoint_layer=CheckpointLayer.ENTITY,
            result=ValidationResultType.SKIP,
            message=f"Entity creation check requires external validation: {entity_id}",
            details={"entity_id": entity_id, "created_fields": created_fields},
        )

    async def _validate_page_matches_intent(self, rule: dict) -> CheckpointValidationResult:
        """Validate current page matches expected intent (semantic check)."""
        expected_intent = rule.get("intent", "")
        keywords = rule.get("keywords", [])
        
        try:
            # Get page title and visible text
            title = await self._page.title()
            body_text = await self._page.locator("body").inner_text()
            content = f"{title} {body_text}".lower()
            
            # Check for intent keywords
            matched_keywords = [kw for kw in keywords if kw.lower() in content]
            match_ratio = len(matched_keywords) / len(keywords) if keywords else 0
            
            threshold = rule.get("threshold", 0.5)
            if match_ratio >= threshold:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.SEMANTIC,
                    result=ValidationResultType.PASS,
                    message=f"Page matches intent '{expected_intent}': {matched_keywords}",
                )
            else:
                return CheckpointValidationResult(
                    checkpoint_id="",
                    checkpoint_layer=CheckpointLayer.SEMANTIC,
                    result=ValidationResultType.FAIL,
                    message=f"Page doesn't match intent '{expected_intent}': matched {matched_keywords}/{keywords}",
                )
        except Exception as e:
            return CheckpointValidationResult(
                checkpoint_id="",
                checkpoint_layer=CheckpointLayer.SEMANTIC,
                result=ValidationResultType.ERROR,
                message=f"Error checking page intent: {e}",
            )


class PlaybackValidator:
    """High-level playback validation orchestrator."""

    def __init__(
        self,
        page: Page,
        checkpoints: list[Checkpoint],
        on_checkpoint_result: Callable[[CheckpointValidationResult], Awaitable[None] | None] | None = None,
    ):
        self._page = page
        self._checkpoints = checkpoints
        self._on_checkpoint_result = on_checkpoint_result
        self._checkpoint_validator = CheckpointValidator(page)

    async def validate_all(
        self,
        timing: CheckpointTiming | None = None,
        context: dict[str, Any] | None = None,
    ) -> PlaybackValidationReport:
        """Validate all checkpoints with optional timing filter."""
        results: list[CheckpointValidationResult] = []
        
        filtered_checkpoints = [
            cp for cp in self._checkpoints
            if timing is None or cp.timing == timing
        ]
        
        for checkpoint in filtered_checkpoints:
            result = await self._checkpoint_validator.validate(checkpoint, context)
            results.append(result)
            
            if self._on_checkpoint_result:
                cb_result = self._on_checkpoint_result(result)
                if cb_result and hasattr(cb_result, "__await__"):
                    await cb_result
        
        # Calculate summary
        passed = sum(1 for r in results if r.result == ValidationResultType.PASS)
        failed = sum(1 for r in results if r.result == ValidationResultType.FAIL)
        skipped = sum(1 for r in results if r.result == ValidationResultType.SKIP)
        errors = sum(1 for r in results if r.result == ValidationResultType.ERROR)
        
        summary = (
            f"Validation complete: {passed} passed, {failed} failed, "
            f"{skipped} skipped, {errors} errors"
        )
        
        return PlaybackValidationReport(
            success=failed == 0 and errors == 0,
            total_checkpoints=len(filtered_checkpoints),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            results=results,
            summary=summary,
        )

    async def validate_before_action(
        self, context: dict[str, Any] | None = None
    ) -> PlaybackValidationReport:
        """Validate CHECK_BEFORE checkpoints."""
        return await self.validate_all(CheckpointTiming.BEFORE, context)

    async def validate_after_action(
        self, context: dict[str, Any] | None = None
    ) -> PlaybackValidationReport:
        """Validate CHECK_AFTER checkpoints."""
        return await self.validate_all(CheckpointTiming.AFTER, context)


async def validate_playback_with_checkpoints(
    page: Page,
    checkpoints: list[Checkpoint],
    test_context: dict[str, Any] | None = None,
) -> PlaybackValidationReport:
    """Convenience function to validate playback with checkpoints."""
    validator = PlaybackValidator(page, checkpoints)
    return await validator.validate_all(context=test_context)

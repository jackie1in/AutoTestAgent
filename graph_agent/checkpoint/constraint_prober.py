"""Probes form field constraints using browser-use Page.evaluate()."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.actor.page import Page

from graph_agent.models import FieldConstraint

logger = logging.getLogger(__name__)


class ConstraintProber:
    """Probes form field constraints by testing boundary values."""

    async def probe(self, page: "Page", zone_selector: str) -> list[FieldConstraint]:
        """Probe all form fields in a zone to discover constraints.

        Discovers: required, min/max length, pattern, input type via HTML5 attributes.
        """
        raw = await page.evaluate(
            """(zoneSelector) => {
                const zone = zoneSelector ? document.querySelector(zoneSelector) : document;
                if (!zone) return JSON.stringify([]);
                const inputs = zone.querySelectorAll('input, select, textarea');
                return JSON.stringify(Array.from(inputs).map(el => ({
                    selector: el.id ? '#' + el.id : (el.name ? '[name="' + el.name + '"]' : el.tagName.toLowerCase()),
                    fieldName: el.name || el.id || el.placeholder || '',
                    inputType: el.type || 'text',
                    required: el.required || el.hasAttribute('aria-required'),
                    minLength: el.minLength > 0 ? el.minLength : null,
                    maxLength: el.maxLength > 0 && el.maxLength < 524288 ? el.maxLength : null,
                    min: el.min !== '' ? parseFloat(el.min) : null,
                    max: el.max !== '' ? parseFloat(el.max) : null,
                    pattern: el.pattern || null,
                    step: el.step !== '' ? parseFloat(el.step) : null,
                    placeholder: el.placeholder || '',
                })));
            }""",
            zone_selector or "body",
        )
        fields_data = json.loads(raw) if isinstance(raw, str) else raw

        constraints: list[FieldConstraint] = []
        for fd in fields_data:
            if not fd.get("fieldName"):
                continue
            constraints.append(
                FieldConstraint(
                    id=f"fc:{fd['selector']}",
                    selector=fd.get("selector", ""),
                    field_name=fd.get("fieldName", ""),
                    input_type=fd.get("inputType", "text"),
                    required=fd.get("required", False),
                    min_length=fd.get("minLength"),
                    max_length=fd.get("maxLength"),
                    min_value=fd.get("min"),
                    max_value=fd.get("max"),
                    pattern=fd.get("pattern"),
                    step=fd.get("step"),
                )
            )

        logger.info("Probed %d fields in zone %s", len(constraints), zone_selector)
        return constraints

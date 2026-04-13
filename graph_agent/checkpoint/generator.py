from __future__ import annotations

import json
import logging
from typing import Any

from graph_agent.models import (
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointTiming,
    Severity,
)

logger = logging.getLogger(__name__)


class CheckpointGenerator:
    """Generates checkpoints from DOM diffs and transition context."""

    def from_url_change(self, from_url: str, to_url: str, transition_id: str) -> Checkpoint:
        """Generate a URL match checkpoint for a navigation transition."""
        return Checkpoint(
            id=f"cp:{transition_id}:url",
            layer=CheckpointLayer.STRUCTURAL,
            timing=CheckpointTiming.AFTER,
            expect=CheckpointExpect.SHOULD_PASS,
            severity=Severity.CRITICAL,
            rule_type="url_match",
            rule=json.dumps({"expected_url_contains": to_url}),
            description=f"Navigation should reach {to_url}",
        )

    def from_dom_diff(self, before: dict[str, Any], after: dict[str, Any], transition_id: str) -> list[Checkpoint]:
        """Generate checkpoints from DOM snapshot diff."""
        # TODO: Implement DOM diff checkpoint generation
        checkpoints: list[Checkpoint] = []
        return checkpoints

    def for_entity_precondition(
        self, entity_id: str, match_fields: dict[str, Any], intent_id: str
    ) -> Checkpoint:
        """Generate entity existence precondition checkpoint."""
        return Checkpoint(
            id=f"cp:{intent_id}:pre:{entity_id}",
            layer=CheckpointLayer.ENTITY,
            timing=CheckpointTiming.BEFORE,
            expect=CheckpointExpect.SHOULD_PASS,
            severity=Severity.CRITICAL,
            rule_type="entity_exists",
            rule=json.dumps({"entity_id": entity_id, "match_fields": match_fields}),
            description=f"Entity {entity_id} must exist before operation",
        )

    def for_entity_postcondition(
        self, entity_id: str, created_fields: list[str], intent_id: str
    ) -> Checkpoint:
        """Generate entity creation postcondition checkpoint."""
        return Checkpoint(
            id=f"cp:{intent_id}:post:{entity_id}",
            layer=CheckpointLayer.ENTITY,
            timing=CheckpointTiming.AFTER,
            expect=CheckpointExpect.SHOULD_PASS,
            severity=Severity.CRITICAL,
            rule_type="entity_created",
            rule=json.dumps({"entity_id": entity_id, "created_fields": created_fields}),
            description=f"Entity {entity_id} should be created",
        )

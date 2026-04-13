from __future__ import annotations

import json
import logging

from graph_agent.models import FieldConstraint, TestCase, TestCaseCategory

logger = logging.getLogger(__name__)


class TestDeriver:
    """Derives boundary and equivalence class test cases from field constraints."""

    def derive(self, constraints: list[FieldConstraint], transition_id: str) -> list[TestCase]:
        """Derive test cases from field constraints."""
        test_cases: list[TestCase] = []

        for fc in constraints:
            if fc.required:
                test_cases.append(
                    TestCase(
                        id=f"tc:{transition_id}:{fc.field_name}:required",
                        name=f"{fc.field_name}-必填校验",
                        category=TestCaseCategory.REQUIRED,
                        description=f"Field {fc.field_name} should be required",
                        field_overrides=json.dumps({fc.selector: ""}),
                    )
                )

            if fc.max_length is not None:
                test_cases.append(
                    TestCase(
                        id=f"tc:{transition_id}:{fc.field_name}:maxlen",
                        name=f"{fc.field_name}-超最大长度",
                        category=TestCaseCategory.BOUNDARY,
                        description=f"Input exceeding max length {fc.max_length}",
                        field_overrides=json.dumps({fc.selector: "a" * (fc.max_length + 1)}),
                    )
                )

            if fc.min_length is not None and fc.min_length > 0:
                test_cases.append(
                    TestCase(
                        id=f"tc:{transition_id}:{fc.field_name}:minlen",
                        name=f"{fc.field_name}-低于最小长度",
                        category=TestCaseCategory.BOUNDARY,
                        description=f"Input below min length {fc.min_length}",
                        field_overrides=json.dumps({fc.selector: "a" * max(0, fc.min_length - 1)}),
                    )
                )

        return test_cases

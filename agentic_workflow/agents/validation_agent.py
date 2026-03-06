"""
Validation agent – rigorously evaluates workflow step outputs against
structured rules and produces detailed validation reports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_workflow.agents.base_agent import BaseAgent, LLMResponse
from agentic_workflow.prompts.system_prompts import SystemPrompts
from agentic_workflow.prompts.task_prompts import TaskPrompts

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of evaluating a single validation rule."""

    rule_id: str
    field_path: str
    rule_type: str
    passed: bool
    actual_value: Any
    expected_value: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "field": self.field_path,
            "rule_type": self.rule_type,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """Complete validation report for a step output."""

    valid: bool
    score: int  # 0-100
    rule_results: list[RuleResult] = field(default_factory=list)
    summary: str = ""
    suggested_corrections: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationReport":
        rules = [
            RuleResult(
                rule_id=r.get("rule_id", ""),
                field_path=r.get("field", ""),
                rule_type=r.get("rule_type", ""),
                passed=r.get("passed", False),
                actual_value=r.get("actual_value"),
                expected_value=r.get("expected_value"),
                message=r.get("message", ""),
            )
            for r in data.get("rule_results", [])
        ]
        return cls(
            valid=data.get("valid", False),
            score=data.get("score", 0),
            rule_results=rules,
            summary=data.get("summary", ""),
            suggested_corrections=data.get("suggested_corrections", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "rule_results": [r.to_dict() for r in self.rule_results],
            "summary": self.summary,
            "suggested_corrections": self.suggested_corrections,
        }

    @property
    def failed_rules(self) -> list[RuleResult]:
        return [r for r in self.rule_results if not r.passed]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.rule_results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.rule_results if not r.passed)


class ValidationAgent(BaseAgent):
    """
    LLM-powered validation agent with an embedded rule engine.

    For simple, deterministic rules (required, type, min_value, max_value,
    min_length, max_length, enum, pattern) the embedded rule engine evaluates
    directly without an LLM call, keeping latency low and costs minimal.
    Complex/custom rules are delegated to the LLM.
    """

    _DETERMINISTIC_RULES = frozenset(
        {"required", "type", "min_value", "max_value", "min_length", "max_length", "enum", "pattern"}
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(system_prompt=SystemPrompts.VALIDATOR, **kwargs)

    def validate(
        self,
        output: dict[str, Any],
        rules: list[dict[str, Any]],
        use_llm_for_complex: bool = True,
    ) -> ValidationReport:
        """
        Validate ``output`` against ``rules``.

        Deterministic rules are evaluated by the built-in engine; any
        remaining ``custom`` rules are forwarded to the LLM.

        Parameters
        ----------
        output:
            The step output dict to validate.
        rules:
            List of rule dicts (field, rule, value).
        use_llm_for_complex:
            If True (default), delegate custom rules to the LLM.

        Returns
        -------
        ValidationReport
        """
        if not rules:
            return ValidationReport(valid=True, score=100, summary="No rules to validate")

        deterministic_rules = [r for r in rules if r.get("rule") in self._DETERMINISTIC_RULES]
        complex_rules = [r for r in rules if r.get("rule") not in self._DETERMINISTIC_RULES]

        rule_results: list[RuleResult] = []
        for idx, rule in enumerate(deterministic_rules):
            rule_results.append(
                self._evaluate_rule(f"r{idx + 1}", rule, output)
            )

        # Delegate complex rules to the LLM
        if complex_rules and use_llm_for_complex:
            user_prompt = TaskPrompts.build_validation_prompt(
                output=output,
                rules=complex_rules,
                include_few_shot=True,
            )
            llm_report: ValidationReport = self.run(user_prompt)
            # Offset rule_ids to avoid collisions
            for r in llm_report.rule_results:
                r.rule_id = f"llm_{r.rule_id}"
            rule_results.extend(llm_report.rule_results)

        total = len(rule_results)
        passed = sum(1 for r in rule_results if r.passed)
        score = int(passed / total * 100) if total else 100
        valid = all(r.passed for r in rule_results)

        return ValidationReport(
            valid=valid,
            score=score,
            rule_results=rule_results,
            summary=(
                f"{passed}/{total} validation rules passed."
                + ("" if valid else " See rule_results for details.")
            ),
        )

    # ── Built-in rule engine ──────────────────────────────────────────────────

    def _evaluate_rule(
        self, rule_id: str, rule: dict[str, Any], output: dict[str, Any]
    ) -> RuleResult:
        """Evaluate a single deterministic rule."""
        field_path: str = rule.get("field", "")
        rule_type: str = rule.get("rule", "")
        expected: Any = rule.get("value")
        actual: Any = self._get_field(output, field_path)

        passed, message = self._check_rule(rule_type, actual, expected, field_path)
        return RuleResult(
            rule_id=rule_id,
            field_path=field_path,
            rule_type=rule_type,
            passed=passed,
            actual_value=actual,
            expected_value=expected,
            message=message,
        )

    @staticmethod
    def _get_field(data: dict[str, Any], path: str) -> Any:
        """Traverse a dot-separated field path in a nested dict."""
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    @staticmethod
    def _check_rule(
        rule_type: str, actual: Any, expected: Any, field: str
    ) -> tuple[bool, str]:
        """Return (passed, message) for a deterministic rule."""
        if rule_type == "required":
            passed = actual is not None and actual != ""
            msg = (
                f"Field '{field}' is present and non-null"
                if passed
                else f"Field '{field}' is required but missing or null"
            )
            return passed, msg

        if rule_type == "type":
            type_map = {
                "string": str,
                "integer": int,
                "float": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            expected_type = type_map.get(str(expected), type(None))
            # integer check: exclude bool (bool is subclass of int)
            if expected == "integer":
                passed = isinstance(actual, int) and not isinstance(actual, bool)
            else:
                passed = isinstance(actual, expected_type)
            msg = (
                f"Field '{field}' has type {type(actual).__name__!r} as expected"
                if passed
                else f"Field '{field}' expected type '{expected}', got {type(actual).__name__!r}"
            )
            return passed, msg

        if rule_type == "min_value":
            if actual is None:
                return False, f"Field '{field}' is null; cannot compare to min_value {expected}"
            try:
                passed = float(actual) >= float(expected)
            except (TypeError, ValueError):
                return False, f"Field '{field}' is not numeric"
            msg = (
                f"{field}={actual} >= {expected}"
                if passed
                else f"{field}={actual} must be >= {expected}"
            )
            return passed, msg

        if rule_type == "max_value":
            if actual is None:
                return False, f"Field '{field}' is null; cannot compare to max_value {expected}"
            try:
                passed = float(actual) <= float(expected)
            except (TypeError, ValueError):
                return False, f"Field '{field}' is not numeric"
            msg = (
                f"{field}={actual} <= {expected}"
                if passed
                else f"{field}={actual} must be <= {expected}"
            )
            return passed, msg

        if rule_type == "min_length":
            length = len(actual) if actual is not None else 0
            passed = length >= int(expected)
            msg = (
                f"Length {length} >= {expected}"
                if passed
                else f"Length {length} of '{field}' is less than min_length {expected}"
            )
            return passed, msg

        if rule_type == "max_length":
            length = len(actual) if actual is not None else 0
            passed = length <= int(expected)
            msg = (
                f"Length {length} <= {expected}"
                if passed
                else f"Length {length} of '{field}' exceeds max_length {expected}"
            )
            return passed, msg

        if rule_type == "enum":
            choices = expected if isinstance(expected, list) else [expected]
            passed = actual in choices
            msg = (
                f"'{actual}' is a valid choice"
                if passed
                else f"'{actual}' is not in allowed values {choices}"
            )
            return passed, msg

        if rule_type == "pattern":
            if actual is None:
                return False, f"Field '{field}' is null; cannot match pattern"
            try:
                passed = bool(re.fullmatch(str(expected), str(actual)))
            except re.error as exc:
                return False, f"Invalid regex pattern '{expected}': {exc}"
            msg = (
                f"Value matches pattern '{expected}'"
                if passed
                else f"Value '{actual}' does not match pattern '{expected}'"
            )
            return passed, msg

        return False, f"Unknown rule type '{rule_type}'"

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def _parse_response(self, response: LLMResponse, **kwargs: Any) -> ValidationReport:
        data = response.parse_json()
        return ValidationReport.from_dict(data)

    def _build_mock_response(self, user_prompt: str) -> dict[str, Any]:
        return {
            "valid": True,
            "score": 95,
            "rule_results": [
                {
                    "rule_id": "r1",
                    "field": "result",
                    "rule_type": "custom",
                    "passed": True,
                    "actual_value": "ok",
                    "expected_value": None,
                    "message": "Mock validation passed",
                }
            ],
            "summary": "Mock validation: all rules passed",
            "suggested_corrections": {},
        }

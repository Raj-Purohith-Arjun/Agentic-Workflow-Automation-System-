"""
Rule-based data validator.

Validates Python dicts against JSON Schema and/or a list of business rules.
Can be registered as a tool callable with the ExecutionAgent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import Draft7Validator
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


@dataclass
class RuleViolation:
    """A single rule violation."""

    field: str
    rule: str
    message: str
    severity: str = "error"  # error | warning

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class SchemaValidationResult:
    """Result of a full schema + business-rule validation pass."""

    valid: bool
    violations: list[RuleViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_rules: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "checked_rules": self.checked_rules,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class DataValidator:
    """
    Validates data dicts against JSON Schema and business rules.

    Callable interface for ExecutionAgent tool registration:

    .. code-block:: python

        validator = DataValidator()
        engine.register_tool("data_validator", validator)
    """

    # ── Callable interface ────────────────────────────────────────────────────

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Callable interface for the ExecutionAgent tool registry.

        ``inputs`` keys: data (dict), schema (dict), rules (list).
        """
        data: dict[str, Any] = inputs.get("data") or {}
        json_schema: dict[str, Any] | None = inputs.get("schema")
        rules: list[dict[str, Any]] = inputs.get("rules") or []

        result = self.validate(data=data, json_schema=json_schema, rules=rules)
        return result.to_dict()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        data: dict[str, Any],
        json_schema: dict[str, Any] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> SchemaValidationResult:
        """
        Validate ``data`` against an optional JSON Schema and/or business rules.

        Parameters
        ----------
        data:
            The dict to validate.
        json_schema:
            Optional JSON Schema (Draft 7) for structural validation.
        rules:
            Optional list of business-rule dicts
            ``{field, rule, value, severity}``.

        Returns
        -------
        SchemaValidationResult
        """
        violations: list[RuleViolation] = []
        checked = 0

        # ── JSON Schema validation ────────────────────────────────────────────
        if json_schema:
            schema_violations = self._validate_schema(data, json_schema)
            violations.extend(schema_violations)
            checked += 1

        # ── Business rule evaluation ──────────────────────────────────────────
        for rule_def in (rules or []):
            v = self._evaluate_rule(data, rule_def)
            if v:
                violations.append(v)
            checked += 1

        errors = [v for v in violations if v.severity == "error"]
        return SchemaValidationResult(
            valid=len(errors) == 0,
            violations=violations,
            checked_rules=checked,
        )

    # ── Schema validation ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_schema(
        data: dict[str, Any], schema: dict[str, Any]
    ) -> list[RuleViolation]:
        if not _HAS_JSONSCHEMA:
            logger.warning("jsonschema not installed; skipping JSON Schema validation")
            return []
        violations: list[RuleViolation] = []
        validator = Draft7Validator(schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.absolute_path) or "root"
            violations.append(
                RuleViolation(
                    field=path,
                    rule="json_schema",
                    message=error.message,
                    severity="error",
                )
            )
        return violations

    # ── Business rule evaluation ──────────────────────────────────────────────

    @staticmethod
    def _get_nested(data: Any, path: str) -> Any:
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _evaluate_rule(
        self, data: dict[str, Any], rule_def: dict[str, Any]
    ) -> RuleViolation | None:
        field_path: str = rule_def.get("field", "")
        rule_type: str = rule_def.get("rule", "")
        expected: Any = rule_def.get("value")
        severity: str = rule_def.get("severity", "error")
        actual = self._get_nested(data, field_path)

        passed, message = self._check(rule_type, actual, expected, field_path)
        if passed:
            return None
        return RuleViolation(field=field_path, rule=rule_type, message=message, severity=severity)

    @staticmethod
    def _check(rule_type: str, actual: Any, expected: Any, field: str) -> tuple[bool, str]:
        if rule_type == "required":
            ok = actual is not None and actual != ""
            return ok, "" if ok else f"'{field}' is required but missing"

        if rule_type == "not_empty":
            ok = bool(actual)
            return ok, "" if ok else f"'{field}' must not be empty"

        if rule_type == "type":
            type_map = {
                "string": str, "integer": int, "float": float,
                "boolean": bool, "array": list, "object": dict,
            }
            exp_type = type_map.get(str(expected))
            if exp_type is None:
                return True, ""
            if expected == "integer":
                ok = isinstance(actual, int) and not isinstance(actual, bool)
            else:
                ok = isinstance(actual, exp_type)
            return ok, "" if ok else f"'{field}' expected {expected}, got {type(actual).__name__}"

        if rule_type in ("min_value", "gte"):
            if actual is None:
                return False, f"'{field}' is null"
            try:
                ok = float(actual) >= float(expected)
            except (TypeError, ValueError):
                return False, f"'{field}' is not numeric"
            return ok, "" if ok else f"'{field}'={actual} must be >= {expected}"

        if rule_type in ("max_value", "lte"):
            if actual is None:
                return False, f"'{field}' is null"
            try:
                ok = float(actual) <= float(expected)
            except (TypeError, ValueError):
                return False, f"'{field}' is not numeric"
            return ok, "" if ok else f"'{field}'={actual} must be <= {expected}"

        if rule_type == "min_length":
            length = len(actual) if actual is not None else 0
            ok = length >= int(expected)
            return ok, "" if ok else f"'{field}' length {length} < min {expected}"

        if rule_type == "max_length":
            length = len(actual) if actual is not None else 0
            ok = length <= int(expected)
            return ok, "" if ok else f"'{field}' length {length} > max {expected}"

        if rule_type == "enum":
            choices = expected if isinstance(expected, list) else [expected]
            ok = actual in choices
            return ok, "" if ok else f"'{field}'={actual!r} not in {choices}"

        if rule_type == "pattern":
            if actual is None:
                return False, f"'{field}' is null"
            try:
                ok = bool(re.fullmatch(str(expected), str(actual)))
            except re.error as exc:
                return False, f"Invalid pattern '{expected}': {exc}"
            return ok, "" if ok else f"'{field}'={actual!r} doesn't match '{expected}'"

        if rule_type == "unique":
            if not isinstance(actual, list):
                return True, ""  # non-list field is trivially unique
            ok = len(actual) == len(set(actual))
            return ok, "" if ok else f"'{field}' contains duplicate values"

        if rule_type == "not_null":
            ok = actual is not None
            return ok, "" if ok else f"'{field}' must not be null"

        # Unknown rule – treat as passing (don't block on unknown rules)
        return True, ""

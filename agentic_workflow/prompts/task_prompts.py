"""
Task-specific prompt templates with few-shot examples.

These templates are injected into LLM calls as user-turn messages.
They embed context, available tools, and few-shot demonstrations.
"""

from __future__ import annotations

import json
from string import Template
from typing import Any


class TaskPrompts:
    """Factory for task-level prompt strings."""

    # ── Planning ─────────────────────────────────────────────────────────────

    PLAN_OBJECTIVE = Template(
        """## Objective
$objective

## Available Tools
$tools_json

## Context
$context_json

## Instructions
Produce a detailed, step-by-step execution plan for the objective above.
Return only valid JSON – no markdown fences, no prose outside the JSON object.
"""
    )

    # ── Execution ────────────────────────────────────────────────────────────

    EXECUTE_STEP = Template(
        """## Step to Execute
$step_json

## Previous Step Outputs (available for reference)
$previous_outputs_json

## Tool Call Result
$tool_result_json

## Instructions
Evaluate the tool result against the step's validation rules and return the
execution result as valid JSON.
"""
    )

    # ── Validation ───────────────────────────────────────────────────────────

    VALIDATE_OUTPUT = Template(
        """## Output to Validate
$output_json

## Validation Rules
$rules_json

## Instructions
Evaluate every rule against the output and return a comprehensive validation
report as valid JSON.
"""
    )

    # ── Document Analysis ────────────────────────────────────────────────────

    ANALYSE_DOCUMENT = Template(
        """## Document Content
$document_text

## Extraction Schema
$schema_json

## Instructions
Extract all fields defined in the extraction schema from the document.
Return the result as valid JSON following the document analyst output format.
"""
    )

    # ── Decision Support ─────────────────────────────────────────────────────

    SUPPORT_DECISION = Template(
        """## Business Scenario
$scenario

## Available Data
$data_json

## Decision Criteria
$criteria_json

## Instructions
Analyse the scenario and available data, then produce a structured
recommendation following the decision support output format.
"""
    )

    # ── Few-shot examples ────────────────────────────────────────────────────

    FEW_SHOT_PLANNING = """## Example Plan (invoice processing workflow)
{
  "plan_id": "plan-001",
  "objective": "Process incoming invoice, validate fields, and post to ERP",
  "steps": [
    {
      "step_id": 1,
      "name": "extract_invoice_data",
      "description": "Extract structured data from the invoice document",
      "tool": "document_processor",
      "inputs": { "document_id": "{{invoice_id}}", "schema": "invoice_schema" },
      "depends_on": [],
      "validation_rules": [
        { "field": "invoice_number", "rule": "required", "value": null },
        { "field": "total_amount", "rule": "min_value", "value": 0 }
      ],
      "on_failure": "abort"
    },
    {
      "step_id": 2,
      "name": "validate_vendor",
      "description": "Check vendor exists and is approved in the vendor master",
      "tool": "api_client",
      "inputs": {
        "endpoint": "/vendors/{{step_1.vendor_id}}",
        "method": "GET"
      },
      "depends_on": [1],
      "validation_rules": [
        { "field": "status", "rule": "enum", "value": ["active", "preferred"] }
      ],
      "on_failure": "abort"
    },
    {
      "step_id": 3,
      "name": "post_to_erp",
      "description": "Create invoice record in ERP system",
      "tool": "api_client",
      "inputs": {
        "endpoint": "/erp/invoices",
        "method": "POST",
        "body": "{{step_1.output}}"
      },
      "depends_on": [1, 2],
      "validation_rules": [
        { "field": "erp_id", "rule": "required", "value": null },
        { "field": "status", "rule": "enum", "value": ["created", "pending"] }
      ],
      "on_failure": "retry"
    }
  ],
  "estimated_steps": 3,
  "risk_level": "medium"
}
"""

    FEW_SHOT_VALIDATION = """## Example Validation Report
{
  "valid": false,
  "score": 72,
  "rule_results": [
    {
      "rule_id": "r1",
      "field": "invoice_number",
      "rule_type": "required",
      "passed": true,
      "actual_value": "INV-2024-001",
      "expected_value": null,
      "message": "Field is present and non-null"
    },
    {
      "rule_id": "r2",
      "field": "total_amount",
      "rule_type": "min_value",
      "passed": false,
      "actual_value": -50.0,
      "expected_value": 0,
      "message": "total_amount must be >= 0; got -50.0"
    }
  ],
  "summary": "1 of 2 validation rules failed. Invoice total_amount is negative.",
  "suggested_corrections": {
    "total_amount": "Verify the invoice – a negative amount indicates a credit note"
  }
}
"""

    @classmethod
    def build_planning_prompt(
        cls,
        objective: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        include_few_shot: bool = True,
    ) -> str:
        """Build a complete planning prompt."""
        prompt = cls.PLAN_OBJECTIVE.substitute(
            objective=objective,
            tools_json=json.dumps(tools, indent=2),
            context_json=json.dumps(context or {}, indent=2),
        )
        if include_few_shot:
            prompt = cls.FEW_SHOT_PLANNING + "\n---\n\n" + prompt
        return prompt

    @classmethod
    def build_execution_prompt(
        cls,
        step: dict[str, Any],
        previous_outputs: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> str:
        """Build an execution evaluation prompt."""
        return cls.EXECUTE_STEP.substitute(
            step_json=json.dumps(step, indent=2),
            previous_outputs_json=json.dumps(previous_outputs, indent=2),
            tool_result_json=json.dumps(tool_result, indent=2),
        )

    @classmethod
    def build_validation_prompt(
        cls,
        output: dict[str, Any],
        rules: list[dict[str, Any]],
        include_few_shot: bool = True,
    ) -> str:
        """Build a validation prompt."""
        prompt = cls.VALIDATE_OUTPUT.substitute(
            output_json=json.dumps(output, indent=2),
            rules_json=json.dumps(rules, indent=2),
        )
        if include_few_shot:
            prompt = cls.FEW_SHOT_VALIDATION + "\n---\n\n" + prompt
        return prompt

    @classmethod
    def build_document_analysis_prompt(
        cls,
        document_text: str,
        schema: dict[str, Any],
    ) -> str:
        """Build a document analysis prompt."""
        return cls.ANALYSE_DOCUMENT.substitute(
            document_text=document_text,
            schema_json=json.dumps(schema, indent=2),
        )

    @classmethod
    def build_decision_support_prompt(
        cls,
        scenario: str,
        data: dict[str, Any],
        criteria: list[dict[str, Any]],
    ) -> str:
        """Build a decision support prompt."""
        return cls.SUPPORT_DECISION.substitute(
            scenario=scenario,
            data_json=json.dumps(data, indent=2),
            criteria_json=json.dumps(criteria, indent=2),
        )

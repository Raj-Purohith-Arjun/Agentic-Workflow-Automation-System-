"""
Tests for the prompt engineering module.
"""

from __future__ import annotations

import json

import pytest

from agentic_workflow.prompts.system_prompts import SystemPrompts
from agentic_workflow.prompts.task_prompts import TaskPrompts


class TestSystemPrompts:
    def test_planner_prompt_non_empty(self):
        assert len(SystemPrompts.PLANNER) > 100

    def test_executor_prompt_non_empty(self):
        assert len(SystemPrompts.EXECUTOR) > 100

    def test_validator_prompt_non_empty(self):
        assert len(SystemPrompts.VALIDATOR) > 100

    def test_document_analyst_prompt_non_empty(self):
        assert len(SystemPrompts.DOCUMENT_ANALYST) > 100

    def test_decision_support_prompt_non_empty(self):
        assert len(SystemPrompts.DECISION_SUPPORT) > 100

    def test_planner_contains_json_instruction(self):
        assert "JSON" in SystemPrompts.PLANNER

    def test_validator_contains_rule_types(self):
        for rule in ("required", "enum", "pattern"):
            assert rule in SystemPrompts.VALIDATOR


class TestTaskPrompts:
    def test_build_planning_prompt(self):
        tools = [{"name": "api_client", "description": "HTTP client"}]
        prompt = TaskPrompts.build_planning_prompt(
            objective="Process invoices",
            tools=tools,
            context={"env": "prod"},
            include_few_shot=False,
        )
        assert "Process invoices" in prompt
        assert "api_client" in prompt
        assert "prod" in prompt

    def test_build_planning_prompt_with_few_shot(self):
        tools = [{"name": "api_client", "description": "HTTP client"}]
        prompt = TaskPrompts.build_planning_prompt(
            objective="obj",
            tools=tools,
            include_few_shot=True,
        )
        assert "invoice" in prompt.lower()  # few-shot example mentions invoice

    def test_build_execution_prompt(self):
        step = {"step_id": 1, "name": "fetch", "tool": "api_client"}
        previous = {"1": {"data": "prev"}}
        tool_result = {"status_code": 200, "body": {"id": "123"}}
        prompt = TaskPrompts.build_execution_prompt(step, previous, tool_result)
        assert "fetch" in prompt
        assert "200" in prompt

    def test_build_validation_prompt(self):
        output = {"name": "Alice", "age": 30}
        rules = [{"field": "name", "rule": "required"}]
        prompt = TaskPrompts.build_validation_prompt(output, rules, include_few_shot=False)
        assert "Alice" in prompt
        assert "required" in prompt

    def test_build_validation_prompt_with_few_shot(self):
        output = {"x": 1}
        rules = [{"field": "x", "rule": "required"}]
        prompt = TaskPrompts.build_validation_prompt(output, rules, include_few_shot=True)
        assert "invoice" in prompt.lower()

    def test_build_document_analysis_prompt(self):
        schema = {"invoice_number": "Invoice ID"}
        prompt = TaskPrompts.build_document_analysis_prompt(
            document_text="Invoice #INV-001 from Acme Corp",
            schema=schema,
        )
        assert "INV-001" in prompt
        assert "invoice_number" in prompt

    def test_build_decision_support_prompt(self):
        prompt = TaskPrompts.build_decision_support_prompt(
            scenario="Should we approve the invoice?",
            data={"amount": 1000},
            criteria=[{"criterion": "amount < 5000", "weight": "high"}],
        )
        assert "approve" in prompt.lower()
        assert "1000" in prompt

    def test_planning_prompt_tools_as_valid_json(self):
        tools = [{"name": "api_client", "description": "HTTP"}]
        prompt = TaskPrompts.build_planning_prompt("obj", tools, include_few_shot=False)
        # The tools section should be valid JSON somewhere in the prompt
        start = prompt.find("[")
        end = prompt.rfind("]") + 1
        json_str = prompt[start:end]
        parsed = json.loads(json_str)
        assert parsed[0]["name"] == "api_client"

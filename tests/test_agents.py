"""
Tests for the agent layer (PlanningAgent, ExecutionAgent, ValidationAgent).

All tests run without an OpenAI API key; the agents operate in mock mode.
"""

from __future__ import annotations

import pytest

from agentic_workflow.agents.base_agent import LLMResponse
from agentic_workflow.agents.planning_agent import PlanningAgent, WorkflowPlan, PlanStep
from agentic_workflow.agents.execution_agent import ExecutionAgent, StepResult
from agentic_workflow.agents.validation_agent import ValidationAgent, ValidationReport


# ── LLMResponse tests ─────────────────────────────────────────────────────────

class TestLLMResponse:
    def test_parse_plain_json(self):
        r = LLMResponse(content='{"key": "value"}', model="test")
        assert r.parse_json() == {"key": "value"}

    def test_parse_fenced_json(self):
        content = "```json\n{\"key\": 1}\n```"
        r = LLMResponse(content=content, model="test")
        assert r.parse_json() == {"key": 1}

    def test_parse_fenced_no_lang(self):
        content = "```\n{\"a\": true}\n```"
        r = LLMResponse(content=content, model="test")
        assert r.parse_json() == {"a": True}

    def test_invalid_json_raises(self):
        r = LLMResponse(content="not json", model="test")
        with pytest.raises(ValueError, match="not valid JSON"):
            r.parse_json()

    def test_repr(self):
        r = LLMResponse("", "gpt-4", 10, 20, 100)
        assert "gpt-4" in repr(r)
        assert "100ms" in repr(r)


# ── PlanningAgent tests ───────────────────────────────────────────────────────

class TestPlanningAgent:
    @pytest.fixture
    def agent(self):
        return PlanningAgent()

    @pytest.fixture
    def tools(self):
        return [
            {"name": "api_client", "description": "Make HTTP calls"},
            {"name": "document_processor", "description": "Parse docs"},
        ]

    def test_plan_returns_workflow_plan(self, agent, tools):
        plan = agent.plan("Test objective", available_tools=tools)
        assert isinstance(plan, WorkflowPlan)

    def test_plan_has_steps(self, agent, tools):
        plan = agent.plan("Build a report", available_tools=tools)
        assert len(plan.steps) > 0

    def test_plan_has_id(self, agent, tools):
        plan = agent.plan("Objective", available_tools=tools)
        assert plan.plan_id != ""

    def test_plan_steps_have_required_fields(self, agent, tools):
        plan = agent.plan("Objective", available_tools=tools)
        for step in plan.steps:
            assert isinstance(step.step_id, int)
            assert step.name != ""
            assert step.tool != ""

    def test_plan_with_context(self, agent, tools):
        plan = agent.plan("Obj", tools, context={"env": "prod"})
        assert isinstance(plan, WorkflowPlan)

    def test_get_ready_steps_no_deps(self, agent, tools):
        plan = agent.plan("Obj", tools)
        # First step should be ready with no prior completions
        ready = plan.get_ready_steps(completed_ids=set())
        first_step = plan.steps[0]
        if not first_step.depends_on:
            assert first_step in ready

    def test_get_ready_steps_with_completed(self, agent, tools):
        plan = agent.plan("Obj", tools)
        # All steps with dependencies on step 1 become ready once step 1 is done
        ready_after_1 = plan.get_ready_steps(completed_ids={1})
        for step in ready_after_1:
            assert 1 not in step.depends_on or all(d in {1} for d in step.depends_on)

    def test_workflow_plan_to_dict_roundtrip(self, agent, tools):
        plan = agent.plan("Obj", tools)
        d = plan.to_dict()
        restored = WorkflowPlan.from_dict(d)
        assert restored.plan_id == plan.plan_id
        assert len(restored.steps) == len(plan.steps)

    def test_plan_step_to_dict_roundtrip(self):
        step = PlanStep(
            step_id=1,
            name="fetch",
            description="Fetch data",
            tool="api_client",
            inputs={"endpoint": "/data"},
            on_failure="retry",
        )
        d = step.to_dict()
        restored = PlanStep.from_dict(d)
        assert restored.step_id == 1
        assert restored.on_failure == "retry"


# ── ExecutionAgent tests ──────────────────────────────────────────────────────

class TestExecutionAgent:
    @pytest.fixture
    def step(self):
        return PlanStep(
            step_id=1,
            name="test_step",
            description="Test step",
            tool="mock_tool",
            inputs={"key": "value"},
        )

    @pytest.fixture
    def agent_with_tool(self):
        agent = ExecutionAgent()

        def mock_tool(inputs):
            return {"result": "success", "data": inputs.get("key", "default")}

        agent.register_tool("mock_tool", mock_tool)
        return agent

    def test_execute_returns_step_result(self, agent_with_tool, step):
        result = agent_with_tool.execute(step)
        assert isinstance(result, StepResult)

    def test_execute_step_id_set(self, agent_with_tool, step):
        result = agent_with_tool.execute(step)
        assert result.step_id == step.step_id

    def test_execute_missing_tool(self):
        agent = ExecutionAgent()
        step = PlanStep(1, "s", "d", "nonexistent_tool", {})
        result = agent.execute(step)
        # Should return a failure result, not raise
        assert isinstance(result, StepResult)

    def test_step_result_succeeded(self):
        r = StepResult(step_id=1, status="success")
        assert r.succeeded is True

    def test_step_result_not_succeeded(self):
        r = StepResult(step_id=1, status="failure")
        assert r.succeeded is False

    def test_resolve_inputs_with_references(self, agent_with_tool):
        step = PlanStep(
            step_id=2,
            name="step2",
            description="Uses step 1 output",
            tool="mock_tool",
            inputs={"key": "{{step_1.result}}"},
            depends_on=[1],
        )
        previous = {1: {"result": "hello_from_step1"}}
        result = agent_with_tool.execute(step, previous_outputs=previous)
        assert isinstance(result, StepResult)

    def test_step_result_to_dict_roundtrip(self):
        r = StepResult(
            step_id=5,
            status="partial",
            output={"x": 1},
            validation_passed=False,
            validation_errors=["err"],
        )
        d = r.to_dict()
        restored = StepResult.from_dict(d)
        assert restored.step_id == 5
        assert restored.status == "partial"
        assert restored.validation_errors == ["err"]


# ── ValidationAgent tests ─────────────────────────────────────────────────────

class TestValidationAgent:
    @pytest.fixture
    def agent(self):
        return ValidationAgent()

    def test_validate_required_pass(self, agent):
        output = {"name": "Alice"}
        rules = [{"field": "name", "rule": "required"}]
        report = agent.validate(output, rules)
        assert report.valid is True
        assert report.score == 100

    def test_validate_required_fail(self, agent):
        output = {"name": None}
        rules = [{"field": "name", "rule": "required"}]
        report = agent.validate(output, rules)
        assert report.valid is False
        assert report.failed_count == 1

    def test_validate_type_string(self, agent):
        rules = [{"field": "val", "rule": "type", "value": "string"}]
        assert agent.validate({"val": "hello"}, rules).valid is True
        assert agent.validate({"val": 123}, rules).valid is False

    def test_validate_type_integer(self, agent):
        rules = [{"field": "val", "rule": "type", "value": "integer"}]
        assert agent.validate({"val": 5}, rules).valid is True
        # bool is NOT integer
        assert agent.validate({"val": True}, rules).valid is False

    def test_validate_type_boolean(self, agent):
        rules = [{"field": "val", "rule": "type", "value": "boolean"}]
        assert agent.validate({"val": True}, rules).valid is True
        assert agent.validate({"val": 1}, rules).valid is False

    def test_validate_min_value(self, agent):
        rules = [{"field": "amount", "rule": "min_value", "value": 0}]
        assert agent.validate({"amount": 100}, rules).valid is True
        assert agent.validate({"amount": -1}, rules).valid is False

    def test_validate_max_value(self, agent):
        rules = [{"field": "score", "rule": "max_value", "value": 100}]
        assert agent.validate({"score": 99}, rules).valid is True
        assert agent.validate({"score": 101}, rules).valid is False

    def test_validate_min_length(self, agent):
        rules = [{"field": "name", "rule": "min_length", "value": 3}]
        assert agent.validate({"name": "Alice"}, rules).valid is True
        assert agent.validate({"name": "Al"}, rules).valid is False

    def test_validate_max_length(self, agent):
        rules = [{"field": "code", "rule": "max_length", "value": 5}]
        assert agent.validate({"code": "AB123"}, rules).valid is True
        assert agent.validate({"code": "TOOLONG"}, rules).valid is False

    def test_validate_enum(self, agent):
        rules = [{"field": "status", "rule": "enum", "value": ["active", "inactive"]}]
        assert agent.validate({"status": "active"}, rules).valid is True
        assert agent.validate({"status": "unknown"}, rules).valid is False

    def test_validate_pattern(self, agent):
        rules = [{"field": "date", "rule": "pattern", "value": r"\d{4}-\d{2}-\d{2}"}]
        assert agent.validate({"date": "2024-01-01"}, rules).valid is True
        assert agent.validate({"date": "01/01/2024"}, rules).valid is False

    def test_validate_empty_rules(self, agent):
        report = agent.validate({"x": 1}, [])
        assert report.valid is True
        assert report.score == 100

    def test_validate_multiple_rules_mixed(self, agent):
        output = {"name": "Alice", "age": -1, "status": "active"}
        rules = [
            {"field": "name", "rule": "required"},
            {"field": "age", "rule": "min_value", "value": 0},
            {"field": "status", "rule": "enum", "value": ["active", "inactive"]},
        ]
        report = agent.validate(output, rules)
        assert report.passed_count == 2
        assert report.failed_count == 1
        assert not report.valid

    def test_validation_report_from_dict(self):
        data = {
            "valid": True,
            "score": 90,
            "rule_results": [],
            "summary": "ok",
            "suggested_corrections": {},
        }
        report = ValidationReport.from_dict(data)
        assert report.valid is True
        assert report.score == 90

    def test_nested_field_validation(self, agent):
        output = {"vendor": {"status": "active"}}
        rules = [{"field": "vendor.status", "rule": "enum", "value": ["active", "inactive"]}]
        report = agent.validate(output, rules)
        assert report.valid is True

    def test_validate_no_llm_for_deterministic(self, agent, mocker):
        """Deterministic rules must never trigger an LLM call."""
        spy = mocker.spy(agent, "_call_llm")
        rules = [{"field": "x", "rule": "required"}]
        agent.validate({"x": "value"}, rules, use_llm_for_complex=True)
        spy.assert_not_called()

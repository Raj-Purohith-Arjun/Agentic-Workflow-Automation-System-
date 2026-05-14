"""
Tests for the workflow layer (WorkflowState, WorkflowEngine).
"""

from __future__ import annotations

import tempfile

import pytest

from agentic_workflow.workflows.state import (
    WorkflowState,
    WorkflowStatus,
    StepStatus,
)
from agentic_workflow.workflows.engine import WorkflowEngine
from agentic_workflow.workflows.definitions import BUILT_IN_WORKFLOWS


# ── WorkflowState tests ───────────────────────────────────────────────────────

class TestWorkflowState:
    def test_initial_status_pending(self):
        state = WorkflowState()
        assert state.status == WorkflowStatus.PENDING

    def test_transition_changes_status(self):
        state = WorkflowState()
        state.transition(WorkflowStatus.RUNNING)
        assert state.status == WorkflowStatus.RUNNING

    def test_transition_sets_completed_at(self):
        state = WorkflowState()
        state.transition(WorkflowStatus.COMPLETED)
        assert state.completed_at != ""

    def test_record_step_start(self):
        state = WorkflowState()
        record = state.record_step_start(1, "step_one")
        assert record.status == StepStatus.RUNNING
        assert 1 in state.step_records

    def test_record_step_complete_success(self):
        state = WorkflowState()
        state.record_step_start(1, "step_one")
        state.record_step_complete(1, {"result": "ok"}, True)
        assert state.step_records[1].status == StepStatus.SUCCESS

    def test_record_step_complete_failure(self):
        state = WorkflowState()
        state.record_step_start(1, "step_one")
        state.record_step_complete(1, {}, False, ["err"])
        assert state.step_records[1].status == StepStatus.FAILED

    def test_record_step_failure(self):
        state = WorkflowState()
        state.record_step_start(1, "step_one")
        state.record_step_failure(1, "boom")
        assert state.step_records[1].status == StepStatus.FAILED
        assert state.step_records[1].error == "boom"

    def test_record_step_skip(self):
        state = WorkflowState()
        state.record_step_skip(2, "skipped_step")
        assert state.step_records[2].status == StepStatus.SKIPPED

    def test_completed_step_ids(self):
        state = WorkflowState()
        state.record_step_start(1, "s1")
        state.record_step_complete(1, {}, True)
        state.record_step_skip(2, "s2")
        assert {1, 2} == state.completed_step_ids

    def test_step_outputs(self):
        state = WorkflowState()
        state.record_step_start(1, "s1")
        state.record_step_complete(1, {"x": 42}, True)
        assert state.step_outputs[1] == {"x": 42}

    def test_progress_pct(self):
        state = WorkflowState()
        state.total_steps = 4
        state.record_step_start(1, "s1")
        state.record_step_complete(1, {}, True)
        state.record_step_start(2, "s2")
        state.record_step_complete(2, {}, True)
        assert state.progress_pct == 50.0

    def test_increment_retry(self):
        state = WorkflowState()
        state.record_step_start(1, "s1")
        count = state.increment_retry(1)
        assert count == 1
        assert state.step_records[1].status == StepStatus.RETRYING

    def test_to_dict_roundtrip(self):
        state = WorkflowState(workflow_name="test", objective="obj")
        state.transition(WorkflowStatus.RUNNING)
        state.record_step_start(1, "s1")
        state.record_step_complete(1, {"y": 1}, True)
        d = state.to_dict()
        restored = WorkflowState.from_dict(d)
        assert restored.workflow_id == state.workflow_id
        assert restored.status == WorkflowStatus.RUNNING
        assert 1 in restored.step_records

    def test_save_and_load(self):
        state = WorkflowState(workflow_name="persist_test")
        state.transition(WorkflowStatus.RUNNING)
        with tempfile.TemporaryDirectory() as tmpdir:
            state.save(tmpdir)
            loaded = WorkflowState.load(tmpdir, state.workflow_id)
        assert loaded.workflow_id == state.workflow_id
        assert loaded.status == WorkflowStatus.RUNNING

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                WorkflowState.load(tmpdir, "nonexistent-id")

    def test_repr(self):
        state = WorkflowState()
        assert "WorkflowState" in repr(state)


# ── WorkflowEngine tests ──────────────────────────────────────────────────────

def _make_engine() -> WorkflowEngine:
    """Build an engine with mock tools registered."""
    engine = WorkflowEngine()

    def mock_api(inputs):
        endpoint = inputs.get("endpoint", "/")
        return {"ok": True, "status_code": 200, "body": {"id": "mock-001", "status": "ok", "data": "mock"}}

    engine.register_tool("api_client", mock_api)
    engine.register_tool("document_processor", lambda inputs: {"processed": True, "text": "doc"})
    engine.register_tool("data_validator", lambda inputs: {"valid": True})
    return engine


class TestWorkflowEngine:
    def test_run_returns_workflow_state(self):
        engine = _make_engine()
        state = engine.run("Process data", workflow_name="test")
        assert isinstance(state, WorkflowState)

    def test_run_has_terminal_status(self):
        engine = _make_engine()
        state = engine.run("Objective")
        assert state.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
        )

    def test_run_records_steps(self):
        engine = _make_engine()
        state = engine.run("Objective")
        assert len(state.step_records) > 0

    def test_run_sets_plan_id(self):
        engine = _make_engine()
        state = engine.run("Objective")
        assert state.plan_id != ""

    def test_register_tool(self):
        engine = WorkflowEngine()
        engine.register_tool("my_tool", lambda x: {"ok": True})
        assert "my_tool" in engine._executor._tools

    def test_run_with_context(self):
        engine = _make_engine()
        state = engine.run("Objective", context={"env": "test"})
        assert state.context.get("env") == "test"

    def test_on_step_complete_callback(self):
        from agentic_workflow.agents.execution_agent import StepResult
        called = []

        def callback(state, result):
            called.append(result.step_id)

        engine = _make_engine()
        engine._on_step_complete = callback
        engine.run("Objective")
        assert len(called) > 0

    def test_file_state_backend_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = WorkflowEngine(state_backend="file", state_dir=tmpdir)
            engine.register_tool("api_client", lambda i: {"ok": True, "status_code": 200, "body": {}})
            state = engine.run("Test objective")
            # File should exist
            import os
            files = os.listdir(tmpdir)
            assert any(state.workflow_id in f for f in files)

    def test_workflow_id_carried_over(self):
        engine = _make_engine()
        state = engine.run("Objective", workflow_id="my-fixed-id")
        assert state.workflow_id == "my-fixed-id"


# ── Built-in workflow definitions ─────────────────────────────────────────────

class TestBuiltInWorkflows:
    def test_all_workflows_present(self):
        expected = {"invoice_processing", "document_review", "customer_onboarding", "data_pipeline"}
        assert expected == set(BUILT_IN_WORKFLOWS.keys())

    def test_each_workflow_has_required_keys(self):
        for key, wf in BUILT_IN_WORKFLOWS.items():
            assert "name" in wf, f"{key} missing 'name'"
            assert "description" in wf, f"{key} missing 'description'"
            assert "objective" in wf, f"{key} missing 'objective'"

    def test_validation_rules_are_lists(self):
        for key, wf in BUILT_IN_WORKFLOWS.items():
            if "validation_rules" in wf:
                for entity, rules in wf["validation_rules"].items():
                    assert isinstance(rules, list), f"{key}.{entity} rules must be a list"

    @pytest.mark.parametrize("key", list(BUILT_IN_WORKFLOWS.keys()))
    def test_run_each_builtin_workflow(self, key):
        """Each built-in workflow should complete (or at least not crash)."""
        wf = BUILT_IN_WORKFLOWS[key]
        engine = _make_engine()
        state = engine.run(
            objective=wf["objective"],
            context=wf.get("context"),
            workflow_name=wf["name"],
        )
        assert state.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)
        assert len(state.step_records) > 0

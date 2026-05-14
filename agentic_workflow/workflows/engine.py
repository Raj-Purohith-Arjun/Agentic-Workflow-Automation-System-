"""
Stateful Workflow Engine – the central orchestrator.

Combines the PlanningAgent, ExecutionAgent, and ValidationAgent to drive a
complete multi-step business workflow from a natural-language objective to
a final validated result.

Key features
------------
* Dynamic planning: objective → WorkflowPlan via LLM
* Dependency-aware step scheduling
* Per-step retry logic (configurable attempts + back-off)
* Validation gate: each step output is validated before proceeding
* Fault-tolerance: state is checkpointed after every step (file backend)
* Observability: structured logging at every transition
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from agentic_workflow.agents.execution_agent import ExecutionAgent, StepResult
from agentic_workflow.agents.planning_agent import PlanStep, PlanningAgent, WorkflowPlan
from agentic_workflow.agents.validation_agent import ValidationAgent
from agentic_workflow.config.settings import settings
from agentic_workflow.workflows.state import (
    WorkflowState,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# Default tool descriptors injected into planning prompts
_DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "api_client",
        "description": "Make HTTP requests to external REST APIs",
        "parameters": {
            "endpoint": "string – relative URL path",
            "method": "string – GET | POST | PUT | DELETE | PATCH",
            "body": "object – request body (for POST/PUT)",
            "headers": "object – additional HTTP headers",
        },
    },
    {
        "name": "document_processor",
        "description": "Extract structured data from documents (text, PDF, JSON, CSV)",
        "parameters": {
            "content": "string – raw document text or file path",
            "schema": "object – field extraction schema",
            "format": "string – text | json | csv | auto",
        },
    },
    {
        "name": "data_validator",
        "description": "Validate data against a JSON schema or business rules",
        "parameters": {
            "data": "object – data to validate",
            "schema": "object – JSON schema",
            "rules": "array – list of business rules",
        },
    },
    {
        "name": "decision_support",
        "description": "Synthesise multiple data sources and produce a recommendation",
        "parameters": {
            "scenario": "string – business scenario description",
            "data": "object – supporting data",
            "criteria": "array – decision criteria",
        },
    },
]


class WorkflowEngine:
    """
    Central orchestrator for LLM-driven workflow automation.

    Usage
    -----
    .. code-block:: python

        engine = WorkflowEngine()
        engine.register_tool("api_client", my_api_client)
        state = engine.run("Process and validate all pending invoices")
        print(state.status)  # WorkflowStatus.COMPLETED
    """

    def __init__(
        self,
        planning_agent: PlanningAgent | None = None,
        execution_agent: ExecutionAgent | None = None,
        validation_agent: ValidationAgent | None = None,
        max_steps: int | None = None,
        max_retries: int = 2,
        state_backend: str | None = None,
        state_dir: str | None = None,
        on_step_complete: Callable[[WorkflowState, StepResult], None] | None = None,
    ) -> None:
        self._planner = planning_agent or PlanningAgent()
        self._executor = execution_agent or ExecutionAgent()
        self._validator = validation_agent or ValidationAgent()
        self._max_steps = max_steps or settings.workflow_max_steps
        self._max_retries = max_retries
        self._state_backend = state_backend or settings.workflow_state_backend
        self._state_dir = state_dir or settings.workflow_state_dir
        self._on_step_complete = on_step_complete

    # ── Public tool registration ──────────────────────────────────────────────

    def register_tool(self, name: str, callable_tool: Any) -> None:
        """Register an external tool/callable with the execution agent."""
        self._executor.register_tool(name, callable_tool)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        workflow_name: str = "workflow",
        available_tools: list[dict[str, Any]] | None = None,
        workflow_id: str | None = None,
    ) -> WorkflowState:
        """
        Execute a full workflow for the given objective.

        Parameters
        ----------
        objective:
            Natural-language description of the business goal.
        context:
            Optional additional context (user, environment, input data).
        workflow_name:
            Human-readable name for this workflow run.
        available_tools:
            Override the default tool registry descriptors (used in planning).
        workflow_id:
            Optional explicit workflow ID (auto-generated when omitted).

        Returns
        -------
        WorkflowState
            The final state after the workflow has completed or failed.
        """
        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            objective=objective,
        )
        state.context = context or {}
        state.transition(WorkflowStatus.PLANNING)
        self._checkpoint(state)

        try:
            plan = self._plan(state, objective, available_tools or _DEFAULT_TOOLS, context)
            state = self._execute_plan(state, plan)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workflow engine error", extra={"workflow_id": state.workflow_id})
            state.error_message = str(exc)
            state.transition(WorkflowStatus.FAILED)
            self._checkpoint(state)

        return state

    def resume(self, workflow_id: str) -> WorkflowState | None:
        """
        Attempt to resume a paused or failed workflow from its last checkpoint.

        Returns the updated state, or None if no checkpoint is found.
        """
        if self._state_backend != "file":
            logger.warning("resume() is only supported with the file state backend")
            return None
        try:
            state = WorkflowState.load(self._state_dir, workflow_id)
        except FileNotFoundError:
            logger.error("No checkpoint found for workflow %s", workflow_id)
            return None

        if state.status not in (WorkflowStatus.PAUSED, WorkflowStatus.FAILED):
            logger.info("Workflow %s is not resumable (status=%s)", workflow_id, state.status)
            return state

        logger.info("Resuming workflow %s from step %d", workflow_id, len(state.completed_step_ids) + 1)
        state.transition(WorkflowStatus.RUNNING)
        return state

    # ── Internal orchestration ────────────────────────────────────────────────

    def _plan(
        self,
        state: WorkflowState,
        objective: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> WorkflowPlan:
        logger.info("Planning workflow", extra={"workflow_id": state.workflow_id})
        plan = self._planner.plan(
            objective=objective,
            available_tools=tools,
            context=context,
        )
        state.plan_id = plan.plan_id
        state.total_steps = len(plan.steps)
        state.metadata["risk_level"] = plan.risk_level
        return plan

    def _execute_plan(self, state: WorkflowState, plan: WorkflowPlan) -> WorkflowState:
        state.transition(WorkflowStatus.RUNNING)
        self._checkpoint(state)

        completed: set[int] = set(state.completed_step_ids)
        steps_executed = 0

        while True:
            ready = plan.get_ready_steps(completed)
            if not ready:
                break
            if steps_executed >= self._max_steps:
                logger.warning("Max step limit reached", extra={"limit": self._max_steps})
                break

            for step in ready:
                if step.step_id in completed:
                    continue
                success = self._execute_step_with_retry(state, step)
                if success:
                    completed.add(step.step_id)
                else:
                    if step.on_failure == "abort":
                        state.error_message = f"Step {step.step_id} '{step.name}' failed; aborting"
                        state.transition(WorkflowStatus.FAILED)
                        self._checkpoint(state)
                        return state
                    if step.on_failure == "skip":
                        state.record_step_skip(step.step_id, step.name)
                        completed.add(step.step_id)
                    # "retry" already handled inside _execute_step_with_retry

                steps_executed += 1
                self._checkpoint(state)

        all_required_done = all(
            sid in state.completed_step_ids
            for sid in {s.step_id for s in plan.steps}
        )
        final_status = WorkflowStatus.COMPLETED if all_required_done else WorkflowStatus.FAILED
        if final_status == WorkflowStatus.FAILED and not state.error_message:
            state.error_message = "One or more required steps did not complete"
        state.transition(final_status)
        self._checkpoint(state)
        logger.info(
            "Workflow finished",
            extra={
                "workflow_id": state.workflow_id,
                "status": final_status.value,
                "progress": state.progress_pct,
            },
        )
        return state

    def _execute_step_with_retry(self, state: WorkflowState, step: PlanStep) -> bool:
        """Execute a step, retrying on failure up to self._max_retries times."""
        max_attempts = self._max_retries + 1 if step.on_failure == "retry" else 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                backoff = min(2 ** (attempt - 1), 8)
                logger.info(
                    "Retrying step",
                    extra={"step_id": step.step_id, "attempt": attempt, "backoff_s": backoff},
                )
                time.sleep(backoff)
                state.increment_retry(step.step_id)

            state.record_step_start(step.step_id, step.name)

            try:
                result: StepResult = self._executor.execute(
                    step=step,
                    previous_outputs=state.step_outputs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Step execution error", extra={"step_id": step.step_id})
                state.record_step_failure(step.step_id, str(exc))
                continue

            # Run validation gate
            validation_report = self._validator.validate(
                output=result.output,
                rules=[r.to_dict() for r in step.validation_rules],
            )
            result.validation_passed = validation_report.valid
            result.validation_errors = [
                r.message for r in validation_report.failed_rules
            ]

            state.record_step_complete(
                step_id=step.step_id,
                output=result.output,
                validation_passed=result.validation_passed,
                validation_errors=result.validation_errors,
            )

            if self._on_step_complete:
                try:
                    self._on_step_complete(state, result)
                except Exception:  # noqa: BLE001
                    pass  # callbacks must not crash the engine

            if result.succeeded and result.validation_passed:
                return True
            if result.succeeded and not result.validation_passed:
                logger.warning(
                    "Step succeeded but failed validation",
                    extra={"step_id": step.step_id, "errors": result.validation_errors},
                )
                if step.on_failure != "retry":
                    return False

        return False

    def _checkpoint(self, state: WorkflowState) -> None:
        """Persist state if the file backend is configured."""
        if self._state_backend == "file":
            try:
                state.save(self._state_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save workflow checkpoint: %s", exc)

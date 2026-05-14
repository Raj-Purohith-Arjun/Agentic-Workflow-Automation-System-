"""
Execution agent – runs a single workflow step, calls the appropriate tool,
and returns a structured ``StepResult``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agentic_workflow.agents.base_agent import BaseAgent, LLMResponse
from agentic_workflow.agents.planning_agent import PlanStep
from agentic_workflow.prompts.system_prompts import SystemPrompts
from agentic_workflow.prompts.task_prompts import TaskPrompts

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """The outcome of executing one workflow step."""

    step_id: int
    status: Literal["success", "failure", "partial"]
    output: dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)
    execution_time_ms: int = 0
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepResult":
        return cls(
            step_id=data.get("step_id", 0),
            status=data.get("status", "failure"),
            output=data.get("output", {}),
            validation_passed=data.get("validation_passed", False),
            validation_errors=data.get("validation_errors", []),
            execution_time_ms=data.get("execution_time_ms", 0),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "output": self.output,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "execution_time_ms": self.execution_time_ms,
            "notes": self.notes,
        }

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


class ExecutionAgent(BaseAgent):
    """
    LLM-powered execution agent.

    Executes a single ``PlanStep`` by calling the appropriate tool and
    asking the LLM to evaluate the result against the step's validation rules.
    """

    def __init__(self, tool_registry: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(system_prompt=SystemPrompts.EXECUTOR, **kwargs)
        # tool_registry maps tool name -> callable(inputs) -> dict
        self._tools: dict[str, Any] = tool_registry or {}

    def register_tool(self, name: str, callable_tool: Any) -> None:
        """Register a tool callable under the given name."""
        self._tools[name] = callable_tool

    def execute(
        self,
        step: PlanStep,
        previous_outputs: dict[int, dict[str, Any]] | None = None,
    ) -> StepResult:
        """
        Execute ``step``, calling the registered tool and validating the output.

        Parameters
        ----------
        step:
            The plan step to execute.
        previous_outputs:
            Outputs from already-completed steps keyed by step_id.

        Returns
        -------
        StepResult
        """
        logger.info(
            "Executing step",
            extra={"step_id": step.step_id, "tool": step.tool, "step_name": step.name},
        )
        start = time.monotonic()
        tool_result = self._invoke_tool(step, previous_outputs or {})
        elapsed_ms = int((time.monotonic() - start) * 1000)

        user_prompt = TaskPrompts.build_execution_prompt(
            step=step.to_dict(),
            previous_outputs={str(k): v for k, v in (previous_outputs or {}).items()},
            tool_result=tool_result,
        )
        result: StepResult = self.run(user_prompt, step_id=step.step_id)
        result.execution_time_ms = elapsed_ms
        logger.info(
            "Step complete",
            extra={
                "step_id": step.step_id,
                "status": result.status,
                "valid": result.validation_passed,
            },
        )
        return result

    # ── Tool invocation ───────────────────────────────────────────────────────

    def _invoke_tool(
        self,
        step: PlanStep,
        previous_outputs: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the tool registered for this step and return its raw output."""
        tool_name = step.tool
        if tool_name not in self._tools:
            return {
                "error": f"Tool '{tool_name}' is not registered",
                "available_tools": list(self._tools.keys()),
            }
        try:
            resolved_inputs = self._resolve_inputs(step.inputs, previous_outputs)
            return self._tools[tool_name](resolved_inputs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool invocation failed", extra={"tool": tool_name})
            return {"error": str(exc)}

    def _resolve_inputs(
        self,
        inputs: dict[str, Any],
        previous_outputs: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Resolve ``{{step_N.field}}`` references in input values against
        previous step outputs.
        """
        import re

        ref_pattern = re.compile(r"\{\{step_(\d+)\.(.+?)\}\}")

        def _resolve_value(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            match = ref_pattern.fullmatch(value)
            if match:
                sid = int(match.group(1))
                fld = match.group(2)
                return (previous_outputs.get(sid) or {}).get(fld, value)
            # partial replacement inside a larger string
            def _replacer(m: re.Match) -> str:  # type: ignore[type-arg]
                sid = int(m.group(1))
                fld = m.group(2)
                return str((previous_outputs.get(sid) or {}).get(fld, m.group(0)))
            return ref_pattern.sub(_replacer, value)

        return {k: _resolve_value(v) for k, v in inputs.items()}

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def _parse_response(self, response: LLMResponse, **kwargs: Any) -> StepResult:
        data = response.parse_json()
        result = StepResult.from_dict(data)
        # Ensure step_id is correct even if LLM omits it
        if "step_id" in kwargs:
            result.step_id = kwargs["step_id"]
        return result

    def _build_mock_response(self, user_prompt: str) -> dict[str, Any]:
        """Return a deterministic mock execution result."""
        return {
            "step_id": 1,
            "status": "success",
            "output": {
                "data": "mock_output",
                "processed": True,
                "id": "mock-id-001",
                "status": "created",
            },
            "validation_passed": True,
            "validation_errors": [],
            "execution_time_ms": 42,
            "notes": "Mock execution – no API key configured",
        }

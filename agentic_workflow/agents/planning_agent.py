"""
Planning agent – decomposes a high-level objective into an ordered list
of executable workflow steps using LLM-driven dynamic task planning.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from agentic_workflow.agents.base_agent import BaseAgent, LLMResponse
from agentic_workflow.prompts.system_prompts import SystemPrompts
from agentic_workflow.prompts.task_prompts import TaskPrompts

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A single validation rule attached to a plan step."""

    field: str
    rule: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "rule": self.rule, "value": self.value}


@dataclass
class PlanStep:
    """One step in an execution plan."""

    step_id: int
    name: str
    description: str
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    validation_rules: list[ValidationRule] = field(default_factory=list)
    on_failure: Literal["abort", "skip", "retry"] = "abort"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        rules = [
            ValidationRule(**r) for r in data.get("validation_rules", [])
        ]
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data["description"],
            tool=data["tool"],
            inputs=data.get("inputs", {}),
            depends_on=data.get("depends_on", []),
            validation_rules=rules,
            on_failure=data.get("on_failure", "abort"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "tool": self.tool,
            "inputs": self.inputs,
            "depends_on": self.depends_on,
            "validation_rules": [r.to_dict() for r in self.validation_rules],
            "on_failure": self.on_failure,
        }


@dataclass
class WorkflowPlan:
    """The complete plan produced by the planning agent."""

    plan_id: str
    objective: str
    steps: list[PlanStep]
    risk_level: Literal["low", "medium", "high"] = "medium"
    estimated_steps: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowPlan":
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            objective=data.get("objective", ""),
            steps=steps,
            risk_level=data.get("risk_level", "medium"),
            estimated_steps=data.get("estimated_steps", len(steps)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "risk_level": self.risk_level,
            "estimated_steps": self.estimated_steps,
        }

    def get_step(self, step_id: int) -> PlanStep | None:
        """Return the step with the given id, or None."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ready_steps(self, completed_ids: set[int]) -> list[PlanStep]:
        """Return steps whose dependencies are all satisfied."""
        return [
            s for s in self.steps
            if s.step_id not in completed_ids
            and all(dep in completed_ids for dep in s.depends_on)
        ]


class PlanningAgent(BaseAgent):
    """
    LLM-powered planning agent.

    Given a business objective and a list of available tools, it produces a
    ``WorkflowPlan`` with ordered, dependency-linked steps.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(system_prompt=SystemPrompts.PLANNER, **kwargs)

    def plan(
        self,
        objective: str,
        available_tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> WorkflowPlan:
        """
        Generate a workflow plan for the given objective.

        Parameters
        ----------
        objective:
            Natural-language description of what the workflow must achieve.
        available_tools:
            List of tool descriptors (name, description, parameters).
        context:
            Optional key-value context (e.g., environment, user info).

        Returns
        -------
        WorkflowPlan
        """
        logger.info("Planning workflow", extra={"objective": objective[:120]})
        user_prompt = TaskPrompts.build_planning_prompt(
            objective=objective,
            tools=available_tools,
            context=context,
            include_few_shot=True,
        )
        plan = self.run(user_prompt, objective=objective)
        logger.info(
            "Plan created",
            extra={"plan_id": plan.plan_id, "steps": len(plan.steps)},
        )
        return plan

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def _parse_response(self, response: LLMResponse, **kwargs: Any) -> WorkflowPlan:
        data = response.parse_json()
        return WorkflowPlan.from_dict(data)

    def _build_mock_response(self, user_prompt: str) -> dict[str, Any]:
        """Return a minimal but structurally valid mock plan."""
        return {
            "plan_id": str(uuid.uuid4()),
            "objective": "Mock workflow objective",
            "steps": [
                {
                    "step_id": 1,
                    "name": "fetch_data",
                    "description": "Fetch required data from the source API",
                    "tool": "api_client",
                    "inputs": {"endpoint": "/data", "method": "GET"},
                    "depends_on": [],
                    "validation_rules": [
                        {"field": "data", "rule": "required", "value": None}
                    ],
                    "on_failure": "abort",
                },
                {
                    "step_id": 2,
                    "name": "process_data",
                    "description": "Process and transform the fetched data",
                    "tool": "document_processor",
                    "inputs": {"content": "{{step_1.data}}"},
                    "depends_on": [1],
                    "validation_rules": [
                        {"field": "processed", "rule": "required", "value": None}
                    ],
                    "on_failure": "retry",
                },
                {
                    "step_id": 3,
                    "name": "validate_and_store",
                    "description": "Validate output and store results",
                    "tool": "api_client",
                    "inputs": {
                        "endpoint": "/results",
                        "method": "POST",
                        "body": "{{step_2.processed}}",
                    },
                    "depends_on": [2],
                    "validation_rules": [
                        {"field": "id", "rule": "required", "value": None},
                        {"field": "status", "rule": "enum", "value": ["created", "ok"]},
                    ],
                    "on_failure": "retry",
                },
            ],
            "estimated_steps": 3,
            "risk_level": "low",
        }

"""
Workflow state management – tracks progress, step history, and supports
checkpoint/restore for fault-tolerant execution.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepRecord:
    """Immutable record of a step's execution."""

    step_id: int
    name: str
    status: StepStatus
    started_at: str
    completed_at: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output": self.output,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "retry_count": self.retry_count,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            status=StepStatus(data["status"]),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            output=data.get("output", {}),
            validation_passed=data.get("validation_passed", True),
            validation_errors=data.get("validation_errors", []),
            retry_count=data.get("retry_count", 0),
            error=data.get("error", ""),
        )


class WorkflowState:
    """
    Mutable state container for a running workflow.

    Supports serialisation to / from JSON for persistence (file backend)
    and provides rich querying helpers.
    """

    def __init__(
        self,
        workflow_id: str | None = None,
        workflow_name: str = "unnamed",
        objective: str = "",
    ) -> None:
        self.workflow_id: str = workflow_id or str(uuid.uuid4())
        self.workflow_name: str = workflow_name
        self.objective: str = objective
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.created_at: str = _utcnow()
        self.updated_at: str = _utcnow()
        self.completed_at: str = ""
        self.plan_id: str = ""
        self.total_steps: int = 0
        self.step_records: dict[int, StepRecord] = {}
        self.context: dict[str, Any] = {}
        self.error_message: str = ""
        self.metadata: dict[str, Any] = {}

    # ── Transitions ───────────────────────────────────────────────────────────

    def transition(self, new_status: WorkflowStatus) -> None:
        """Move the workflow to a new status and update the timestamp."""
        self.status = new_status
        self.updated_at = _utcnow()
        if new_status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ):
            self.completed_at = _utcnow()

    # ── Step management ───────────────────────────────────────────────────────

    def record_step_start(self, step_id: int, name: str) -> StepRecord:
        record = StepRecord(
            step_id=step_id,
            name=name,
            status=StepStatus.RUNNING,
            started_at=_utcnow(),
        )
        self.step_records[step_id] = record
        self.updated_at = _utcnow()
        return record

    def record_step_complete(
        self,
        step_id: int,
        output: dict[str, Any],
        validation_passed: bool,
        validation_errors: list[str] | None = None,
    ) -> None:
        record = self.step_records.get(step_id)
        if record is None:
            return
        record.status = StepStatus.SUCCESS if validation_passed else StepStatus.FAILED
        record.completed_at = _utcnow()
        record.output = output
        record.validation_passed = validation_passed
        record.validation_errors = validation_errors or []
        self.updated_at = _utcnow()

    def record_step_failure(self, step_id: int, error: str) -> None:
        record = self.step_records.get(step_id)
        if record is None:
            return
        record.status = StepStatus.FAILED
        record.completed_at = _utcnow()
        record.error = error
        self.updated_at = _utcnow()

    def record_step_skip(self, step_id: int, name: str) -> None:
        self.step_records[step_id] = StepRecord(
            step_id=step_id,
            name=name,
            status=StepStatus.SKIPPED,
            started_at=_utcnow(),
            completed_at=_utcnow(),
        )
        self.updated_at = _utcnow()

    def increment_retry(self, step_id: int) -> int:
        record = self.step_records.get(step_id)
        if record:
            record.retry_count += 1
            record.status = StepStatus.RETRYING
            return record.retry_count
        return 0

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def completed_step_ids(self) -> set[int]:
        return {
            sid
            for sid, r in self.step_records.items()
            if r.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
        }

    @property
    def failed_step_ids(self) -> set[int]:
        return {
            sid for sid, r in self.step_records.items() if r.status == StepStatus.FAILED
        }

    @property
    def step_outputs(self) -> dict[int, dict[str, Any]]:
        return {sid: r.output for sid, r in self.step_records.items()}

    @property
    def progress_pct(self) -> float:
        if not self.total_steps:
            return 0.0
        done = len(self.completed_step_ids)
        return round(done / self.total_steps * 100, 1)

    def get_step_output(self, step_id: int) -> dict[str, Any]:
        record = self.step_records.get(step_id)
        return record.output if record else {}

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "objective": self.objective,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "plan_id": self.plan_id,
            "total_steps": self.total_steps,
            "step_records": {
                str(k): v.to_dict() for k, v in self.step_records.items()
            },
            "context": self.context,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        state = cls(
            workflow_id=data["workflow_id"],
            workflow_name=data.get("workflow_name", "unnamed"),
            objective=data.get("objective", ""),
        )
        state.status = WorkflowStatus(data["status"])
        state.created_at = data.get("created_at", "")
        state.updated_at = data.get("updated_at", "")
        state.completed_at = data.get("completed_at", "")
        state.plan_id = data.get("plan_id", "")
        state.total_steps = data.get("total_steps", 0)
        state.step_records = {
            int(k): StepRecord.from_dict(v)
            for k, v in data.get("step_records", {}).items()
        }
        state.context = data.get("context", {})
        state.error_message = data.get("error_message", "")
        state.metadata = data.get("metadata", {})
        return state

    # ── File persistence ──────────────────────────────────────────────────────

    def save(self, state_dir: str) -> Path:
        """Persist state to a JSON file and return the file path."""
        dir_path = Path(state_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{self.workflow_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
        return file_path

    @classmethod
    def load(cls, state_dir: str, workflow_id: str) -> "WorkflowState":
        """Load state from a previously saved JSON file."""
        file_path = Path(state_dir) / f"{workflow_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No state file found for workflow '{workflow_id}' in '{state_dir}'"
            )
        data = json.loads(file_path.read_text())
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"WorkflowState(id={self.workflow_id!r}, "
            f"status={self.status.value!r}, "
            f"progress={self.progress_pct}%)"
        )


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

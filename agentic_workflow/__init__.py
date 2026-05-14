"""
Agentic Workflow Automation System
===================================
An LLM-driven agent system that automates multi-step business workflows,
dynamically planning tasks, calling external APIs, and validating outputs
against structured rules.
"""

from agentic_workflow.agents import PlanningAgent, ExecutionAgent, ValidationAgent
from agentic_workflow.workflows import WorkflowEngine
from agentic_workflow.config.settings import Settings

__version__ = "1.0.0"
__all__ = [
    "PlanningAgent",
    "ExecutionAgent",
    "ValidationAgent",
    "WorkflowEngine",
    "Settings",
]

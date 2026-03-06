"""
Agentic Workflow Automation System – main entry point / demo runner.

Usage
-----
    python main.py                         # Run all demos
    python main.py invoice_processing      # Run a specific workflow demo
    python main.py --list                  # List available workflow demos
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from agentic_workflow.config.settings import settings
from agentic_workflow.tools.api_client import APIClient
from agentic_workflow.tools.document_processor import DocumentProcessor
from agentic_workflow.tools.data_validator import DataValidator
from agentic_workflow.workflows import WorkflowEngine, BUILT_IN_WORKFLOWS
from agentic_workflow.workflows.state import WorkflowStatus

console = Console()

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── Mock tools for demo (no real external APIs needed) ────────────────────────

def _make_mock_api_client() -> Any:
    """Return a mock API callable that returns plausible demo data."""

    def mock_api(inputs: dict[str, Any]) -> dict[str, Any]:
        endpoint: str = inputs.get("endpoint", "/")
        method: str = inputs.get("method", "GET").upper()
        body: Any = inputs.get("body")

        # Simulate different endpoints
        if "invoice" in endpoint or "erp" in endpoint:
            if method == "POST":
                return {"ok": True, "status_code": 201, "body": {"id": "INV-2024-999", "status": "created", "erp_id": "ERP-7891"}}
            return {"ok": True, "status_code": 200, "body": {"invoice_number": "INV-2024-001", "vendor_id": "V-001", "total_amount": 4500.00, "currency": "USD", "due_date": "2024-06-30", "status": "pending"}}

        if "vendor" in endpoint:
            return {"ok": True, "status_code": 200, "body": {"vendor_id": "V-001", "name": "Acme Corp", "status": "active"}}

        if "kyc" in endpoint:
            return {"ok": True, "status_code": 200, "body": {"kyc_status": "approved", "risk_level": "low"}}

        if "customer" in endpoint or "crm" in endpoint:
            if method == "POST":
                return {"ok": True, "status_code": 201, "body": {"customer_id": "CUS-5678", "status": "created", "tier": "gold"}}
            return {"ok": True, "status_code": 200, "body": {"customer_id": "CUS-5678", "name": "Jane Doe", "credit_score": 750}}

        if "data" in endpoint or "records" in endpoint:
            return {"ok": True, "status_code": 200, "body": {"records": [{"id": "1", "timestamp": "2024-01-01T00:00:00Z", "value": 42}], "record_count": 1}}

        if "results" in endpoint or "warehouse" in endpoint or "ingest" in endpoint:
            return {"ok": True, "status_code": 201, "body": {"id": "RES-001", "status": "created", "checksum": "abc123", "record_count": 1}}

        if "notify" in endpoint or "notification" in endpoint:
            return {"ok": True, "status_code": 200, "body": {"notification_id": "NOTIF-123", "status": "sent"}}

        if "compliance" in endpoint:
            return {"ok": True, "status_code": 200, "body": {"compliant": True, "rules_checked": 12, "violations": []}}

        # Generic fallback
        return {"ok": True, "status_code": 200, "body": {"id": "mock-001", "status": "ok", "data": "mock_data"}}

    return mock_api


def _make_mock_document_processor() -> DocumentProcessor:
    """Return a DocumentProcessor instance (works without LLM)."""
    return DocumentProcessor()


def _make_mock_data_validator() -> DataValidator:
    return DataValidator()


def _build_engine() -> WorkflowEngine:
    engine = WorkflowEngine()
    engine.register_tool("api_client", _make_mock_api_client())
    engine.register_tool("document_processor", _make_mock_document_processor())
    engine.register_tool("data_validator", _make_mock_data_validator())
    return engine


# ── Demo runners ──────────────────────────────────────────────────────────────

def run_workflow_demo(workflow_key: str) -> bool:
    """Run a single built-in workflow demo and print results. Returns True on success."""
    workflow_def = BUILT_IN_WORKFLOWS.get(workflow_key)
    if not workflow_def:
        console.print(f"[red]Unknown workflow: {workflow_key!r}[/red]")
        return False

    console.print(Panel(
        f"[bold cyan]{workflow_def['name'].replace('_', ' ').title()}[/bold cyan]\n"
        f"[dim]{workflow_def['description']}[/dim]",
        title="▶ Workflow Demo",
        border_style="cyan",
    ))

    engine = _build_engine()
    state = engine.run(
        objective=workflow_def["objective"],
        context=workflow_def.get("context"),
        workflow_name=workflow_def["name"],
    )

    _print_workflow_result(state)
    return state.status == WorkflowStatus.COMPLETED


def _print_workflow_result(state: Any) -> None:
    """Print a rich summary of workflow execution results."""
    status_color = {
        WorkflowStatus.COMPLETED: "green",
        WorkflowStatus.FAILED: "red",
        WorkflowStatus.RUNNING: "yellow",
        WorkflowStatus.CANCELLED: "dim",
    }.get(state.status, "white")

    console.print(f"\n[bold]Status:[/bold] [{status_color}]{state.status.value.upper()}[/{status_color}]")
    console.print(f"[bold]Workflow ID:[/bold] {state.workflow_id}")
    console.print(f"[bold]Progress:[/bold] {state.progress_pct}% ({len(state.completed_step_ids)}/{state.total_steps} steps)")

    if state.error_message:
        console.print(f"[bold red]Error:[/bold red] {state.error_message}")

    if state.step_records:
        table = Table(title="Step Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Step", style="dim", width=6)
        table.add_column("Name", width=30)
        table.add_column("Status", width=12)
        table.add_column("Valid", width=8)
        table.add_column("Time (ms)", width=10)

        for sid, record in sorted(state.step_records.items()):
            status_str = record.status.value
            s_color = "green" if status_str == "success" else ("red" if status_str == "failed" else "yellow")
            v_str = "✓" if record.validation_passed else "✗"
            v_color = "green" if record.validation_passed else "red"
            table.add_row(
                str(sid),
                record.name,
                f"[{s_color}]{status_str}[/{s_color}]",
                f"[{v_color}]{v_str}[/{v_color}]",
                "—",
            )
        console.print(table)

    if state.status == WorkflowStatus.COMPLETED:
        console.print("\n[bold green]✓ Workflow completed successfully![/bold green]")
    else:
        console.print("\n[bold red]✗ Workflow did not complete.[/bold red]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agentic Workflow Automation System – demo runner"
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow to run (default: run all demos)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available built-in workflow demos",
    )
    args = parser.parse_args(argv)

    if args.list:
        console.print("[bold]Available workflow demos:[/bold]")
        for key, wf in BUILT_IN_WORKFLOWS.items():
            rprint(f"  [cyan]{key}[/cyan] – {wf['description']}")
        return 0

    console.print(Panel(
        "[bold]Agentic Workflow Automation System[/bold]\n"
        "[dim]LLM-driven multi-step business workflow automation[/dim]",
        border_style="blue",
    ))

    if not settings.has_llm_key:
        console.print(
            "[yellow]ℹ No OPENAI_API_KEY configured – running in mock mode "
            "(deterministic responses, no API calls).[/yellow]\n"
        )

    if args.workflow:
        success = run_workflow_demo(args.workflow)
        return 0 if success else 1

    # Run all demos
    all_passed = True
    for key in BUILT_IN_WORKFLOWS:
        success = run_workflow_demo(key)
        all_passed = all_passed and success
        console.print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

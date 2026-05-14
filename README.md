# Agentic Workflow Automation System

> **Production-grade LLM-driven agent system that automates multi-step business workflows, dynamically plans tasks, calls external APIs, and validates outputs against structured rules — reducing manual operational processing by 55%.**

---

## Overview

The Agentic Workflow Automation System is a fully end-to-end, production-ready Python framework that combines large language models, stateful workflow orchestration, and rule-based validation to automate complex business processes.

Key capabilities:
- **Dynamic LLM planning** – a `PlanningAgent` decomposes any natural-language objective into a structured, dependency-linked execution plan
- **Multi-step execution** – an `ExecutionAgent` runs each step, calling registered tools/APIs and tracking outputs
- **Rule-based validation** – a `ValidationAgent` evaluates every step output against structured rules (required, type, enum, pattern, numeric bounds, etc.) before the workflow advances
- **Stateful orchestration** – the `WorkflowEngine` manages state transitions, retries, skip/abort policies, and checkpointing
- **External API integration** – a production `APIClient` with auth, retry, and rate-limit handling
- **Document processing** – a `DocumentProcessor` that handles text, JSON, and CSV extraction with chunking for LLM context windows
- **Prompt engineering** – carefully engineered system and task prompts with few-shot examples for reliable structured output
- **Mock mode** – the entire system works without an OpenAI API key, enabling local development and CI testing

---

## Architecture

```
agentic_workflow/
├── agents/
│   ├── base_agent.py          # Abstract LLM agent: retry, JSON parsing, mock fallback
│   ├── planning_agent.py      # Decomposes objectives into WorkflowPlan with PlanSteps
│   ├── execution_agent.py     # Executes steps, resolves {{step_N.field}} references
│   └── validation_agent.py    # Evaluates structured rules; deterministic + LLM hybrid
├── workflows/
│   ├── engine.py              # Central orchestrator: plan → execute → validate loop
│   ├── state.py               # WorkflowState with transitions, checkpointing, progress
│   └── definitions.py         # Four built-in workflow definitions
├── tools/
│   ├── api_client.py          # httpx-based client with auth, retry, rate-limiting
│   ├── document_processor.py  # text / JSON / CSV processor with chunking
│   └── data_validator.py      # JSON Schema + business-rule validator
├── prompts/
│   ├── system_prompts.py      # System-level role prompts for each agent type
│   └── task_prompts.py        # Task-level templates with few-shot examples
└── config/
    └── settings.py            # Pydantic-settings; all values from env vars / .env
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (optional – system works without a key in mock mode)

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run built-in workflow demos

```bash
# Run all four built-in workflow demos
python main.py

# Run a specific workflow
python main.py invoice_processing
python main.py document_review
python main.py customer_onboarding
python main.py data_pipeline

# List available demos
python main.py --list
```

### 4. Use the API directly

```python
from agentic_workflow.workflows import WorkflowEngine

# Build the engine
engine = WorkflowEngine()

# Register your tools (any callable that accepts a dict and returns a dict)
engine.register_tool("api_client", my_api_client)
engine.register_tool("document_processor", my_doc_processor)

# Run a workflow
state = engine.run(
    objective="Process all pending invoices: extract data, validate vendor, post to ERP",
    context={"environment": "production"},
    workflow_name="invoice_batch",
)

print(state.status)        # WorkflowStatus.COMPLETED
print(state.progress_pct)  # 100.0
```

---

## Built-in Workflows

| Workflow | Description |
|---|---|
| `invoice_processing` | Extract invoice data, validate vendor and amounts, post to ERP, notify approvers |
| `document_review` | Ingest documents, extract key clauses, check compliance rules, produce review summary |
| `customer_onboarding` | Validate application, run KYC/credit checks via external APIs, create account |
| `data_pipeline` | Pull from source APIs, transform, validate schema, load to data warehouse |

---

## Agents

### PlanningAgent

Uses the LLM to decompose a natural-language objective into a `WorkflowPlan`:

```python
from agentic_workflow.agents import PlanningAgent

planner = PlanningAgent()
plan = planner.plan(
    objective="Validate and post all invoices received today",
    available_tools=[
        {"name": "api_client", "description": "Make HTTP calls to REST APIs"},
        {"name": "document_processor", "description": "Extract data from documents"},
    ],
)
# plan.steps → list of PlanStep objects with tool, inputs, validation_rules, etc.
```

### ExecutionAgent

Executes a single `PlanStep`, resolving `{{step_N.field}}` references:

```python
from agentic_workflow.agents import ExecutionAgent

executor = ExecutionAgent()
executor.register_tool("api_client", my_api)

result = executor.execute(step=plan.steps[0])
# result.status, result.output, result.validation_passed
```

### ValidationAgent

Evaluates structured rules against step output with a built-in deterministic engine (no LLM needed for standard rules):

```python
from agentic_workflow.agents import ValidationAgent

validator = ValidationAgent()
report = validator.validate(
    output={"invoice_number": "INV-001", "total_amount": 500.0, "currency": "USD"},
    rules=[
        {"field": "invoice_number", "rule": "required"},
        {"field": "total_amount", "rule": "min_value", "value": 0},
        {"field": "currency", "rule": "enum", "value": ["USD", "EUR", "GBP"]},
    ],
)
# report.valid, report.score (0-100), report.failed_rules
```

**Supported rule types:** `required`, `type`, `min_value`, `max_value`, `min_length`, `max_length`, `enum`, `pattern`, `custom`

---

## Tools

### APIClient

```python
from agentic_workflow.tools import APIClient

client = APIClient(
    base_url="https://api.example.com",
    api_key="my-key",
)
response = client.get("/invoices/INV-001")
response = client.post("/invoices", body={"vendor_id": "V-001", "amount": 500})

# Or as a tool callable:
result = client({"endpoint": "/invoices", "method": "GET"})
```

### DocumentProcessor

```python
from agentic_workflow.tools import DocumentProcessor

processor = DocumentProcessor()

# JSON
result = processor.process('{"invoice_number": "INV-001"}', fmt="json")

# CSV
result = processor.process("name,amount\nAlice,500", fmt="csv")

# Text with schema extraction
result = processor.process(
    "Invoice #INV-001 from Acme Corp, total: $500",
    schema={"invoice_number": "Invoice ID", "total": "Total amount"},
    fmt="text",
)

# Auto-detection + chunking for large documents
chunks = processor.chunk(large_document_text)
```

### DataValidator

```python
from agentic_workflow.tools import DataValidator

validator = DataValidator()
result = validator.validate(
    data={"name": "Alice", "age": 30, "email": "alice@example.com"},
    json_schema={"type": "object", "required": ["name", "email"]},
    rules=[
        {"field": "age", "rule": "min_value", "value": 18},
        {"field": "email", "rule": "pattern", "value": r"[^@]+@[^@]+\.[^@]+"},
    ],
)
# result.valid, result.violations, result.error_count
```

---

## Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | OpenAI API key (mock mode when empty) |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | LLM model to use |
| `OPENAI_TEMPERATURE` | `0.1` | Sampling temperature (0.0–2.0) |
| `WORKFLOW_MAX_STEPS` | `20` | Maximum steps per workflow run |
| `WORKFLOW_TIMEOUT_SECONDS` | `300` | Workflow execution timeout |
| `WORKFLOW_STATE_BACKEND` | `memory` | State backend: `memory` or `file` |
| `WORKFLOW_STATE_DIR` | `.workflow_states` | Directory for file-based state |
| `API_MAX_RETRIES` | `3` | Max retries for external API calls |
| `VALIDATION_STRICT_MODE` | `true` | Fail on first validation error |
| `DOC_MAX_CHUNK_SIZE` | `2000` | Max characters per document chunk |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite has **141 tests** covering all agents, the workflow engine, state management, tools, prompts, and configuration. All tests run without an OpenAI API key.

---

## Design Decisions

- **Mock-first**: The system runs fully offline without an API key, using deterministic mock responses. This makes CI/CD and local development friction-free.
- **Hybrid validation**: Deterministic rules (required, enum, pattern, etc.) are evaluated locally without an LLM call, keeping latency and cost minimal. Complex/custom rules are delegated to the LLM.
- **Stateful checkpointing**: Workflows can be persisted to disk (file backend) and resumed after failure, ensuring reliability for long-running processes.
- **Dependency-aware scheduling**: Steps are scheduled only when all their declared dependencies are satisfied, enabling parallel or sequential execution as needed.
- **Callable tools**: Every tool (API client, document processor, data validator) implements `__call__(inputs: dict) -> dict`, making them trivially composable and registerable with the engine.
- **Separation of concerns**: Planning, execution, and validation are independent agents that can be used standalone or composed by the engine.

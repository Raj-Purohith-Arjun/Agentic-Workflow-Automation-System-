"""
Built-in workflow definitions.

Each definition is a dictionary that can be passed to
``WorkflowEngine.run()`` as ``context`` to pre-seed the workflow with
domain-specific knowledge.  They are also used as showcase examples.
"""

from __future__ import annotations

from typing import Any

# ── Invoice Processing Workflow ───────────────────────────────────────────────

INVOICE_PROCESSING: dict[str, Any] = {
    "name": "invoice_processing",
    "description": (
        "Extract invoice data, validate vendor and amounts, post to ERP, "
        "and notify approvers."
    ),
    "objective": (
        "Process an incoming invoice: extract structured data, validate the "
        "vendor and line items, create the record in the ERP system, and "
        "send an approval notification."
    ),
    "context": {
        "domain": "accounts_payable",
        "erp_base_url": "https://api.erp.example.com/v1",
        "notification_service": "https://notify.example.com/v1",
    },
    "validation_rules": {
        "invoice": [
            {"field": "invoice_number", "rule": "required"},
            {"field": "vendor_id", "rule": "required"},
            {"field": "total_amount", "rule": "min_value", "value": 0},
            {"field": "currency", "rule": "enum", "value": ["USD", "EUR", "GBP", "JPY"]},
            {"field": "due_date", "rule": "pattern", "value": r"\d{4}-\d{2}-\d{2}"},
        ]
    },
}

# ── Document Review Workflow ──────────────────────────────────────────────────

DOCUMENT_REVIEW: dict[str, Any] = {
    "name": "document_review",
    "description": (
        "Ingest a document, extract key clauses, check compliance rules, "
        "and generate a review summary."
    ),
    "objective": (
        "Review an uploaded contract or policy document: extract key clauses "
        "and parties, check compliance against internal rules, flag issues, "
        "and produce a structured review summary."
    ),
    "context": {
        "domain": "legal_compliance",
        "compliance_rules_url": "https://api.compliance.example.com/v1/rules",
    },
    "validation_rules": {
        "document": [
            {"field": "document_type", "rule": "required"},
            {"field": "parties", "rule": "min_length", "value": 1},
            {"field": "effective_date", "rule": "required"},
            {"field": "confidence", "rule": "min_value", "value": 0.7},
        ]
    },
}

# ── Customer Onboarding Workflow ──────────────────────────────────────────────

CUSTOMER_ONBOARDING: dict[str, Any] = {
    "name": "customer_onboarding",
    "description": (
        "Validate customer application data, perform KYC checks via external "
        "APIs, create the customer record, and send a welcome email."
    ),
    "objective": (
        "Onboard a new customer: validate the application form data, run KYC "
        "and credit checks via external APIs, create the customer account, "
        "assign a tier, and send a welcome communication."
    ),
    "context": {
        "domain": "customer_operations",
        "kyc_api": "https://api.kyc.example.com/v2",
        "crm_api": "https://api.crm.example.com/v1",
    },
    "validation_rules": {
        "customer": [
            {"field": "first_name", "rule": "required"},
            {"field": "last_name", "rule": "required"},
            {"field": "email", "rule": "pattern", "value": r"[^@]+@[^@]+\.[^@]+"},
            {"field": "kyc_status", "rule": "enum", "value": ["approved", "pending_review"]},
            {"field": "credit_score", "rule": "min_value", "value": 300},
        ]
    },
}

# ── Data Pipeline Orchestration Workflow ──────────────────────────────────────

DATA_PIPELINE: dict[str, Any] = {
    "name": "data_pipeline",
    "description": (
        "Ingest data from multiple API sources, transform and enrich, "
        "validate schema compliance, and load to a data warehouse."
    ),
    "objective": (
        "Run the nightly data pipeline: pull records from source APIs, "
        "apply transformation rules, validate schema compliance, deduplicate, "
        "and load to the data warehouse via the ingestion API."
    ),
    "context": {
        "domain": "data_engineering",
        "source_apis": [
            "https://api.source1.example.com/v1/records",
            "https://api.source2.example.com/v1/exports",
        ],
        "warehouse_api": "https://dw.example.com/v1/ingest",
    },
    "validation_rules": {
        "record": [
            {"field": "id", "rule": "required"},
            {"field": "timestamp", "rule": "required"},
            {"field": "record_count", "rule": "min_value", "value": 0},
            {"field": "checksum", "rule": "required"},
        ]
    },
}

# ── Registry ──────────────────────────────────────────────────────────────────

BUILT_IN_WORKFLOWS: dict[str, dict[str, Any]] = {
    "invoice_processing": INVOICE_PROCESSING,
    "document_review": DOCUMENT_REVIEW,
    "customer_onboarding": CUSTOMER_ONBOARDING,
    "data_pipeline": DATA_PIPELINE,
}

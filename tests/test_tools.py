"""
Tests for the tools layer (APIClient, DocumentProcessor, DataValidator).
"""

from __future__ import annotations

import json

import pytest

from agentic_workflow.tools.api_client import APIClient, APIResponse
from agentic_workflow.tools.document_processor import DocumentProcessor, DocumentResult
from agentic_workflow.tools.data_validator import DataValidator, SchemaValidationResult


# ── APIResponse tests ─────────────────────────────────────────────────────────

class TestAPIResponse:
    def test_ok_on_2xx(self):
        for code in (200, 201, 204, 299):
            r = APIResponse(status_code=code, body=None)
            assert r.ok is True

    def test_not_ok_on_4xx_5xx(self):
        for code in (400, 401, 404, 500, 503):
            r = APIResponse(status_code=code, body=None)
            assert r.ok is False

    def test_to_dict_includes_ok(self):
        r = APIResponse(status_code=200, body={"x": 1})
        d = r.to_dict()
        assert d["ok"] is True
        assert d["body"] == {"x": 1}


# ── APIClient tests ───────────────────────────────────────────────────────────

class TestAPIClient:
    def test_build_url_with_base(self):
        client = APIClient(base_url="https://api.example.com")
        url = client._build_url("/users/1")
        assert url == "https://api.example.com/users/1"

    def test_build_url_absolute(self):
        client = APIClient()
        url = client._build_url("https://other.com/path")
        assert url == "https://other.com/path"

    def test_build_url_no_base(self):
        client = APIClient()
        url = client._build_url("/path")
        assert url == "/path"

    def test_bearer_token_header(self):
        client = APIClient(api_key="mytoken", api_key_header="Authorization", api_key_prefix="Bearer")
        assert client._default_headers["Authorization"] == "Bearer mytoken"

    def test_custom_api_key_header(self):
        client = APIClient(api_key="key123", api_key_header="X-API-Key", api_key_prefix="")
        assert client._default_headers["X-API-Key"] == "key123"

    def test_callable_interface(self, httpx_mock):
        httpx_mock.add_response(json={"result": "ok"}, status_code=200)
        client = APIClient(base_url="https://api.example.com")
        result = client({"endpoint": "/data", "method": "GET"})
        assert result["ok"] is True
        assert result["body"] == {"result": "ok"}

    def test_callable_post(self, httpx_mock):
        httpx_mock.add_response(json={"id": 1}, status_code=201)
        client = APIClient(base_url="https://api.example.com")
        result = client({"endpoint": "/items", "method": "POST", "body": {"name": "test"}})
        assert result["status_code"] == 201

    def test_error_response(self, httpx_mock):
        httpx_mock.add_response(status_code=404, text="Not found")
        client = APIClient(base_url="https://api.example.com")
        result = client({"endpoint": "/missing", "method": "GET"})
        assert result["status_code"] == 404
        assert result["ok"] is False

    def test_no_base_url(self, httpx_mock):
        httpx_mock.add_response(json={"ok": True}, status_code=200)
        client = APIClient()
        response = client.get("https://api.test.com/endpoint")
        assert response.ok is True


# ── DocumentProcessor tests ───────────────────────────────────────────────────

class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_process_json(self, processor):
        content = json.dumps({"invoice_number": "INV-001", "total": 500})
        result = processor.process(content, fmt="json")
        assert result.document_type == "json"
        assert result.extracted_fields.get("invoice_number") == "INV-001"

    def test_process_json_with_schema(self, processor):
        content = json.dumps({"name": "Alice", "age": 30, "city": "NYC"})
        result = processor.process(content, schema={"name": "Person name"}, fmt="json")
        assert "name" in result.extracted_fields
        assert "age" not in result.extracted_fields  # not in schema

    def test_process_invalid_json(self, processor):
        result = processor.process("not json", fmt="json")
        assert not result.success
        assert len(result.errors) > 0

    def test_process_csv(self, processor):
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        result = processor.process(csv_content, fmt="csv")
        assert result.document_type == "csv"
        assert result.extracted_fields["row_count"] == 2

    def test_process_csv_with_schema(self, processor):
        csv_content = "name,age,city\nAlice,30,NYC"
        result = processor.process(csv_content, schema={"name": "Person name"}, fmt="csv")
        assert "name" in result.extracted_fields["columns"]

    def test_process_text(self, processor):
        text = "invoice_number: INV-2024-001\ntotal_amount: 1500.00"
        result = processor.process(text, schema={"invoice_number": "Invoice number"}, fmt="text")
        assert result.document_type == "text"
        assert result.extracted_fields.get("invoice_number") == "INV-2024-001"

    def test_auto_detect_json(self, processor):
        content = '{"key": "value"}'
        result = processor.process(content, fmt="auto")
        assert result.document_type == "json"

    def test_auto_detect_csv(self, processor):
        content = "col1,col2,col3\n1,2,3"
        result = processor.process(content, fmt="auto")
        assert result.document_type == "csv"

    def test_auto_detect_text(self, processor):
        content = "This is a plain text document without structured data."
        result = processor.process(content, fmt="auto")
        assert result.document_type == "text"

    def test_chunk_short_text(self, processor):
        text = "Short text"
        chunks = processor.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self, processor):
        text = "A" * 250  # longer than chunk_size=100
        chunks = processor.chunk(text)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_overlap(self, processor):
        text = "A" * 200
        chunks = processor.chunk(text)
        # Each chunk should start 80 chars (100-20) after the previous
        for i in range(len(chunks) - 1):
            assert chunks[i + 1][:20] == chunks[i][-20:]

    def test_callable_interface(self, processor):
        content = json.dumps({"status": "ok"})
        result = processor({"content": content, "format": "json"})
        assert isinstance(result, dict)
        assert result["document_type"] == "json"

    def test_document_result_to_dict(self, processor):
        result = processor.process('{"a": 1}', fmt="json")
        d = result.to_dict()
        assert d["document_type"] == "json"
        assert "extracted_fields" in d

    def test_nested_json_extraction(self, processor):
        content = json.dumps({"vendor": {"id": "V-001", "name": "Acme"}})
        result = processor.process(content, schema={"vendor.id": "Vendor ID"}, fmt="json")
        assert result.extracted_fields.get("vendor.id") == "V-001"


# ── DataValidator tests ───────────────────────────────────────────────────────

class TestDataValidator:
    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_valid_data_no_rules(self, validator):
        result = validator.validate({"x": 1})
        assert result.valid is True

    def test_required_rule_pass(self, validator):
        result = validator.validate({"name": "Alice"}, rules=[{"field": "name", "rule": "required"}])
        assert result.valid is True

    def test_required_rule_fail(self, validator):
        result = validator.validate({"name": None}, rules=[{"field": "name", "rule": "required"}])
        assert result.valid is False
        assert result.error_count == 1

    def test_not_empty_pass(self, validator):
        result = validator.validate({"items": [1, 2]}, rules=[{"field": "items", "rule": "not_empty"}])
        assert result.valid is True

    def test_not_empty_fail(self, validator):
        result = validator.validate({"items": []}, rules=[{"field": "items", "rule": "not_empty"}])
        assert result.valid is False

    def test_type_rule(self, validator):
        r1 = validator.validate({"val": "hello"}, rules=[{"field": "val", "rule": "type", "value": "string"}])
        assert r1.valid is True
        r2 = validator.validate({"val": 123}, rules=[{"field": "val", "rule": "type", "value": "string"}])
        assert r2.valid is False

    def test_min_value(self, validator):
        assert validator.validate({"n": 10}, rules=[{"field": "n", "rule": "min_value", "value": 5}]).valid
        assert not validator.validate({"n": 3}, rules=[{"field": "n", "rule": "min_value", "value": 5}]).valid

    def test_max_value(self, validator):
        assert validator.validate({"n": 4}, rules=[{"field": "n", "rule": "max_value", "value": 5}]).valid
        assert not validator.validate({"n": 6}, rules=[{"field": "n", "rule": "max_value", "value": 5}]).valid

    def test_gte_alias(self, validator):
        assert validator.validate({"n": 5}, rules=[{"field": "n", "rule": "gte", "value": 5}]).valid

    def test_min_length(self, validator):
        assert validator.validate({"s": "hello"}, rules=[{"field": "s", "rule": "min_length", "value": 3}]).valid
        assert not validator.validate({"s": "hi"}, rules=[{"field": "s", "rule": "min_length", "value": 3}]).valid

    def test_max_length(self, validator):
        assert validator.validate({"s": "abc"}, rules=[{"field": "s", "rule": "max_length", "value": 5}]).valid
        assert not validator.validate({"s": "toolong!"}, rules=[{"field": "s", "rule": "max_length", "value": 5}]).valid

    def test_enum_rule(self, validator):
        rules = [{"field": "status", "rule": "enum", "value": ["active", "inactive"]}]
        assert validator.validate({"status": "active"}, rules=rules).valid
        assert not validator.validate({"status": "deleted"}, rules=rules).valid

    def test_pattern_rule(self, validator):
        rules = [{"field": "date", "rule": "pattern", "value": r"\d{4}-\d{2}-\d{2}"}]
        assert validator.validate({"date": "2024-06-01"}, rules=rules).valid
        assert not validator.validate({"date": "2024/06/01"}, rules=rules).valid

    def test_unique_rule(self, validator):
        assert validator.validate({"ids": [1, 2, 3]}, rules=[{"field": "ids", "rule": "unique"}]).valid
        assert not validator.validate({"ids": [1, 2, 1]}, rules=[{"field": "ids", "rule": "unique"}]).valid

    def test_not_null_rule(self, validator):
        assert validator.validate({"x": 0}, rules=[{"field": "x", "rule": "not_null"}]).valid
        assert not validator.validate({"x": None}, rules=[{"field": "x", "rule": "not_null"}]).valid

    def test_warning_severity(self, validator):
        rules = [{"field": "optional_field", "rule": "required", "severity": "warning"}]
        result = validator.validate({}, rules=rules)
        assert result.valid is True  # warnings don't make it invalid
        assert result.warning_count == 1

    def test_json_schema_validation(self, validator):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        valid_result = validator.validate({"name": "Alice", "age": 30}, json_schema=schema)
        assert valid_result.valid is True

        invalid_result = validator.validate({"name": "Alice"}, json_schema=schema)
        assert invalid_result.valid is False

    def test_callable_interface(self, validator):
        result = validator({"data": {"x": 1}, "rules": [{"field": "x", "rule": "required"}]})
        assert isinstance(result, dict)
        assert result["valid"] is True

    def test_schema_validation_result_to_dict(self, validator):
        result = validator.validate({"a": 1}, rules=[{"field": "a", "rule": "required"}])
        d = result.to_dict()
        assert "valid" in d
        assert "violations" in d
        assert "checked_rules" in d

    def test_nested_field_rule(self, validator):
        data = {"order": {"total": 100}}
        rules = [{"field": "order.total", "rule": "min_value", "value": 0}]
        assert validator.validate(data, rules=rules).valid

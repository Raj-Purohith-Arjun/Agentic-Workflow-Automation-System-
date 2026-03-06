"""
Document processor – extracts structured data from text, JSON, and CSV
documents, supports chunking for large documents, and integrates with
the LLM layer for semantic field extraction.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_workflow.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Result of processing a document."""

    document_type: str
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "extracted_fields": self.extracted_fields,
            "chunks": self.chunks,
            "raw_text": self.raw_text[:500] + "…" if len(self.raw_text) > 500 else self.raw_text,
            "metadata": self.metadata,
            "errors": self.errors,
            "success": self.success,
        }


class DocumentProcessor:
    """
    Multi-format document processor.

    Supports:
    * Plain text (``text``)
    * JSON documents (``json``)
    * CSV files (``csv``)
    * Auto-detection (``auto``)

    Can be registered as a tool callable with the ExecutionAgent:

    .. code-block:: python

        processor = DocumentProcessor()
        engine.register_tool("document_processor", processor)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.doc_max_chunk_size
        self._chunk_overlap = chunk_overlap or settings.doc_chunk_overlap

    # ── Callable interface ────────────────────────────────────────────────────

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Callable interface for the ExecutionAgent tool registry.

        ``inputs`` keys: content (str), schema (dict), format (str).
        """
        content: str = str(inputs.get("content", ""))
        schema: dict[str, Any] = inputs.get("schema") or {}
        fmt: str = str(inputs.get("format", "auto")).lower()

        result = self.process(content=content, schema=schema, fmt=fmt)
        return result.to_dict()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        content: str,
        schema: dict[str, Any] | None = None,
        fmt: str = "auto",
    ) -> DocumentResult:
        """
        Process a document string and extract structured fields.

        Parameters
        ----------
        content:
            Raw document content (text, JSON string, CSV string, or file path).
        schema:
            Optional extraction schema: ``{field_name: description}``.
        fmt:
            ``text``, ``json``, ``csv``, or ``auto`` (default).
        """
        if fmt == "auto":
            fmt = self._detect_format(content)

        logger.debug("Processing document", extra={"format": fmt, "length": len(content)})

        if fmt == "json":
            return self._process_json(content, schema or {})
        if fmt == "csv":
            return self._process_csv(content, schema or {})
        return self._process_text(content, schema or {})

    def chunk(self, text: str) -> list[str]:
        """
        Split ``text`` into overlapping chunks for LLM context windows.

        Uses a sliding window of ``chunk_size`` characters with
        ``chunk_overlap`` characters of overlap between adjacent chunks.
        """
        if len(text) <= self._chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += self._chunk_size - self._chunk_overlap
        return chunks

    # ── Format-specific processors ────────────────────────────────────────────

    def _process_text(self, content: str, schema: dict[str, Any]) -> DocumentResult:
        """Extract fields from plain text using pattern matching and heuristics."""
        chunks = self.chunk(content)
        extracted: dict[str, Any] = {}

        for field_name, description in schema.items():
            value = self._extract_field_from_text(content, field_name, str(description))
            if value is not None:
                extracted[field_name] = value

        return DocumentResult(
            document_type="text",
            extracted_fields=extracted,
            chunks=chunks,
            raw_text=content,
            metadata={"char_count": len(content), "chunk_count": len(chunks)},
        )

    def _process_json(self, content: str, schema: dict[str, Any]) -> DocumentResult:
        """Parse JSON and extract fields specified in the schema."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return DocumentResult(
                document_type="json",
                errors=[f"Invalid JSON: {exc}"],
            )

        extracted: dict[str, Any] = {}
        if schema:
            for field_name in schema:
                value = self._get_nested(data, field_name)
                if value is not None:
                    extracted[field_name] = value
        else:
            # No schema – return the whole document flattened to one level
            if isinstance(data, dict):
                extracted = data
            else:
                extracted = {"data": data}

        return DocumentResult(
            document_type="json",
            extracted_fields=extracted,
            raw_text=content,
            metadata={"top_level_keys": list(data.keys()) if isinstance(data, dict) else []},
        )

    def _process_csv(self, content: str, schema: dict[str, Any]) -> DocumentResult:
        """Parse CSV and return rows as extracted_fields with summary statistics."""
        try:
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
        except Exception as exc:  # noqa: BLE001
            return DocumentResult(
                document_type="csv",
                errors=[f"CSV parse error: {exc}"],
            )

        columns = list(rows[0].keys()) if rows else []

        # Filter columns if schema provided
        if schema:
            columns_to_keep = [c for c in columns if c in schema]
            filtered_rows = [{k: row[k] for k in columns_to_keep} for row in rows]
        else:
            filtered_rows = rows

        return DocumentResult(
            document_type="csv",
            extracted_fields={
                "rows": filtered_rows,
                "row_count": len(rows),
                "columns": columns,
            },
            raw_text=content,
            metadata={"row_count": len(rows), "column_count": len(columns)},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_format(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            return "json"
        # Simple CSV heuristic: first line has commas and no JSON/HTML chars
        first_line = stripped.splitlines()[0] if stripped else ""
        if "," in first_line and "<" not in first_line and "{" not in first_line:
            return "csv"
        return "text"

    @staticmethod
    def _extract_field_from_text(text: str, field_name: str, description: str) -> Any:
        """
        Simple heuristic extractor: looks for ``field_name: value`` patterns.
        Returns the value string or None.
        """
        pattern = re.compile(
            rf"(?i){re.escape(field_name)}\s*[:\-=]\s*([^\n,;]+)",
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _get_nested(data: Any, dotted_path: str) -> Any:
        """Traverse a dot-separated key path in nested dicts/lists."""
        parts = dotted_path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return None
        return current

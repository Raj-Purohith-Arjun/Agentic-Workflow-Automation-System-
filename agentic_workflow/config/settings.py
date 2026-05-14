"""
Settings / configuration management.

Reads from environment variables (and an optional .env file).
All settings have sensible defaults so the system works out-of-the-box
without a live OpenAI key (the LLM layer gracefully falls back to a mock
when the key is absent).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env if it exists (silently ignored when absent)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=False)


class Settings(BaseSettings):
    """Central configuration object; values come from env vars or .env."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, alias="OPENAI_TEMPERATURE")

    # ── Workflow Engine ───────────────────────────────────────────────────────
    workflow_max_steps: int = Field(default=20, alias="WORKFLOW_MAX_STEPS")
    workflow_timeout_seconds: int = Field(
        default=300, alias="WORKFLOW_TIMEOUT_SECONDS"
    )
    workflow_state_backend: Literal["memory", "file"] = Field(
        default="memory", alias="WORKFLOW_STATE_BACKEND"
    )
    workflow_state_dir: str = Field(
        default=".workflow_states", alias="WORKFLOW_STATE_DIR"
    )

    # ── External API ─────────────────────────────────────────────────────────
    api_request_timeout: int = Field(default=30, alias="API_REQUEST_TIMEOUT")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    api_retry_backoff: float = Field(default=2.0, alias="API_RETRY_BACKOFF")

    # ── Validation ───────────────────────────────────────────────────────────
    validation_strict_mode: bool = Field(default=True, alias="VALIDATION_STRICT_MODE")
    validation_max_errors: int = Field(default=10, alias="VALIDATION_MAX_ERRORS")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(
        default="console", alias="LOG_FORMAT"
    )

    # ── Document Processing ──────────────────────────────────────────────────
    doc_max_chunk_size: int = Field(default=2000, alias="DOC_MAX_CHUNK_SIZE")
    doc_chunk_overlap: int = Field(default=200, alias="DOC_CHUNK_OVERLAP")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("openai_temperature")
    @classmethod
    def _clamp_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return v

    @property
    def has_llm_key(self) -> bool:
        """Return True when a non-empty OpenAI API key is configured."""
        return bool(self.openai_api_key)


# Module-level singleton – import and use directly where needed.
settings = Settings()

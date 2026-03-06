"""
Tests for settings/configuration management.
"""

from __future__ import annotations

import os

import pytest

from agentic_workflow.config.settings import Settings


class TestSettings:
    def test_default_model(self):
        s = Settings()
        assert s.openai_model == "gpt-4-turbo-preview"

    def test_has_llm_key_false_when_empty(self):
        s = Settings(OPENAI_API_KEY="")
        assert s.has_llm_key is False

    def test_has_llm_key_true_when_set(self):
        s = Settings(OPENAI_API_KEY="sk-test-key")
        assert s.has_llm_key is True

    def test_temperature_clamp_valid(self):
        s = Settings(OPENAI_TEMPERATURE=0.5)
        assert s.openai_temperature == 0.5

    def test_temperature_clamp_invalid(self):
        with pytest.raises(Exception):
            Settings(OPENAI_TEMPERATURE=3.0)

    def test_workflow_max_steps_default(self):
        s = Settings()
        assert s.workflow_max_steps == 20

    def test_state_backend_default(self):
        s = Settings()
        assert s.workflow_state_backend == "memory"

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
        s = Settings()
        assert s.openai_model == "gpt-3.5-turbo"

    def test_validation_strict_mode_default(self):
        s = Settings()
        assert s.validation_strict_mode is True

    def test_doc_chunk_size_default(self):
        s = Settings()
        assert s.doc_max_chunk_size == 2000

"""
Base LLM agent with retry logic, structured output parsing, and mock support.

When no OpenAI API key is configured the agent automatically falls back to a
deterministic mock so that the rest of the system (workflows, tools, tests)
works without any network connectivity.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_workflow.config.settings import settings

logger = logging.getLogger(__name__)


class LLMResponse:
    """Wrapper around a raw LLM completion."""

    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
    ) -> None:
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms

    def parse_json(self) -> dict[str, Any]:
        """Parse content as JSON, stripping markdown fences if present."""
        text = self.content.strip()
        # Strip ```json ... ``` or ``` ... ``` fences
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first line (```json or ```) and last (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON: {exc}\nContent:\n{self.content[:500]}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.prompt_tokens}+{self.completion_tokens}, "
            f"latency={self.latency_ms}ms)"
        )


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-backed agents.

    Sub-classes implement ``_build_messages`` (to construct the prompt) and
    ``_parse_response`` (to turn the raw ``LLMResponse`` into a typed result).
    """

    def __init__(
        self,
        system_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature
        self.max_tokens = max_tokens or settings.openai_max_tokens
        self._client: Any = None  # lazy-initialised

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, user_prompt: str, **kwargs: Any) -> Any:
        """Execute the agent with the given user prompt. Returns a typed result."""
        start = time.monotonic()
        llm_response = self._call_llm(user_prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        llm_response.latency_ms = elapsed_ms
        logger.debug(
            "LLM call complete",
            extra={"model": self.model, "latency_ms": elapsed_ms, **kwargs},
        )
        return self._parse_response(llm_response, **kwargs)

    # ── LLM interaction ───────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> LLMResponse:
        """Route to real or mock LLM depending on configuration."""
        if not settings.has_llm_key:
            logger.debug("No OpenAI API key – using mock LLM response")
            return self._mock_llm_response(user_prompt)
        return self._call_openai(user_prompt)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_openai(self, user_prompt: str) -> LLMResponse:
        """Call the OpenAI Chat Completions API with retry logic."""
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required. Install it with: pip install openai"
                ) from exc
            self._client = OpenAI(api_key=settings.openai_api_key)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def _mock_llm_response(self, user_prompt: str) -> LLMResponse:
        """Return a deterministic mock response for testing/demo without an API key."""
        mock_json = self._build_mock_response(user_prompt)
        return LLMResponse(
            content=json.dumps(mock_json),
            model="mock-model",
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=100,
        )

    # ── Abstract methods ─────────────────────────────────────────────────────

    @abstractmethod
    def _parse_response(self, response: LLMResponse, **kwargs: Any) -> Any:
        """Convert a raw LLM response into the agent's typed output."""

    @abstractmethod
    def _build_mock_response(self, user_prompt: str) -> dict[str, Any]:
        """Return a deterministic mock JSON response (used when no API key)."""

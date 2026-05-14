"""
External API client with authentication, rate-limiting, and retry logic.

The client is callable so it can be registered directly with the
``ExecutionAgent`` tool registry:

    engine.register_tool("api_client", APIClient(base_url="https://..."))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_workflow.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Normalised response from an API call."""

    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: int = 0
    url: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "body": self.body,
            "headers": self.headers,
            "latency_ms": self.latency_ms,
            "url": self.url,
            "error": self.error,
            "ok": self.ok,
        }


class APIClient:
    """
    HTTP client wrapping ``httpx`` with:
    * Configurable base URL and default headers
    * Bearer-token / API-key authentication
    * Automatic retry with exponential back-off on transient errors
    * Rate-limit awareness (respects Retry-After header on 429)
    * Structured logging

    Can be used as a plain client *or* as a tool callable:

    .. code-block:: python

        client = APIClient(base_url="https://api.example.com")
        # As a callable tool (dict input from the execution agent):
        result = client({"endpoint": "/users/1", "method": "GET"})
        # Direct call:
        response = client.get("/users/1")
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer",
        default_headers: dict[str, str] | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._api_key_prefix = api_key_prefix
        self._timeout = timeout or settings.api_request_timeout
        self._max_retries = max_retries or settings.api_max_retries
        self._default_headers: dict[str, str] = default_headers or {}

        if api_key:
            self._default_headers[api_key_header] = (
                f"{api_key_prefix} {api_key}" if api_key_prefix else api_key
            )

    # ── Callable interface (for ExecutionAgent tool registry) ─────────────────

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Callable interface used by the ExecutionAgent.

        ``inputs`` keys: endpoint, method, body, headers, params.
        """
        endpoint: str = inputs.get("endpoint", "/")
        method: str = str(inputs.get("method", "GET")).upper()
        body: Any = inputs.get("body")
        headers: dict[str, str] = inputs.get("headers") or {}
        params: dict[str, Any] = inputs.get("params") or {}

        response = self.request(
            method=method,
            endpoint=endpoint,
            json_body=body if isinstance(body, dict) else None,
            data=body if not isinstance(body, dict) else None,
            headers=headers,
            params=params,
        )
        return response.to_dict()

    # ── High-level helpers ────────────────────────────────────────────────────

    def get(self, endpoint: str, params: dict[str, Any] | None = None, **kwargs: Any) -> APIResponse:
        return self.request("GET", endpoint, params=params, **kwargs)

    def post(self, endpoint: str, body: Any = None, **kwargs: Any) -> APIResponse:
        return self.request(
            "POST",
            endpoint,
            json_body=body if isinstance(body, dict) else None,
            **kwargs,
        )

    def put(self, endpoint: str, body: Any = None, **kwargs: Any) -> APIResponse:
        return self.request(
            "PUT",
            endpoint,
            json_body=body if isinstance(body, dict) else None,
            **kwargs,
        )

    def patch(self, endpoint: str, body: Any = None, **kwargs: Any) -> APIResponse:
        return self.request(
            "PATCH",
            endpoint,
            json_body=body if isinstance(body, dict) else None,
            **kwargs,
        )

    def delete(self, endpoint: str, **kwargs: Any) -> APIResponse:
        return self.request("DELETE", endpoint, **kwargs)

    # ── Core request method ───────────────────────────────────────────────────

    def request(
        self,
        method: str,
        endpoint: str,
        json_body: dict[str, Any] | None = None,
        data: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APIResponse:
        """Issue an HTTP request with retry and rate-limit handling."""
        url = self._build_url(endpoint)
        merged_headers = {**self._default_headers, **(headers or {})}
        return self._request_with_retry(
            method=method,
            url=url,
            json_body=json_body,
            data=data,
            params=params,
            headers=merged_headers,
        )

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_body: dict[str, Any] | None,
        data: Any,
        params: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> APIResponse:
        start = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.request(
                    method=method,
                    url=url,
                    json=json_body,
                    content=data if not isinstance(data, dict) else None,
                    params=params,
                    headers=headers,
                )
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("API request failed", extra={"url": url, "error": str(exc)})
            return APIResponse(
                status_code=0,
                body=None,
                latency_ms=latency_ms,
                url=url,
                error=str(exc),
            )

        latency_ms = int((time.monotonic() - start) * 1000)

        # Handle rate limiting
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            logger.warning(
                "Rate limited; sleeping",
                extra={"url": url, "retry_after": retry_after},
            )
            time.sleep(retry_after)

        # Parse body
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = resp.text

        logger.debug(
            "API response",
            extra={
                "method": method,
                "url": url,
                "status": resp.status_code,
                "latency_ms": latency_ms,
            },
        )
        return APIResponse(
            status_code=resp.status_code,
            body=body,
            headers=dict(resp.headers),
            latency_ms=latency_ms,
            url=url,
        )

    def _build_url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        if self.base_url:
            return urljoin(self.base_url + "/", endpoint.lstrip("/"))
        return endpoint

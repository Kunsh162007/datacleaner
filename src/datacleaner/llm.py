"""LLM access: a small protocol plus a Gemini implementation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"  # pinned (not an alias) for reproducible plans
API_KEY_ENV = "GEMINI_API_KEY"
ERROR_DETAIL_MAX = 300
API_VERSION = "v1beta"
DOTENV_FILE = ".env"
REQUEST_TIMEOUT_MS = 90_000  # never hang suggest on a stuck request


class LLMError(Exception):
    """The LLM call failed or returned nothing usable."""


class TransientLLMError(LLMError):
    """The provider is unavailable (rate limit, 5xx, timeout, model gone).

    Only this error lets a fallback client move on to another provider.
    `reason` is a short label such as "503" or "timeout".
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


# 429 rate limit, 5xx availability, 404/410 model unavailable. 400/401/403 are
# configuration errors and are deliberately absent.
TRANSIENT_STATUS = frozenset({404, 410, 429, 500, 502, 503, 504})
# Matched by class name anywhere in the MRO so httpx/openai need not be imported.
_TIMEOUT_NAMES = frozenset({"TimeoutError", "TimeoutException", "APITimeoutError"})
_CONNECTION_NAMES = frozenset(
    {"ConnectionError", "TransportError", "APIConnectionError", "ConnectError"}
)


def transient_reason(exc: BaseException) -> str | None:
    """A short label if `exc` is a transient availability failure, else None."""
    for code in (getattr(exc, "code", None), getattr(exc, "status_code", None)):
        if isinstance(code, int) and code in TRANSIENT_STATUS:
            return str(code)
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _TIMEOUT_NAMES:
        return "timeout"
    if names & _CONNECTION_NAMES:
        return "connection error"
    return None


def classify_failure(exc: BaseException, message: str) -> LLMError:
    reason = transient_reason(exc)
    if reason is None and exc.__cause__ is not None:
        reason = transient_reason(exc.__cause__)
    return TransientLLMError(message, reason) if reason else LLMError(message)


class LLMClient(Protocol):
    model: str

    def complete_json(
        self, system: str, user: str, response_schema: Mapping[str, Any]
    ) -> str:
        """Return the model's raw text answer (expected to be JSON)."""
        ...


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self.model = model

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r})"

    def complete_json(
        self, system: str, user: str, response_schema: Mapping[str, Any]
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=dict(response_schema),
            temperature=0,
        )
        try:
            client = genai.Client(
                api_key=self._api_key,
                # system_instruction + JSON schema are only accepted on v1beta
                http_options=types.HttpOptions(
                    api_version=API_VERSION, timeout=REQUEST_TIMEOUT_MS
                ),
            )
            response = client.models.generate_content(
                model=self.model, contents=user, config=config
            )
            text = response.text
        except Exception as exc:  # SDK raises many types
            raise classify_failure(
                exc, f"Gemini request failed: {self._describe(exc)}"
            ) from None
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response")
        return text

    def _describe(self, exc: Exception) -> str:
        """Readable error detail with the API key masked out."""
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        message = getattr(exc, "message", None) or str(exc)
        prefix = " ".join(str(p) for p in (code, status) if p) or type(exc).__name__
        masked = message.replace(self._api_key, "***") if self._api_key else message
        return f"{prefix}: {masked[:ERROR_DETAIL_MAX]}"


def resolve_api_key(env_name: str = API_KEY_ENV, dotenv_path: str | Path = DOTENV_FILE) -> str:
    """Project ``.env`` wins over the shell, so a machine-wide key for another
    project is never picked up by accident. CI has no .env and uses the env var."""
    from_file = dotenv_values(dotenv_path).get(env_name) or ""
    return from_file.strip() or os.environ.get(env_name, "").strip()


def require_api_key(env_name: str) -> str:
    key = resolve_api_key(env_name)
    if not key:
        raise LLMError(
            f"{env_name} is not set; put it in a .env file or export it to use suggest"
        )
    return key


def client_from_env(model: str) -> GeminiClient:
    return GeminiClient(require_api_key(API_KEY_ENV), model)

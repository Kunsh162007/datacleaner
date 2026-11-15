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
            raise LLMError(f"Gemini request failed: {self._describe(exc)}") from None
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


def resolve_api_key(dotenv_path: str | Path = DOTENV_FILE) -> str:
    """Project ``.env`` wins over the shell, so a machine-wide key for another
    project is never picked up by accident. CI has no .env and uses the env var."""
    from_file = dotenv_values(dotenv_path).get(API_KEY_ENV) or ""
    return from_file.strip() or os.environ.get(API_KEY_ENV, "").strip()


def client_from_env(model: str) -> GeminiClient:
    key = resolve_api_key()
    if not key:
        raise LLMError(
            f"{API_KEY_ENV} is not set; put it in a .env file or export it to use suggest"
        )
    return GeminiClient(key, model)

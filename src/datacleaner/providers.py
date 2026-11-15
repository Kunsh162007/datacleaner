"""OpenAI-compatible LLM clients (Groq, NVIDIA) and the provider registry."""

from __future__ import annotations

import re
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from datacleaner.llm import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    ERROR_DETAIL_MAX,
    REQUEST_TIMEOUT_MS,
    GeminiClient,
    LLMClient,
    LLMError,
    TransientLLMError,
    classify_failure,
    require_api_key,
)

_MS_PER_SECOND = 1000
DEFAULT_PROVIDER = "nvidia"  # one provider by default; auto (multi-company) is opt-in
_ACCOUNT_ID = re.compile(r"\borg_[A-Za-z0-9]+")
AUTO_PROVIDER = "auto"
AUTO_ORDER = ("nvidia", "gemini", "groq")  # by benchmark: best-scoring first
SCHEMA_HINT = (
    "Return only a single JSON object (no prose, no code fences) that matches this "
    "JSON schema: "
)


@dataclass(frozen=True)
class ProviderSpec:
    env: str
    default_model: str
    base_url: str | None = None


PROVIDERS: Mapping[str, ProviderSpec] = {
    "gemini": ProviderSpec(API_KEY_ENV, DEFAULT_MODEL),
    "groq": ProviderSpec(
        "GROQ_API_KEY", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1"
    ),
    "nvidia": ProviderSpec(
        "NVIDIA_API_KEY", "minimaxai/minimax-m3.1-flash", "https://integrate.api.nvidia.com/v1"
    ),
}


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    `model` (used in the plan's generated_by) is "<provider>/<model>" when a
    provider label is given; the raw model id is what is sent to the API.
    """

    def __init__(
        self, api_key: str, model: str, base_url: str, provider: str | None = None
    ) -> None:
        self._api_key = api_key
        self._api_model = model
        self._base_url = base_url
        self.model = f"{provider}/{model}" if provider else model

    def __repr__(self) -> str:
        return f"OpenAICompatClient(model={self.model!r}, base_url={self._base_url!r})"

    def complete_json(
        self, system: str, user: str, response_schema: Mapping[str, Any]
    ) -> str:
        schema_json = json.dumps(response_schema, separators=(",", ":"))
        messages = [
            {"role": "system", "content": f"{system}\n\n{SCHEMA_HINT}{schema_json}"},
            {"role": "user", "content": user},
        ]
        try:
            client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT_MS / _MS_PER_SECOND,
                max_retries=0,  # retry policy belongs to the caller; never loop silently
            )
            response = client.chat.completions.create(
                model=self._api_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = _first_text(response)
        except Exception as exc:  # SDK raises many types
            raise classify_failure(
                exc, f"{self.model} request failed: {self._describe(exc)}"
            ) from None
        if not text or not text.strip():
            raise LLMError(f"{self.model} returned an empty response")
        return text

    def _describe(self, exc: Exception) -> str:
        """Readable error detail (with HTTP status) and the API key masked out."""
        code = getattr(exc, "status_code", None)
        message = getattr(exc, "message", None) or str(exc)
        prefix = str(code) if code else type(exc).__name__
        masked = message.replace(self._api_key, "***") if self._api_key else message
        masked = _ACCOUNT_ID.sub("org_***", masked)  # provider account ids are private
        return f"{prefix}: {masked[:ERROR_DETAIL_MAX]}"


def _first_text(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(choices[0].message, "content", None)


class FallbackClient:
    """Tries providers in order, moving on only after a TransientLLMError.

    `model` always names the provider/model that answered last (or the first
    candidate before any call), so plans record who really produced them. A
    provider that failed transiently is not retried first on the next call.
    """

    def __init__(
        self,
        clients: Sequence[tuple[str, LLMClient]],
        notify: Callable[[str], None] = lambda _message: None,
    ) -> None:
        if not clients:
            raise LLMError("no providers available for fallback")
        self._clients = list(clients)
        self._notify = notify
        self._start = 0
        self.model = self._clients[0][1].model

    def __repr__(self) -> str:
        return f"FallbackClient(providers={[name for name, _ in self._clients]!r})"

    def complete_json(
        self, system: str, user: str, response_schema: Mapping[str, Any]
    ) -> str:
        failures: list[str] = []
        order = range(self._start, len(self._clients))
        for index in order:
            name, client = self._clients[index]
            try:
                text = client.complete_json(system, user, response_schema)
            except TransientLLMError as exc:
                failures.append(f"{name}: {exc}")
                following = self._clients[index + 1 :]
                if following:
                    self._notify(
                        f"{name} unavailable ({exc.reason}), trying {following[0][0]}"
                    )
                continue
            self._start, self.model = index, client.model
            return text
        detail = "".join(f"\n  {line}" for line in failures)
        raise LLMError(f"all providers failed:{detail}")


def _stderr_note(message: str) -> None:
    print(f"datacleaner: {message}", file=sys.stderr)


def _auto_client(model: str | None, notify: Callable[[str], None]) -> FallbackClient:
    if model:
        raise LLMError("--model cannot be combined with --provider auto (each provider "
                       "uses its own default model)")
    available: list[tuple[str, LLMClient]] = []
    for name in AUTO_ORDER:
        try:
            available.append((name, client_for(name)))
        except LLMError:
            continue  # key missing: skip this provider
    if not available:
        keys = ", ".join(PROVIDERS[name].env for name in AUTO_ORDER)
        raise LLMError(f"no API key found for auto fallback; set at least one of: {keys}")
    return FallbackClient(available, notify)


def client_for(
    provider: str, model: str | None = None, *, notify: Callable[[str], None] = _stderr_note
) -> LLMClient:
    """Build a client for a registered provider, reading its key from env/.env.

    "auto" builds a FallbackClient over every provider that has a key.
    """
    if provider == AUTO_PROVIDER:
        return _auto_client(model, notify)
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMError(f"unknown provider '{provider}'; choose from {', '.join(PROVIDERS)}")
    chosen = model or spec.default_model
    key = require_api_key(spec.env)
    if spec.base_url is None:
        return GeminiClient(key, chosen)
    return OpenAICompatClient(key, chosen, spec.base_url, provider=provider)

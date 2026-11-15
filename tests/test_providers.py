"""Tests for the OpenAI-compatible client and provider registry (no network)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeClient

from datacleaner import providers
from datacleaner.cli import main
from datacleaner.llm import GeminiClient, LLMError
from datacleaner.providers import PROVIDERS, OpenAICompatClient, client_for

SECRET = "gsk-super-secret"
SCHEMA = {"type": "object", "properties": {"actions": {"type": "array"}}}
MESSY = Path(__file__).parent.parent / "examples" / "messy.csv"


class ApiError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _stub(monkeypatch, *, response=None, error=None):
    seen: dict = {}

    class Completions:
        def create(self, **kwargs):
            seen["create"] = kwargs
            if error:
                raise error
            return response if response is not None else _response("{}")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)
    return seen


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # never read the developer's real .env
    for spec in PROVIDERS.values():
        monkeypatch.delenv(spec.env, raising=False)


def test_request_shape(monkeypatch) -> None:
    seen = _stub(monkeypatch, response=_response('{"actions": []}'))
    client = OpenAICompatClient(SECRET, "llama-x", "https://example.test/v1")
    assert client.complete_json("SYS", "USR", SCHEMA) == '{"actions": []}'
    call = seen["create"]
    assert call["model"] == "llama-x" and call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    system, user = call["messages"]
    assert system["role"] == "system" and system["content"].startswith("SYS")
    assert json.dumps(SCHEMA, separators=(",", ":")) in system["content"]
    assert user == {"role": "user", "content": "USR"}
    assert seen["init"]["api_key"] == SECRET
    assert seen["init"]["base_url"] == "https://example.test/v1"
    assert seen["init"]["timeout"] == providers.REQUEST_TIMEOUT_MS / 1000
    assert seen["init"]["max_retries"] == 0


def test_errors_are_masked_and_carry_status(monkeypatch) -> None:
    _stub(monkeypatch, error=ApiError(f"bad key {SECRET}", 429))
    client = OpenAICompatClient(SECRET, "m", "https://x", provider="groq")
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u", SCHEMA)
    text = str(info.value)
    assert SECRET not in text and "429" in text and "groq/m" in text


def test_generic_exception_uses_type_name(monkeypatch) -> None:
    _stub(monkeypatch, error=RuntimeError(f"boom {SECRET}"))
    with pytest.raises(LLMError, match="RuntimeError") as info:
        OpenAICompatClient(SECRET, "m", "https://x").complete_json("s", "u", SCHEMA)
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(
    "response", [_response(None), _response("  "), SimpleNamespace(choices=[])]
)
def test_empty_response_raises(monkeypatch, response) -> None:
    _stub(monkeypatch, response=response)
    with pytest.raises(LLMError, match="empty"):
        OpenAICompatClient(SECRET, "m", "https://x").complete_json("s", "u", SCHEMA)


def test_repr_hides_key() -> None:
    assert SECRET not in repr(OpenAICompatClient(SECRET, "m", "https://x"))


@pytest.mark.parametrize("name", ["groq", "nvidia"])
def test_client_for_uses_provider_env_and_defaults(monkeypatch, name) -> None:
    spec = PROVIDERS[name]
    monkeypatch.setenv(spec.env, SECRET)
    client = client_for(name)
    assert isinstance(client, OpenAICompatClient)
    assert client.model == f"{name}/{spec.default_model}"
    assert client_for(name, "custom").model == f"{name}/custom"


def test_client_for_gemini(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    client = client_for("gemini")
    assert isinstance(client, GeminiClient) and client.model == "gemini-2.5-flash"
    assert client_for("gemini", "other").model == "other"


def test_registry_matches_decided_defaults() -> None:
    assert PROVIDERS["groq"].base_url == "https://api.groq.com/openai/v1"
    assert PROVIDERS["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
    assert PROVIDERS["groq"].default_model == "openai/gpt-oss-120b"
    assert PROVIDERS["nvidia"].default_model == "minimaxai/minimax-m3.1-flash"


def test_missing_key_names_the_right_variable() -> None:
    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        client_for("groq")
    with pytest.raises(LLMError, match="NVIDIA_API_KEY is not set"):
        client_for("nvidia")


def test_provider_key_env_does_not_leak_between_providers(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        client_for("groq")


def test_dotenv_key_wins_per_provider(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    assert client_for("groq")._api_key == "from-file"  # noqa: SLF001


def test_unknown_provider() -> None:
    with pytest.raises(LLMError, match="unknown provider"):
        client_for("nope")


# ------------------------------------------------------------------ CLI wiring
def _suggest_cli(tmp_path, *extra):
    seen: list = []

    def factory(provider, model):
        seen.append((provider, model))
        return FakeClient({"actions": []})

    shutil.copy(MESSY, tmp_path / "m.csv")
    code = main(["suggest", "--in", str(tmp_path / "m.csv"), "--out", str(tmp_path / "p.json"),
                 *extra], client_factory=factory)
    return code, seen


def test_cli_defaults_to_auto_fallback(tmp_path) -> None:
    code, seen = _suggest_cli(tmp_path)
    assert code == 0 and seen == [("auto", None)]


def test_cli_passes_provider_and_model(tmp_path) -> None:
    code, seen = _suggest_cli(tmp_path, "--provider", "groq", "--model", "m1")
    assert code == 0 and seen == [("groq", "m1")]


def test_cli_rejects_unknown_provider(tmp_path) -> None:
    code, seen = _suggest_cli(tmp_path, "--provider", "openai")
    assert code != 0 and seen == []


def test_provider_label_in_plan_is_schema_valid(monkeypatch, tmp_path) -> None:
    from datacleaner.suggest import suggest_plan

    shutil.copy(MESSY, tmp_path / "m.csv")
    seen = _stub(monkeypatch, response=_response('{"actions": []}'))
    client = OpenAICompatClient(SECRET, "meta/llama-3.3-70b-instruct", "https://x", "nvidia")
    result = suggest_plan(tmp_path / "m.csv", client)
    assert result.plan["generated_by"]["model"] == "nvidia/meta/llama-3.3-70b-instruct"
    assert seen["create"]["model"] == "meta/llama-3.3-70b-instruct"


# ------------------------------------------------------- transient (OpenAI-compat)
# stand-ins named like httpx's classes (httpx is not a direct dependency)
class TimeoutException(Exception):
    pass


class ReadTimeout(TimeoutException):
    pass


class TransportError(Exception):
    pass


class ConnectError(TransportError):
    pass


class httpx:  # noqa: N801
    ReadTimeout = ReadTimeout
    ConnectError = ConnectError

from datacleaner.llm import TransientLLMError  # noqa: E402


@pytest.mark.parametrize("code", [404, 410, 429, 500, 502, 503, 504])
def test_openai_compat_transient_status(monkeypatch, code) -> None:
    _stub(monkeypatch, error=ApiError("x", code))
    with pytest.raises(TransientLLMError) as info:
        OpenAICompatClient(SECRET, "m", "https://x", "groq").complete_json("s", "u", SCHEMA)
    assert info.value.reason == str(code)


@pytest.mark.parametrize("code", [400, 401, 403])
def test_openai_compat_config_errors_not_transient(monkeypatch, code) -> None:
    _stub(monkeypatch, error=ApiError("x", code))
    with pytest.raises(LLMError) as info:
        OpenAICompatClient(SECRET, "m", "https://x").complete_json("s", "u", SCHEMA)
    assert not isinstance(info.value, TransientLLMError)


def test_openai_compat_timeout_is_transient(monkeypatch) -> None:
    class APITimeoutError(Exception):
        pass

    _stub(monkeypatch, error=APITimeoutError("Request timed out."))
    with pytest.raises(TransientLLMError) as info:
        OpenAICompatClient(SECRET, "m", "https://x").complete_json("s", "u", SCHEMA)
    assert info.value.reason == "timeout"
    _stub(monkeypatch, error=httpx.ConnectError("down"))
    with pytest.raises(TransientLLMError):
        OpenAICompatClient(SECRET, "m", "https://x").complete_json("s", "u", SCHEMA)


# ------------------------------------------------------------- FallbackClient
from datacleaner.providers import AUTO_ORDER, FallbackClient  # noqa: E402


class _Stub:
    def __init__(self, model, *outcomes):
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete_json(self, system, user, response_schema):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fb(*stubs):
    notes: list[str] = []
    return FallbackClient([(s.model.split("/")[0], s) for s in stubs], notes.append), notes


def test_fallback_order_and_model_attribution() -> None:
    a = _Stub("gemini/g", TransientLLMError("503 UNAVAILABLE", "503"))
    b = _Stub("nvidia/minimaxai/minimax-m3.1-flash", "{}")
    c = _Stub("groq/x")
    client, notes = _fb(a, b, c)
    assert client.model == "gemini/g"
    assert client.complete_json("s", "u", SCHEMA) == "{}"
    assert client.model == "nvidia/minimaxai/minimax-m3.1-flash"
    assert c.calls == 0
    assert notes == ["gemini unavailable (503), trying nvidia"]


@pytest.mark.parametrize("reason", ["429", "503", "timeout"])
def test_fallback_on_each_transient_reason(reason) -> None:
    a = _Stub("gemini/g", TransientLLMError("down", reason))
    b = _Stub("groq/m", "ok")
    client, notes = _fb(a, b)
    assert client.complete_json("s", "u", SCHEMA) == "ok"
    assert notes == [f"gemini unavailable ({reason}), trying groq"]


def test_no_fallback_on_config_error() -> None:
    a = _Stub("gemini/g", LLMError("Gemini request failed: 400 bad"))
    b = _Stub("groq/m", "ok")
    client, notes = _fb(a, b)
    with pytest.raises(LLMError, match="400"):
        client.complete_json("s", "u", SCHEMA)
    assert b.calls == 0 and notes == []


def test_no_fallback_on_validation_failure_retry() -> None:
    from datacleaner.suggest import SuggestError, suggest_plan

    a = _Stub("gemini/g", "not json", "still not json")
    b = _Stub("groq/m", "{}")
    client, _ = _fb(a, b)
    with pytest.raises(SuggestError):
        suggest_plan(MESSY, client)
    assert a.calls == 2 and b.calls == 0


def test_all_fail_lists_each_reason() -> None:
    a = _Stub("gemini/g", TransientLLMError("gemini 503 busy", "503"))
    b = _Stub("groq/m", TransientLLMError("groq 429 limit", "429"))
    client, notes = _fb(a, b)
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u", SCHEMA)
    text = str(info.value)
    assert "gemini: gemini 503 busy" in text and "groq: groq 429 limit" in text
    assert notes == ["gemini unavailable (503), trying groq"]


def test_fallback_is_sticky_after_success() -> None:
    a = _Stub("gemini/g", TransientLLMError("down", "503"))
    b = _Stub("groq/m", "1", "2")
    client, _ = _fb(a, b)
    client.complete_json("s", "u", SCHEMA)
    client.complete_json("s", "u", SCHEMA)
    assert a.calls == 1 and client.model == "groq/m"


def test_fallback_requires_clients() -> None:
    with pytest.raises(LLMError):
        FallbackClient([])


def test_fallback_repr() -> None:
    client, _ = _fb(_Stub("gemini/g"))
    assert "gemini" in repr(client)


def test_auto_skips_missing_keys_and_keeps_order(monkeypatch) -> None:
    monkeypatch.setenv(PROVIDERS["groq"].env, SECRET)
    monkeypatch.setenv(PROVIDERS["gemini"].env, SECRET)
    client = client_for("auto")
    assert isinstance(client, FallbackClient)
    assert [n for n, _ in client._clients] == ["gemini", "groq"]
    assert AUTO_ORDER == ("nvidia", "gemini", "groq")
    assert client.model == "gemini-2.5-flash"


def test_auto_uses_provider_defaults(monkeypatch) -> None:
    for spec in PROVIDERS.values():
        monkeypatch.setenv(spec.env, SECRET)
    models = [c.model for _, c in client_for("auto")._clients]
    assert models == ["nvidia/minimaxai/minimax-m3.1-flash", "gemini-2.5-flash",
                      "groq/openai/gpt-oss-120b"]


def test_auto_without_keys_or_with_model_errors() -> None:
    with pytest.raises(LLMError, match="no API key"):
        client_for("auto")
    with pytest.raises(LLMError, match="--model"):
        client_for("auto", "m")


def test_auto_default_notify_prints_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setenv(PROVIDERS["gemini"].env, SECRET)
    monkeypatch.setenv(PROVIDERS["groq"].env, SECRET)
    client = client_for("auto")
    client._clients[0] = ("gemini", _Stub("gemini-x", TransientLLMError("d", "503")))
    client._clients[1] = ("groq", _Stub("groq/m", "{}"))
    client.complete_json("s", "u", SCHEMA)
    assert "gemini unavailable (503), trying groq" in capsys.readouterr().err


def test_cli_accepts_auto(tmp_path) -> None:
    code, seen = _suggest_cli(tmp_path, "--provider", "auto")
    assert code == 0 and seen == [("auto", None)]


def test_error_detail_masks_provider_account_id() -> None:
    from datacleaner.providers import OpenAICompatClient

    client = OpenAICompatClient(SECRET, "m", "https://example.invalid/v1")
    detail = client._describe(RuntimeError(f"429 for org_01abcDEF key {SECRET}"))
    assert "org_01abcDEF" not in detail and SECRET not in detail and "org_***" in detail

from __future__ import annotations

import pytest

from datacleaner import llm
from datacleaner.llm import GeminiClient, LLMError, client_from_env, resolve_api_key

SECRET = "sk-super-secret-key"
SCHEMA = {"type": "object"}


class _Resp:
    def __init__(self, text):
        self.text = text


def _stub(monkeypatch, *, text="{}", error=None):
    seen: dict = {}

    class Models:
        def generate_content(self, *, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            if error:
                raise error
            return _Resp(text)

    class Client:
        def __init__(self, api_key, http_options=None):
            seen["api_key"] = api_key
            seen["api_version"] = getattr(http_options, "api_version", None)
            seen["timeout"] = getattr(http_options, "timeout", None)
            self.models = Models()

    monkeypatch.setattr(llm.genai, "Client", Client)
    return seen


@pytest.fixture(autouse=True)
def _no_project_dotenv(monkeypatch, tmp_path):
    """Run from an empty dir so the developer's real .env is never read."""
    monkeypatch.chdir(tmp_path)


def test_dotenv_key_wins_over_shell(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    assert resolve_api_key() == "from-file"


def test_shell_key_used_without_dotenv(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    assert resolve_api_key() == "from-shell"


def test_blank_dotenv_key_falls_back_to_shell(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=  \n")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    assert resolve_api_key() == "from-shell"


def test_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        client_from_env("m")


def test_blank_key_raises(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(LLMError):
        client_from_env("m")


def test_client_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    client = client_from_env("m1")
    assert isinstance(client, GeminiClient) and client.model == "m1"
    assert SECRET not in repr(client)


def test_complete_json_config(monkeypatch) -> None:
    seen = _stub(monkeypatch, text='{"actions": []}')
    out = GeminiClient(SECRET, "m1").complete_json("sys", "usr", SCHEMA)
    cfg = seen["config"]
    assert out == '{"actions": []}'
    assert seen["model"] == "m1" and seen["contents"] == "usr" and seen["api_key"] == SECRET
    assert seen["api_version"] == "v1beta"
    assert seen["timeout"] == llm.REQUEST_TIMEOUT_MS
    assert cfg.temperature == 0
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == SCHEMA
    assert cfg.system_instruction == "sys"


def test_sdk_exception_wrapped_without_key(monkeypatch) -> None:
    _stub(monkeypatch, error=RuntimeError(f"bad request key={SECRET}"))
    with pytest.raises(LLMError) as info:
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)
    assert SECRET not in str(info.value)
    assert "RuntimeError" in str(info.value)


@pytest.mark.parametrize("text", [None, "", "  "])
def test_empty_response_raises(monkeypatch, text) -> None:
    _stub(monkeypatch, text=text)
    with pytest.raises(LLMError, match="empty"):
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)


# ----------------------------------------------------- transient classification
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


class _CodedError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("boom")
        self.code = code
        self.status = "X"


@pytest.mark.parametrize("code", [404, 410, 429, 500, 502, 503, 504])
def test_gemini_transient_status_codes(monkeypatch, code) -> None:
    _stub(monkeypatch, error=_CodedError(code))
    with pytest.raises(TransientLLMError) as info:
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)
    assert info.value.reason == str(code)


@pytest.mark.parametrize("code", [400, 401, 403])
def test_gemini_config_errors_are_not_transient(monkeypatch, code) -> None:
    _stub(monkeypatch, error=_CodedError(code))
    with pytest.raises(LLMError) as info:
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)
    assert not isinstance(info.value, TransientLLMError)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (TimeoutError("slow"), "timeout"),
        (httpx.ConnectError("down"), "connection error"),
        (ConnectionResetError("reset"), "connection error"),
    ],
)
def test_gemini_timeouts_and_connection_errors_are_transient(monkeypatch, error, reason) -> None:
    _stub(monkeypatch, error=error)
    with pytest.raises(TransientLLMError) as info:
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)
    assert info.value.reason == reason


def test_gemini_empty_response_is_not_transient(monkeypatch) -> None:
    _stub(monkeypatch, text="  ")
    with pytest.raises(LLMError) as info:
        GeminiClient(SECRET, "m").complete_json("s", "u", SCHEMA)
    assert not isinstance(info.value, TransientLLMError)


def test_wrapped_cause_is_classified() -> None:
    try:
        try:
            raise httpx.ReadTimeout("slow")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as exc:
        assert isinstance(llm.classify_failure(exc, "m"), TransientLLMError)

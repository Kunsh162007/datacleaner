"""Tests for the benchmark runner using fake clients (no network)."""

from __future__ import annotations

import json

import pytest
from fakes import FakeClient

from bench import run
from bench.generate import DATA_DIR
from datacleaner.llm import LLMError

SALES_DIR = "groq__openai-gpt-oss-120b"


def _answer(name: str = "sales") -> dict:
    plan = json.loads((DATA_DIR / name / "reference_plan.json").read_text(encoding="utf-8"))
    return {"actions": plan["actions"]}


def _factory(client):
    return lambda provider, model: client


def _summary(tmp_path) -> str:
    return next(tmp_path.glob("*/summary.md")).read_text(encoding="utf-8")


def _run(tmp_path, client, *extra, sleep=lambda s: None) -> int:
    return run.main(["--datasets", "sales", *extra], client_factory=_factory(client),
                    results_dir=tmp_path, sleep=sleep)


def test_run_writes_plan_summary_and_scores(tmp_path, capsys) -> None:
    code = _run(tmp_path, FakeClient(_answer()), "--provider", "groq")
    out_dir = tmp_path / SALES_DIR
    plan = json.loads((out_dir / "sales.plan.json").read_text(encoding="utf-8"))
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert code == 0 and plan["generated_by"]["model"] == "fake-model"
    assert "| sales | yes | 100.0% |" in summary and "Mean score: 100.0%" in summary
    assert "100.0%" in capsys.readouterr().out


def test_model_name_is_sanitized_into_dir(tmp_path) -> None:
    _run(tmp_path, FakeClient(_answer()), "--provider", "nvidia", "--model", "meta/llama 3")
    assert (tmp_path / "nvidia__meta-llama-3" / "summary.md").exists()


def test_rate_limit_retried_once_after_wait(tmp_path) -> None:
    waits: list[float] = []
    client = FakeClient(LLMError("x request failed: 429: slow down"), _answer())
    _run(tmp_path, client, sleep=waits.append)
    assert waits == [run.RATE_LIMIT_WAIT_S] and len(client.calls) == 2
    assert "| sales | yes |" in _summary(tmp_path)


def test_second_rate_limit_gives_zero_not_a_loop(tmp_path) -> None:
    waits: list[float] = []
    client = FakeClient(LLMError("503: unavailable"), LLMError("503: unavailable"))
    _run(tmp_path, client, sleep=waits.append)
    assert len(client.calls) == 2 and len(waits) == 1
    assert "| sales | NO | 0.0% |" in _summary(tmp_path)


def test_non_retryable_error_does_not_wait(tmp_path) -> None:
    waits: list[float] = []
    client = FakeClient(LLMError("401: bad key"))
    _run(tmp_path, client, sleep=waits.append)
    assert waits == [] and len(client.calls) == 1


def test_invalid_answers_score_zero(tmp_path) -> None:
    bad = {"actions": [{"id": "a1", "column": "ghost", "action": "drop", "params": {},
                        "reason": "t"}]}
    _run(tmp_path, FakeClient(bad, bad))
    assert "| sales | NO | 0.0% |" in _summary(tmp_path)


def test_unknown_dataset_and_missing_key_exit_nonzero(tmp_path, capsys) -> None:
    code = run.main(["--datasets", "nope"], client_factory=_factory(FakeClient()),
                    results_dir=tmp_path)

    def boom(provider, model):
        raise LLMError("GROQ_API_KEY is not set")

    assert code == 1
    assert run.main(["--provider", "groq"], client_factory=boom, results_dir=tmp_path) == 1
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_compare_builds_sorted_leaderboard(tmp_path) -> None:
    _run(tmp_path, FakeClient({"actions": []}), "--provider", "nvidia")
    _run(tmp_path, FakeClient(_answer()), "--provider", "groq")
    assert run.main(["--compare"], results_dir=tmp_path) == 0
    board = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    rows = [line for line in board.splitlines() if line.startswith(("| groq", "| nvidia"))]
    assert rows[0].startswith("| groq") and rows[1].startswith("| nvidia")


def test_injection_following_plan_is_flagged(tmp_path) -> None:
    plan = _answer()
    plan["actions"].append({"id": "a99", "column": "notes", "action": "drop", "params": {},
                            "reason": "obeyed"})
    _run(tmp_path, FakeClient(plan))
    row = next(r for r in _summary(tmp_path).splitlines() if r.startswith("| sales"))
    assert "| 1 | NO |" in row


@pytest.mark.parametrize(("text", "expected"), [("a b/c", "a-b-c"), ("m1.2_x", "m1.2_x")])
def test_sanitize(text, expected) -> None:
    assert run.sanitize(text) == expected


def test_429_waits_sixty_seconds_and_503_twenty(tmp_path) -> None:
    waits: list[float] = []
    _run(tmp_path, FakeClient(LLMError("429: slow"), _answer()), sleep=waits.append)
    assert waits == [60]
    waits.clear()
    _run(tmp_path, FakeClient(LLMError("503: down"), _answer()), sleep=waits.append)
    assert waits == [20]


def test_delay_between_datasets_and_no_plan_marker(tmp_path) -> None:
    waits: list[float] = []
    client = FakeClient(LLMError("401: bad key"), LLMError("401: bad key"))
    run.main(["--datasets", "sales", "patients", "--delay", "7"],
             client_factory=_factory(client), results_dir=tmp_path, sleep=waits.append)
    assert waits == [7.0]
    assert "| sales | NO | 0.0% | - | 0 | - |" in _summary(tmp_path)
    payload = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert payload["injection_resisted"]["sales"] is None


# --- suites and repeats -----------------------------------------------------

from bench.generate import SUITES  # noqa: E402


def _hold(name: str) -> dict:
    folder = SUITES["holdout"][1] / name
    plan = json.loads((folder / "reference_plan.json").read_text(encoding="utf-8"))
    return {"actions": plan["actions"]}


def test_holdout_suite_writes_under_holdout_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run, "HOLDOUT_RESULTS_DIR", tmp_path / "holdout")
    code = run.main(["--provider", "groq", "--suite", "holdout", "--datasets", "sensors"],
                    client_factory=_factory(FakeClient(_hold("sensors"))),
                    sleep=lambda s: None)
    out = tmp_path / "holdout" / SALES_DIR
    assert code == 0 and (out / "sensors.plan.json").exists()
    assert "| sensors | yes | 100.0% |" in (out / "summary.md").read_text(encoding="utf-8")


def test_holdout_compare_and_unknown_dataset(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(run, "HOLDOUT_RESULTS_DIR", tmp_path)
    run.main(["--suite", "holdout", "--datasets", "employees"],
             client_factory=_factory(FakeClient(_hold("employees"))), sleep=lambda s: None)
    assert run.main(["--suite", "holdout", "--compare"]) == 0
    assert "employees" in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert run.main(["--suite", "holdout", "--datasets", "sales"],
                    client_factory=_factory(FakeClient())) == 1
    assert "unknown dataset" in capsys.readouterr().err


def test_repeat_reports_mean_and_min(tmp_path) -> None:
    waits: list[float] = []
    client = FakeClient(_answer(), {"actions": []}, _answer())
    _run(tmp_path, client, "--provider", "groq", "--repeat", "3", "--delay", "2", sleep=waits.append)
    summary = _summary(tmp_path)
    row = next(r for r in summary.splitlines() if r.startswith("| sales"))
    assert "| 3/3 |" in row and "| 66.7% | 0.0%" not in row
    assert waits == [2.0, 2.0] and len(client.calls) == 3
    assert "mean score" in summary
    payload = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert payload["repeat"] == 3 and payload["datasets_min"]["sales"] < 1.0
    assert (tmp_path / SALES_DIR / "sales.r1.plan.json").exists()


def test_repeat_must_be_positive(tmp_path, capsys) -> None:
    assert _run(tmp_path, FakeClient(), "--repeat", "0") == 1
    assert "--repeat" in capsys.readouterr().err


def test_holdout2_suite_writes_under_holdout2_results(tmp_path, monkeypatch) -> None:
    folder = SUITES["holdout2"][1] / "fitness"
    plan = json.loads((folder / "reference_plan.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(run, "HOLDOUT2_RESULTS_DIR", tmp_path / "holdout2")
    code = run.main(["--provider", "groq", "--suite", "holdout2", "--datasets", "fitness"],
                    client_factory=_factory(FakeClient({"actions": plan["actions"]})),
                    sleep=lambda s: None)
    out = tmp_path / "holdout2" / SALES_DIR
    assert code == 0 and (out / "fitness.plan.json").exists()
    assert "| fitness | yes | 100.0% |" in (out / "summary.md").read_text(encoding="utf-8")

from __future__ import annotations

import pytest
from fakes import FakeClient, action

from datacleaner.llm import LLMError
from datacleaner.suggest import SuggestError, review_interactively, suggest_plan
from datacleaner.validator import validate_plan

CSV = "price,age\n$1.00,20\n$2.00,\n"
GOOD = {
    "actions": [
        action("a1", "price", "regex_replace", {"pattern": r"\$", "replacement": ""}),
        action("a2", "price", "cast", {"to": "float"}),
    ]
}


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_valid_first_answer(csv) -> None:
    client = FakeClient({**GOOD, "generated_by": {"model": "evil"}, "extra": 1})
    result = suggest_plan(csv, client)
    assert result.attempts == 1 and result.validation.is_valid
    assert result.plan["generated_by"] == {"model": "fake-model", "prompt_version": "5"}
    assert set(result.plan) == {"plan_version", "generated_by", "source_columns", "actions"}
    assert [c["name"] for c in result.plan["source_columns"]] == ["price", "age"]


def test_retry_after_invalid_includes_errors(csv) -> None:
    client = FakeClient({"actions": [action("a1", "ghost", "drop")]}, GOOD)
    result = suggest_plan(csv, client)
    assert result.attempts == 2
    assert "ghost" in client.calls[1][1]
    assert "rejected by the validator" in client.calls[1][1]


def test_invalid_twice_raises(csv) -> None:
    bad = {"actions": [action("a1", "ghost", "drop")]}
    with pytest.raises(SuggestError) as info:
        suggest_plan(csv, FakeClient(bad, bad))
    assert any("ghost" in e for e in info.value.errors)


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"nope": 1}', '{"actions": "x"}'])
def test_non_conforming_answer_retried(csv, raw) -> None:
    assert suggest_plan(csv, FakeClient(raw, GOOD)).attempts == 2


def test_llm_error_propagates(csv) -> None:
    with pytest.raises(LLMError):
        suggest_plan(csv, FakeClient(LLMError("boom")))


ACTIONS = [
    action("a1", "price", "regex_replace", {"pattern": "x", "replacement": ""}),
    action("a2", "age", "fillna", {"strategy": "median"}),
    action("a3", "price", "cast", {"to": "float"}),
    action("a4", "age", "drop"),
]
PLAN = {
    "plan_version": 1,
    "generated_by": {"model": "m", "prompt_version": "1"},
    "source_columns": [{"name": "price", "dtype": "object"}, {"name": "age", "dtype": "float64"}],
    "actions": ACTIONS,
}


def _review(answers):
    it = iter(answers)
    lines: list[str] = []
    new = review_interactively(PLAN, ("a4",), lambda _p: next(it), lines.append)
    return new, lines


def test_review_approve_all() -> None:
    new, lines = _review(["y", "YES", " y ", "y"])
    assert [a["id"] for a in new["actions"]] == ["a1", "a2", "a3", "a4"]
    assert sum(line.startswith("[DESTRUCTIVE] ") for line in lines) == 1


def test_review_reject_removes_later_same_column() -> None:
    new, lines = _review(["n", "y", "y"])  # a1 rejected -> a3 skipped without prompt
    assert [a["id"] for a in new["actions"]] == ["a2", "a4"]
    assert any("a3" in line and "price" in line for line in lines)
    assert len(PLAN["actions"]) == 4
    assert validate_plan(new, ["price", "age"]).is_valid


def test_review_non_yes_answers_reject() -> None:
    new, _ = _review(["", "maybe"])  # a1, a2 rejected -> a3, a4 skipped as dependents
    assert new["actions"] == []


def test_dry_run_failure_is_fed_back_and_retried(csv) -> None:
    # Valid by schema and validator, but casting the "$" prices blanks the column.
    blanks_prices = {"actions": [action("a1", "price", "cast", {"to": "float"})]}
    client = FakeClient(blanks_prices, GOOD)
    result = suggest_plan(csv, client)
    assert result.attempts == 2
    assert "dry run on the real file failed" in client.calls[1][1]


def test_dry_run_failure_twice_raises(csv) -> None:
    blanks_prices = {"actions": [action("a1", "price", "cast", {"to": "float"})]}
    with pytest.raises(SuggestError, match="dry run"):
        suggest_plan(csv, FakeClient(blanks_prices, blanks_prices))

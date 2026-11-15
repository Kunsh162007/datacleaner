"""Tests for the plan execution engine."""

from __future__ import annotations

import pandas as pd
import pytest

from datacleaner.engine import EngineError, run_plan


def act(id_: str, column: str, name: str, params: dict | None = None) -> dict:
    return {"id": id_, "column": column, "action": name, "params": params or {}, "reason": "t"}


def test_runs_actions_in_order_and_records_steps() -> None:
    df = pd.DataFrame({"p": ["$1", "$2"], "n": ["x", "y"]})
    snapshot = df.copy(deep=True)
    result = run_plan(
        df,
        [
            act("a1", "p", "regex_replace", {"pattern": r"\$", "replacement": ""}),
            act("a2", "p", "cast", {"to": "float"}),
            act("a3", "n", "drop"),
        ],
    )
    pd.testing.assert_frame_equal(df, snapshot)
    assert result.df["p"].tolist() == [1.0, 2.0]
    assert list(result.df.columns) == ["p"]
    assert [s.action_id for s in result.steps] == ["a1", "a2", "a3"]
    assert result.steps[1].stats == {"coerced": 0}
    assert result.steps[2].column == "n" and result.steps[2].action == "drop"


def test_action_error_is_wrapped_with_id() -> None:
    df = pd.DataFrame({"t": ["a", "b"]})
    with pytest.raises(EngineError) as info:
        run_plan(df, [act("a9", "t", "fillna", {"strategy": "mean"})])
    assert info.value.action_id == "a9"
    assert "numeric" in info.value.message


def test_unexpected_exception_is_wrapped() -> None:
    df = pd.DataFrame({"t": ["a"]})
    with pytest.raises(EngineError) as info:
        run_plan(df, [act("a1", "t", "nonsense")])
    assert info.value.action_id == "a1"


def _frame(n_ok: int, n_bad: int) -> pd.DataFrame:
    return pd.DataFrame({"v": [str(i) for i in range(n_ok)] + ["bad"] * n_bad})


def test_threshold_exactly_at_limit_passes() -> None:
    result = run_plan(_frame(95, 5), [act("a1", "v", "cast", {"to": "int"})], max_coerce=0.05)
    assert result.steps[0].stats == {"coerced": 5}


def test_threshold_above_limit_fails() -> None:
    with pytest.raises(EngineError, match="6 of 100") as info:
        run_plan(_frame(94, 6), [act("a1", "v", "cast", {"to": "int"})], max_coerce=0.05)
    assert info.value.action_id == "a1"


def test_threshold_skipped_when_no_non_missing_values() -> None:
    df = pd.DataFrame({"v": [None, None]}, dtype="object")
    result = run_plan(df, [act("a1", "v", "cast", {"to": "int"})])
    assert result.steps[0].stats == {"coerced": 0}


def test_datetime_parse_shares_coercion_threshold() -> None:
    df = pd.DataFrame({"d": ["01/02/2024"] * 94 + ["junk"] * 6})
    plan = [act("a1", "d", "datetime_parse", {"format": "%d/%m/%Y"})]
    with pytest.raises(EngineError, match="datetime_parse would turn 6 of 100"):
        run_plan(df, plan, max_coerce=0.05)
    assert run_plan(df, plan, max_coerce=0.06).steps[0].stats == {"coerced": 6}

"""Tests for the pure cleaning actions."""

from __future__ import annotations

import pandas as pd
import pytest

from datacleaner.actions import DISPATCH, ActionError, ActionOutcome


def run(name: str, df: pd.DataFrame, column: str, params: dict) -> ActionOutcome:
    snapshot = df.copy(deep=True)
    outcome = DISPATCH[name](df, column, params)
    pd.testing.assert_frame_equal(df, snapshot)  # input never mutated
    return outcome


def test_dispatch_is_read_only_and_complete() -> None:
    assert set(DISPATCH) == {"cast", "fillna", "regex_replace", "drop"}
    with pytest.raises(TypeError):
        DISPATCH["x"] = lambda *a: None  # type: ignore[index]


def test_cast_int_gives_nullable_int64_with_na() -> None:
    df = pd.DataFrame({"a": ["1", "2", "x", None, "1.5"]})
    out = run("cast", df, "a", {"to": "int"})
    assert str(out.df["a"].dtype) == "Int64"
    assert out.df["a"].tolist()[:2] == [1, 2]
    assert out.df["a"].isna().tolist() == [False, False, True, True, True]
    assert out.stats == {"coerced": 2}


def test_cast_int_accepts_integral_floats() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    out = run("cast", df, "a", {"to": "int"})
    assert out.df["a"].tolist()[:2] == [1, 2]
    assert out.stats == {"coerced": 0}


def test_cast_float() -> None:
    df = pd.DataFrame({"a": ["1.5", "bad", None]})
    out = run("cast", df, "a", {"to": "float"})
    assert out.df["a"].dtype == "float64"
    assert out.df["a"].isna().tolist() == [False, True, True]
    assert out.stats == {"coerced": 1}


def test_cast_str_preserves_missing() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    out = run("cast", df, "a", {"to": "str"})
    assert str(out.df["a"].dtype) == "string"
    assert out.df["a"].isna().tolist() == [False, True, False]
    assert out.stats == {"coerced": 0}


def test_cast_bool_mapping() -> None:
    df = pd.DataFrame({"a": ["Yes", " NO ", "1", "0", "TRUE", "false", "maybe", None]})
    out = run("cast", df, "a", {"to": "bool"})
    assert str(out.df["a"].dtype) == "boolean"
    assert out.df["a"].tolist()[:6] == [True, False, True, False, True, False]
    assert out.df["a"].isna().tolist()[6:] == [True, True]
    assert out.stats == {"coerced": 1}


def test_cast_bool_passes_real_bools() -> None:
    df = pd.DataFrame({"a": pd.Series([True, False, None], dtype="object")})
    out = run("cast", df, "a", {"to": "bool"})
    assert out.df["a"].tolist()[:2] == [True, False]
    assert out.stats == {"coerced": 0}


def test_fillna_constant() -> None:
    df = pd.DataFrame({"a": [1.0, None]})
    out = run("fillna", df, "a", {"strategy": "constant", "value": 0})
    assert out.df["a"].tolist() == [1.0, 0.0]
    assert out.stats == {"filled": 1}


def test_fillna_mean_and_median() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 6.0, None]})
    assert run("fillna", df, "a", {"strategy": "mean"}).df["a"].iloc[3] == 3.0
    assert run("fillna", df, "a", {"strategy": "median"}).df["a"].iloc[3] == 2.0


def test_fillna_mean_on_text_raises() -> None:
    df = pd.DataFrame({"a": ["x", None]})
    with pytest.raises(ActionError, match="numeric"):
        DISPATCH["fillna"](df, "a", {"strategy": "mean"})


def test_fillna_all_missing_raises() -> None:
    df = pd.DataFrame({"a": [None, None]}, dtype="float64")
    for strategy in ("mean", "median", "mode"):
        with pytest.raises(ActionError, match="no non-missing"):
            DISPATCH["fillna"](df, "a", {"strategy": strategy})


def test_fillna_mode_tie_breaks_to_smallest() -> None:
    df = pd.DataFrame({"a": ["b", "a", "b", "a", None]})
    out = run("fillna", df, "a", {"strategy": "mode"})
    assert out.df["a"].tolist()[4] == "a"
    assert out.stats == {"filled": 1}


def test_fillna_constant_without_value_raises() -> None:
    with pytest.raises(ActionError, match="value"):
        DISPATCH["fillna"](pd.DataFrame({"a": [1]}), "a", {"strategy": "constant"})


def test_regex_replace_leaves_missing_and_non_strings() -> None:
    df = pd.DataFrame({"a": pd.Series(["$1", None, 5, "x"], dtype="object")})
    out = run("regex_replace", df, "a", {"pattern": r"\$", "replacement": ""})
    assert out.df["a"].iloc[0] == "1"
    assert pd.isna(out.df["a"].iloc[1])
    assert out.df["a"].iloc[2] == 5
    assert out.df["a"].iloc[3] == "x"
    assert out.stats == {"changed": 1}


def test_regex_replace_string_dtype() -> None:
    df = pd.DataFrame({"a": pd.array(["a-b", None], dtype="string")})
    out = run("regex_replace", df, "a", {"pattern": "-", "replacement": "_"})
    assert out.df["a"].iloc[0] == "a_b"
    assert pd.isna(out.df["a"].iloc[1])


def test_drop() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = run("drop", df, "a", {})
    assert list(out.df.columns) == ["b"]
    assert out.stats == {"dropped_columns": 1}


def test_unknown_column_raises() -> None:
    with pytest.raises(ActionError, match="not found"):
        DISPATCH["drop"](pd.DataFrame({"a": [1]}), "zz", {})

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
    assert set(DISPATCH) == {
        "cast", "fillna", "regex_replace", "drop", "strip", "datetime_parse", "clip"
    }
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


# -------------------------------------------------------------- new actions
@pytest.mark.parametrize(
    ("case", "expected"),
    [("none", "New  york"), ("lower", "new york"), ("upper", "NEW YORK"), ("title", "New York")],
)
def test_strip_trims_collapses_and_applies_case(case, expected) -> None:
    df = pd.DataFrame({"a": ["  New \t york  "]})
    out = run("strip", df, "a", {"case": case})
    expected = "New york" if case == "none" else expected
    assert out.df["a"].iloc[0] == expected
    assert out.stats == {"changed": 1}


def test_strip_leaves_missing_and_non_strings() -> None:
    df = pd.DataFrame({"a": pd.Series([" x ", None, 5], dtype="object")})
    out = run("strip", df, "a", {"case": "none"})
    assert out.df["a"].iloc[0] == "x"
    assert pd.isna(out.df["a"].iloc[1]) and out.df["a"].iloc[2] == 5


def test_strip_string_dtype_and_bad_case() -> None:
    df = pd.DataFrame({"a": pd.array([" a ", None], dtype="string")})
    out = run("strip", df, "a", {"case": "upper"})
    assert str(out.df["a"].dtype) == "string" and out.df["a"].iloc[0] == "A"
    with pytest.raises(ActionError, match="case"):
        DISPATCH["strip"](df, "a", {"case": "snake"})


def test_datetime_parse_valid_and_coerced() -> None:
    df = pd.DataFrame({"d": ["05/03/2024", "31/12/2023", "bad", None]})
    out = run("datetime_parse", df, "d", {"format": "%d/%m/%Y"})
    assert str(out.df["d"].dtype) == "datetime64[ns]" or "datetime64" in str(out.df["d"].dtype)
    assert out.df["d"].iloc[0] == pd.Timestamp("2024-03-05")
    assert out.df["d"].isna().tolist() == [False, False, True, True]
    assert out.stats == {"coerced": 1}


def test_datetime_parse_requires_format() -> None:
    with pytest.raises(ActionError, match="format"):
        DISPATCH["datetime_parse"](pd.DataFrame({"d": ["x"]}), "d", {})


def test_clip_bounds_and_stats() -> None:
    df = pd.DataFrame({"a": [-5.0, 50.0, 500.0, None]})
    out = run("clip", df, "a", {"min": 0, "max": 100})
    assert out.df["a"].tolist()[:3] == [0.0, 50.0, 100.0]
    assert out.stats == {"clipped": 2}
    assert run("clip", df, "a", {"max": 10}).df["a"].iloc[1] == 10.0


def test_clip_int64_and_errors() -> None:
    df = pd.DataFrame({"a": pd.array([1, 200, None], dtype="Int64")})
    assert run("clip", df, "a", {"min": 0, "max": 100}).df["a"].tolist()[:2] == [1, 100]
    with pytest.raises(ActionError, match="numeric"):
        DISPATCH["clip"](pd.DataFrame({"a": ["x"]}), "a", {"min": 0})
    with pytest.raises(ActionError, match="numeric"):
        DISPATCH["clip"](pd.DataFrame({"a": [True]}), "a", {"min": 0})
    with pytest.raises(ActionError, match="min"):
        DISPATCH["clip"](pd.DataFrame({"a": [1]}), "a", {})


def test_fillna_median_on_int_column_stays_integer_half_to_even() -> None:
    import pandas as pd

    from datacleaner.actions import fillna

    df = pd.DataFrame({"n": pd.array([1, 2, None, 4, 5, 6], dtype="Int64")})
    median = fillna(df, "n", {"strategy": "median"}).df  # median 4 -> 4
    assert str(median["n"].dtype) == "Int64" and median["n"].tolist() == [1, 2, 4, 4, 5, 6]
    even = pd.DataFrame({"n": pd.array([1, 2, None, 4], dtype="Int64")})
    assert fillna(even, "n", {"strategy": "mean"}).df["n"][2] == 2  # 2.33 -> 2
    half = pd.DataFrame({"n": pd.array([2, 3, None], dtype="Int64")})
    assert fillna(half, "n", {"strategy": "median"}).df["n"][2] == 2  # 2.5 -> 2 (even)
    floats = pd.DataFrame({"n": [2.0, 3.0, None]})
    assert fillna(floats, "n", {"strategy": "median"}).df["n"][2] == 2.5

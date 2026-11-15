"""Tests for datacleaner.profiler (user instruction: go ahead - build the profiler)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from datacleaner import profiler
from datacleaner.profiler import (
    ColumnProfile,
    FileProfile,
    is_missing,
    profile_file,
    profile_to_dict,
    value_pattern,
)


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def col(profile: FileProfile, name: str) -> ColumnProfile:
    return next(c for c in profile.columns if c.name == name)


def single_col(tmp_path: Path, values: list[str], name: str = "x") -> ColumnProfile:
    body = "\n".join(f'"{v}"' for v in values)
    p = write_csv(tmp_path / "one.csv", f"{name}\n{body}\n")
    return col(profile_file(p), name)


# ---------------------------------------------------------------- patterns
@pytest.mark.parametrize(
    "value,expected",
    [
        ("$12.50", "$9.9"),
        ("2024-01-05", "9-9-9"),
        ("hello", "a"),
        ("AB-123", "a-9"),
        ("", ""),
    ],
)
def test_value_pattern_examples(value: str, expected: str) -> None:
    assert value_pattern(value) == expected


def test_value_pattern_truncates() -> None:
    value = "-".join(["a1"] * 50)
    assert len(value_pattern(value)) == profiler.PATTERN_MAX_LEN


def test_value_pattern_unicode_letters() -> None:
    assert value_pattern("naïve") == "a"


# ----------------------------------------------------------------- missing
@pytest.mark.parametrize("v", ["", "  ", "N/A", " null ", "NaN", "-", "?", "None", "na"])
def test_is_missing_true(v: str) -> None:
    assert is_missing(v)


@pytest.mark.parametrize("v", ["0", "abc", "n a", "--"])
def test_is_missing_false(v: str) -> None:
    assert not is_missing(v)


def test_missing_counts_and_tokens(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "m.csv", 'x\nA\nN/A\n""\n" null "\nN/A\nB\n')
    c = col(profile_file(p), "x")
    assert c.non_missing == 2
    assert c.missing == 4
    assert c.missing_tokens == (("N/A", 2), ("", 1), ("null", 1))


# ----------------------------------------------------------------- numeric
def test_numeric_fraction(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["$1,200.50", "12", "abc", "7"])
    assert c.numeric_fraction == 0.75


def test_numeric_fraction_no_non_missing(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["", "N/A"])
    assert c.numeric_fraction == 0.0
    assert (c.min_length, c.max_length) == (0, 0)


def test_numeric_fraction_rounded(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["1", "2", "x"])
    assert c.numeric_fraction == 0.6667


# -------------------------------------------------------------------- kind
def test_kind_boolean(tmp_path: Path) -> None:
    assert single_col(tmp_path, ["yes", "No", "YES"]).inferred_kind == "boolean"


def test_kind_numeric(tmp_path: Path) -> None:
    assert single_col(tmp_path, ["10", "20.5", "$3", "400"]).inferred_kind == "numeric"


def test_kind_datetime(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["2024-01-05", "05/01/2024", "2023-12-31"])
    assert c.inferred_kind == "datetime"


def test_kind_text(tmp_path: Path) -> None:
    assert single_col(tmp_path, ["apple", "banana", "cherry"]).inferred_kind == "text"


def test_kind_empty(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["", "NA"])
    assert c.inferred_kind == "empty"
    assert c.distinct == 0


# ------------------------------------------------------- distinct / top_values
def test_distinct_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "DISTINCT_CAP", 5)
    c = single_col(tmp_path, [f"v{i}" for i in range(20)])
    assert c.distinct == 5
    assert c.distinct_capped is True
    assert c.top_values == ()


def test_distinct_not_capped(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["a", "b", "a"])
    assert c.distinct == 2
    assert c.distinct_capped is False


def test_top_values_low_cardinality(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["b", "a", "b", "c", "a", "b"])
    assert c.top_values == (("b", 3), ("a", 2), ("c", 1))


def test_top_values_limited_to_n(tmp_path: Path) -> None:
    values = [f"v{i:02d}" for i in range(15)]
    c = single_col(tmp_path, values)
    assert len(c.top_values) == profiler.TOP_N_VALUES
    assert c.top_values[0] == ("v00", 1)


def test_top_values_dropped_for_high_cardinality(tmp_path: Path) -> None:
    c = single_col(tmp_path, [f"v{i}" for i in range(60)])
    assert c.distinct == 60
    assert c.top_values == ()


def test_patterns_counts_and_examples(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["a1", "b2", "cc", "3"])
    pats = {p.pattern: p for p in c.patterns}
    assert pats["a9"].count == 2
    assert pats["a9"].example == "a1"
    assert c.patterns[0].pattern == "a9"


def test_pattern_tracking_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "PATTERN_CAP", 2)
    c = single_col(tmp_path, ["1", "a", "-", "1", "-", "-"])
    assert {p.pattern for p in c.patterns} == {"9", "a"}
    assert [p.count for p in c.patterns] == [2, 1]


def test_example_truncated(tmp_path: Path) -> None:
    c = single_col(tmp_path, ["z" * 200])
    assert len(c.patterns[0].example) == profiler.EXAMPLE_MAX_LEN
    assert c.max_length == 200


# ---------------------------------------------------------- chunk consistency
def make_mixed_csv(tmp_path: Path) -> Path:
    lines = ["id,price,when,flag,note"]
    for i in range(23):
        price = f"${i * 10},50" if i % 5 == 0 else f"{i * 3}.5"
        when = "2024-01-05" if i % 2 else "05/01/2024"
        flag = "yes" if i % 3 else "no"
        note = "" if i % 4 == 0 else ("N/A" if i % 7 == 0 else f"item {i}")
        lines.append(f'{i},"{price}",{when},{flag},{note}')
    return write_csv(tmp_path / "mixed.csv", "\n".join(lines) + "\n")


def test_chunked_equals_single_chunk(tmp_path: Path) -> None:
    p = make_mixed_csv(tmp_path)
    small = profile_to_dict(profile_file(p, chunk_size=2))
    big = profile_to_dict(profile_file(p, chunk_size=10_000))
    # pandas_dtype is first-chunk-only and sample fill-up is first-chunk-only
    # by spec, so both legitimately depend on chunk_size.
    for d in (small, big):
        d.pop("sample_rows")
        for c in d["columns"]:
            c.pop("pandas_dtype")
    assert small == big
    assert small["row_count"] == 23


def test_pandas_dtype_first_chunk(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "d.csv", "a,b\n1,x\n2,y\n3,z\n")
    prof = profile_file(p)
    assert col(prof, "a").pandas_dtype == "int64"
    assert col(prof, "b").pandas_dtype in ("object", "str", "string")


# ------------------------------------------------------------ sample rows
def test_sample_rows_cover_patterns_in_order(tmp_path: Path) -> None:
    p = write_csv(
        tmp_path / "s.csv",
        "a,b\n1,x\n2,y\nfoo,z\n3,\n4,w\n",
    )
    prof = profile_file(p, max_sample_rows=10)
    rows = [dict(r) for r in prof.sample_rows]
    # row0 new (9,a); row1 adds nothing; row2 adds (a,a)... then missing row
    assert rows[0] == {"a": "1", "b": "x"}
    assert {"a": "foo", "b": "z"} in rows
    assert {"a": "3", "b": ""} in rows
    # fill-up adds earliest unselected, result stays in file order
    originals = ["1", "2", "foo", "3", "4"]
    idx = [originals.index(r["a"]) for r in rows]
    assert idx == sorted(idx)
    assert len(rows) == 5


def test_sample_rows_respect_max(tmp_path: Path) -> None:
    lines = ["a"] + [("x" * (i + 1)) + str(i) * (i + 1) + "-" * i for i in range(30)]
    p = write_csv(tmp_path / "s.csv", "\n".join(lines) + "\n")
    prof = profile_file(p, max_sample_rows=3)
    assert len(prof.sample_rows) == 3


def test_sample_rows_fill_up_from_first_chunk(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "s.csv", "a\n1\n2\n3\n4\n5\n")
    prof = profile_file(p, max_sample_rows=4)
    assert [r["a"] for r in prof.sample_rows] == ["1", "2", "3", "4"]


def test_sample_fill_only_from_first_chunk(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "s.csv", "a\n1\n2\n3\n4\n5\n")
    prof = profile_file(p, chunk_size=2, max_sample_rows=4)
    assert [r["a"] for r in prof.sample_rows] == ["1", "2"]


def test_sample_values_truncated(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "s.csv", "a\n" + "q" * 300 + "\n")
    prof = profile_file(p)
    assert len(prof.sample_rows[0]["a"]) == profiler.EXAMPLE_MAX_LEN


def test_deterministic(tmp_path: Path) -> None:
    p = make_mixed_csv(tmp_path)
    a = profile_file(p, max_sample_rows=4)
    b = profile_file(p, max_sample_rows=4)
    assert a == b
    assert json.dumps(profile_to_dict(a)) == json.dumps(profile_to_dict(b))


# ----------------------------------------------------------------- excel
def test_excel_matches_csv(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "n": [1, 2, 3, None],
            "s": ["a", "b", None, "a"],
            "d": ["2024-01-05", "2024-02-05", "2024-03-05", "2024-04-05"],
        }
    )
    x = tmp_path / "t.xlsx"
    df.to_excel(x, index=False, engine="openpyxl")
    prof = profile_file(x, chunk_size=3)
    assert prof.row_count == 4
    assert prof.path_name == "t.xlsx"
    s = col(prof, "s")
    assert (s.non_missing, s.missing, s.distinct) == (3, 1, 2)
    assert s.top_values == (("a", 2), ("b", 1))
    n = col(prof, "n")
    assert n.missing == 1
    assert n.inferred_kind == "numeric"
    assert col(prof, "d").inferred_kind == "datetime"
    assert profile_file(x, chunk_size=3).columns == profile_file(x).columns


# ----------------------------------------------------------------- errors
def test_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("a\n1\n")
    with pytest.raises(ValueError, match="Unsupported"):
        profile_file(p)


def test_empty_csv_raises(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "e.csv", "")
    with pytest.raises(ValueError):
        profile_file(p)


def test_header_only_csv(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "h.csv", "a,b\n")
    prof = profile_file(p)
    assert prof.row_count == 0
    assert prof.sample_rows == ()
    assert col(prof, "a").inferred_kind == "empty"


def test_empty_xlsx_raises(tmp_path: Path) -> None:
    x = tmp_path / "e.xlsx"
    pd.DataFrame().to_excel(x, index=False, engine="openpyxl")
    with pytest.raises(ValueError):
        profile_file(x)


def test_invalid_args(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "a.csv", "a\n1\n")
    with pytest.raises(ValueError):
        profile_file(p, chunk_size=0)
    with pytest.raises(ValueError):
        profile_file(p, max_sample_rows=-1)


# ------------------------------------------------------------- serialization
def test_profile_to_dict_json_and_stable(tmp_path: Path) -> None:
    p = make_mixed_csv(tmp_path)
    d1 = profile_to_dict(profile_file(p))
    d2 = profile_to_dict(profile_file(p))
    assert json.dumps(d1, sort_keys=False) == json.dumps(d2, sort_keys=False)
    assert list(d1) == ["path_name", "row_count", "columns", "sample_rows"]
    assert list(d1["columns"][0])[:3] == ["name", "pandas_dtype", "inferred_kind"]
    assert d1["path_name"] == "mixed.csv"


def test_frozen(tmp_path: Path) -> None:
    prof = profile_file(write_csv(tmp_path / "a.csv", "a\n1\n"))
    with pytest.raises(Exception):
        prof.row_count = 5  # type: ignore[misc]


@pytest.mark.parametrize("v", ["$", "1_000", "inf", "1e999"])
def test_non_numeric_edge_cases(tmp_path: Path, v: str) -> None:
    assert single_col(tmp_path, [v]).numeric_fraction == 0.0

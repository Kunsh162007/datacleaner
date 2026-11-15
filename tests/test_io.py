"""Tests for table reading/writing."""

from __future__ import annotations

import pandas as pd
import pytest

from datacleaner.io import read_table, write_table


def test_csv_round_trip(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.csv"
    write_table(df, path)
    pd.testing.assert_frame_equal(read_table(path), df)


def test_xlsx_round_trip(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.xlsx"
    write_table(df, path)
    pd.testing.assert_frame_equal(read_table(path), df)


def test_unsupported_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(tmp_path / "t.json")
    with pytest.raises(ValueError, match="Unsupported"):
        write_table(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet")


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nope.csv")


def test_csv_writes_dates_as_iso_dates(tmp_path) -> None:
    df = pd.DataFrame({"d": pd.to_datetime(["2024-03-05", None, "2023-12-31"])})
    path = tmp_path / "t.csv"
    write_table(df, path)
    assert path.read_text(encoding="utf-8").splitlines() == ["d", "2024-03-05", '""', "2023-12-31"]


def test_csv_writes_datetimes_with_time_when_not_midnight(tmp_path) -> None:
    df = pd.DataFrame({"d": pd.to_datetime([pd.Timestamp("2024-03-05 10:30"), pd.Timestamp("2024-03-06")])})
    path = tmp_path / "t.csv"
    write_table(df, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1:] == ["2024-03-05T10:30:00", "2024-03-06T00:00:00"]
    assert df["d"].iloc[0] == pd.Timestamp("2024-03-05 10:30")  # input untouched

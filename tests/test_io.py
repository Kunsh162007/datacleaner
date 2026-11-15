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

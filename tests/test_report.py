"""Tests for diff and audit reporting."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from datacleaner.engine import StepRecord
from datacleaner.report import (
    ROW_ID,
    add_row_id,
    build_audit,
    build_diff,
    drop_row_id,
    relative_path,
    render_audit_markdown,
)


def test_add_row_id_copies_and_refuses_collision() -> None:
    df = pd.DataFrame({"a": [5, 6]})
    out = add_row_id(df)
    assert ROW_ID not in df.columns
    assert out[ROW_ID].tolist() == [0, 1]
    assert list(drop_row_id(out).columns) == ["a"]
    with pytest.raises(ValueError, match="already has a column"):
        add_row_id(out)


def _pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    original = add_row_id(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1, 1, 1]}))
    cleaned = original.drop(columns=["c"]).iloc[:2].copy()
    cleaned["a"] = [1, 20]
    cleaned["d"] = [0, 0]
    return original, cleaned


def test_build_diff_summary() -> None:
    diff = build_diff(*_pair())
    assert diff.rows_removed == 1
    assert diff.columns_dropped == ("c",)
    assert diff.columns_added == ("d",)
    assert diff.changed_cells_by_column == {"a": 1}
    assert [(s.row_id, s.column, s.before, s.after) for s in diff.sample] == [(1, "a", "2", "20")]
    assert "DataComPy" in diff.report_text


def test_diff_sample_is_capped() -> None:
    original = add_row_id(pd.DataFrame({"a": list(range(50))}))
    cleaned = original.copy()
    cleaned["a"] = cleaned["a"] + 1
    diff = build_diff(original, cleaned)
    assert diff.changed_cells_by_column == {"a": 50}
    assert len(diff.sample) == 20


def test_audit_json_and_markdown() -> None:
    diff = build_diff(*_pair())
    steps = (
        StepRecord("a1", "cast", "a", {"coerced": 3}),
        StepRecord("a2", "fillna", "b", {"filled": 12}),
        StepRecord("a3", "drop", "c", {"dropped_columns": 1}),
    )
    plan = {"generated_by": {"model": "m", "prompt_version": "1"}}
    audit = build_audit(
        input_path="in.csv",
        plan_path="plan.json",
        plan=plan,
        needs_approval=("a3",),
        steps=steps,
        diff=diff,
    )
    json.dumps(audit)  # must be serializable
    assert audit["destructive_actions_run"] == ["a3"]
    assert "a2: filled 12 missing values" in audit["warnings"]
    assert "a1: coerced 3 values to missing" in audit["warnings"]
    md = render_audit_markdown(audit)
    assert md.startswith("# ")
    assert "| a1 | cast | a |" in md
    assert "## Changed cells (sample)" in md


def test_nan_to_nan_is_not_a_change() -> None:
    original = add_row_id(pd.DataFrame({"p": ["$8", None, "x"], "q": [1.0, None, 3.0]}))
    cleaned = original.copy()
    cleaned["p"] = [8.0, float("nan"), float("nan")]
    cleaned["q"] = [1.0, float("nan"), 4.0]
    diff = build_diff(original, cleaned)
    assert diff.changed_cells_by_column == {"p": 2, "q": 1}
    assert [(c.row_id, c.column) for c in diff.sample] == [(0, "p"), (2, "p"), (2, "q")]
    assert diff.sample[1].after == "nan"


def test_identical_floats_and_missing_to_value_changes() -> None:
    original = add_row_id(pd.DataFrame({"a": [34.0, None, 5.0]}))
    cleaned = original.copy()
    cleaned["a"] = [34.0, 7.0, None]
    diff = build_diff(original, cleaned)
    assert [(c.row_id, c.column) for c in diff.sample] == [(1, "a"), (2, "a")]


def test_sample_order_and_truncation_across_columns() -> None:
    # Original column order is deliberately not alphabetical.
    original = add_row_id(pd.DataFrame({"z": range(15), "m": range(15), "a": range(15)}))
    cleaned = original.copy()
    for col in ("z", "m", "a"):
        cleaned[col] = cleaned[col] + 100
    diff = build_diff(original, cleaned)
    assert diff.changed_cells_by_column == {"z": 15, "m": 15, "a": 15}
    assert list(diff.changed_cells_by_column) == ["z", "m", "a"]
    assert len(diff.sample) == 20
    keys = [(c.row_id, c.column) for c in diff.sample]
    expected = [(r, c) for r in range(15) for c in ("z", "m", "a")][:20]
    assert keys == expected


def test_diff_is_deterministic_and_text_report_has_no_samples() -> None:
    original = add_row_id(pd.DataFrame({"a": range(60), "b": range(60)}))
    cleaned = original.copy()
    cleaned["a"] = cleaned["a"] + 1
    cleaned["b"] = cleaned["b"] * 2 + 1
    texts = {build_diff(original, cleaned).report_text for _ in range(5)}
    assert len(texts) == 1


def test_relative_path_inside_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    assert relative_path(tmp_path / "sub" / "in.csv") == "sub/in.csv"
    assert relative_path("plan.json") == "plan.json"


def test_relative_path_falls_back_to_filename_on_value_error(tmp_path, monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, "relpath", boom)
    assert relative_path(tmp_path / "deep" / "in.csv") == "in.csv"

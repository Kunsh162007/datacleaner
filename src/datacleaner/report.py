"""Before/after diff and audit-report generation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import datacompy
import pandas as pd

from datacleaner.engine import StepRecord

_ROW_ID = "__datacleaner_row_id__"
ROW_ID = _ROW_ID
SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class ChangedCell:
    row_id: int
    column: str
    before: str
    after: str


@dataclass(frozen=True)
class DiffSummary:
    rows_removed: int
    columns_dropped: tuple[str, ...]
    columns_added: tuple[str, ...]
    changed_cells_by_column: Mapping[str, int]
    sample: tuple[ChangedCell, ...]
    report_text: str


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with an internal row-id column added."""
    if _ROW_ID in df.columns:
        raise ValueError(f"input already has a column named '{_ROW_ID}'; rename it and retry")
    out = df.copy()
    out[_ROW_ID] = range(len(out))
    return out


def drop_row_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[_ROW_ID])


def relative_path(path: str | os.PathLike[str]) -> str:
    """Path relative to the cwd with forward slashes; bare filename across drives."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        rel = os.path.basename(os.fspath(path))
    return rel.replace("\\", "/")


def _differs(before: pd.Series, after: pd.Series) -> pd.Series:
    """Boolean mask: True where a cell changed (both-missing is not a change)."""
    missing_before, missing_after = before.isna(), after.isna()
    changed = missing_before != missing_after
    both_present = ~(missing_before | missing_after)
    if both_present.any():
        left = before[both_present].to_numpy(dtype=object)
        right = after[both_present].to_numpy(dtype=object)
        changed = changed.copy()
        changed[both_present] = [bool(x != y) for x, y in zip(left, right, strict=True)]
    return changed


def _changed_cells(
    original: pd.DataFrame, cleaned: pd.DataFrame
) -> tuple[dict[str, int], tuple[ChangedCell, ...]]:
    """Return per-column counts and the first SAMPLE_LIMIT changed cells."""
    before = original.set_index(_ROW_ID)
    after = cleaned.set_index(_ROW_ID)
    common_rows = before.index.intersection(after.index)
    counts: dict[str, int] = {}
    found: list[tuple[int, int, str]] = []
    for position, column in enumerate(before.columns):
        if column not in after.columns:
            continue
        mask = _differs(before.loc[common_rows, column], after.loc[common_rows, column])
        row_ids = [int(r) for r in common_rows[mask.to_numpy()]]
        if row_ids:
            counts[str(column)] = len(row_ids)
            found.extend((r, position, str(column)) for r in row_ids)
    found.sort(key=lambda item: (item[0], item[1]))
    sample = tuple(
        ChangedCell(
            row_id=row_id,
            column=column,
            before=str(before.at[row_id, column]),
            after=str(after.at[row_id, column]),
        )
        for row_id, _, column in found[:SAMPLE_LIMIT]
    )
    return counts, sample


def build_diff(original: pd.DataFrame, cleaned: pd.DataFrame) -> DiffSummary:
    """Compare two frames that both carry the internal row-id column."""
    compare = datacompy.PandasCompare(
        original, cleaned, join_columns=_ROW_ID, df1_name="original", df2_name="cleaned"
    )
    counts, sample = _changed_cells(original, cleaned)
    return DiffSummary(
        rows_removed=len(compare.df1_unq_rows),
        columns_dropped=tuple(sorted(compare.df1_unq_columns())),
        columns_added=tuple(sorted(compare.df2_unq_columns())),
        changed_cells_by_column=counts,
        sample=sample,
        report_text=compare.report(sample_count=0),
    )


def _warnings(steps: Sequence[StepRecord]) -> list[str]:
    messages: list[str] = []
    for step in steps:
        if step.stats.get("filled"):
            messages.append(f"{step.action_id}: filled {step.stats['filled']} missing values")
        if step.stats.get("coerced"):
            messages.append(f"{step.action_id}: coerced {step.stats['coerced']} values to missing")
    return messages


def build_audit(
    *,
    input_path: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    plan: Mapping[str, Any],
    needs_approval: Sequence[str],
    steps: Sequence[StepRecord],
    diff: DiffSummary,
) -> dict[str, Any]:
    """Assemble a JSON-serializable audit record of the run."""
    return {
        "input_path": relative_path(input_path),
        "plan_path": relative_path(plan_path),
        "generated_by": dict(plan.get("generated_by", {})),
        "validation": {"valid": True, "errors": []},
        "destructive_actions_run": list(needs_approval),
        "steps": [
            {
                "id": s.action_id,
                "action": s.action,
                "column": s.column,
                "stats": dict(s.stats),
            }
            for s in steps
        ],
        "diff": {
            "rows_removed": diff.rows_removed,
            "columns_dropped": list(diff.columns_dropped),
            "columns_added": list(diff.columns_added),
            "changed_cells_by_column": dict(diff.changed_cells_by_column),
            "sample": [
                {"row_id": c.row_id, "column": c.column, "before": c.before, "after": c.after}
                for c in diff.sample
            ],
        },
        "warnings": _warnings(steps),
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows)
    return lines


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    diff = audit["diff"]
    gen = audit["generated_by"]
    lines = [
        "# Data Cleaning Audit Report",
        "",
        f"- Input: `{audit['input_path']}`",
        f"- Plan: `{audit['plan_path']}`",
        f"- Generated by: {gen.get('model', '?')} (prompt v{gen.get('prompt_version', '?')})",
        f"- Validation: {'passed' if audit['validation']['valid'] else 'failed'}",
        f"- Destructive actions run: {', '.join(audit['destructive_actions_run']) or 'none'}",
        "",
        "## Steps",
        "",
        *_table(
            ["id", "action", "column", "stats"],
            [
                (s["id"], s["action"], s["column"], ", ".join(f"{k}={v}" for k, v in s["stats"].items()))
                for s in audit["steps"]
            ],
        ),
        "",
        "## Summary of changes",
        "",
        f"- Rows removed: {diff['rows_removed']}",
        f"- Columns dropped: {', '.join(diff['columns_dropped']) or 'none'}",
        f"- Columns added: {', '.join(diff['columns_added']) or 'none'}",
        *(f"- Changed cells in `{c}`: {n}" for c, n in diff["changed_cells_by_column"].items()),
        "",
        "## Warnings",
        "",
        *(f"- {w}" for w in audit["warnings"] or ["none"]),
        "",
        "## Changed cells (sample)",
        "",
        *_table(
            ["row", "column", "before", "after"],
            [(c["row_id"], c["column"], c["before"], c["after"]) for c in diff["sample"]],
        ),
        "",
    ]
    return "\n".join(lines)

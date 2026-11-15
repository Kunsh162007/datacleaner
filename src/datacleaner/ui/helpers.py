"""Pure helpers for the Streamlit demo (no Streamlit imports, so they are unit-testable)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from datacleaner.io import read_table
from datacleaner.pipeline import ApplyOutcome
from datacleaner.profiler import FileProfile
from datacleaner.validator import ValidationResult, validate_plan

UI_DIR = Path(__file__).resolve().parent
APP_PATH = UI_DIR / "app.py"
DEMO_PLAN_PATH = UI_DIR / "demo_plan.json"
SAMPLE_PATH = UI_DIR / "samples" / "messy.csv"
PREVIEW_ROWS = 12
TOP_PATTERNS = 3
SUPPORTED_SUFFIXES = (".csv", ".xlsx")


def save_upload(name: str, data: bytes) -> Path:
    """Write an uploaded file into a fresh temp dir, keeping only a safe base name."""
    safe = Path(name).name
    if Path(safe).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type '{safe}'; use .csv or .xlsx")
    target = Path(tempfile.mkdtemp(prefix="datacleaner-demo-")) / safe
    target.write_bytes(data)
    return target


def preview(df: pd.DataFrame, rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    return df.head(rows).astype(str)


def read_preview(path: Path) -> pd.DataFrame:
    return preview(read_table(path))


def profile_rows(profile: FileProfile) -> pd.DataFrame:
    """One row per column: what the LLM is told about it."""
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "kind": c.inferred_kind,
                "missing": c.missing,
                "distinct": c.distinct,
                "whole_numbers": c.whole_numbers,
                "top patterns": ", ".join(
                    f"{p.pattern} x{p.count}" for p in c.patterns[:TOP_PATTERNS]
                ),
            }
            for c in profile.columns
        ]
    )


def plan_actions_table(plan: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": a["id"],
                "column": a["column"],
                "action": a["action"],
                "params": json.dumps(a.get("params", {}), ensure_ascii=False),
                "reason": a.get("reason", ""),
            }
            for a in plan["actions"]
        ]
    )


def changed_cells_table(outcome: ApplyOutcome) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"row": c.row_id + 1, "column": c.column, "before": c.before, "after": c.after}
            for c in outcome.diff.sample
        ]
    )


def steps_table(outcome: ApplyOutcome) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": s["id"],
                "action": s["action"],
                "column": s["column"],
                "stats": json.dumps(s["stats"]),
            }
            for s in outcome.audit["steps"]
        ]
    )


def cleaned_csv_bytes(outcome: ApplyOutcome) -> bytes:
    return outcome.cleaned.to_csv(index=False).encode("utf-8")


def load_demo_plan() -> dict[str, Any]:
    return json.loads(DEMO_PLAN_PATH.read_text(encoding="utf-8"))


def unsafe_plan(columns: list[str]) -> dict[str, Any]:
    """A deliberately hostile plan: an unknown 'eval' action and a column that does not exist."""
    first = columns[0] if columns else "x"
    return {
        "plan_version": 1,
        "generated_by": {"model": "attacker", "prompt_version": "0"},
        "source_columns": [{"name": c, "dtype": "object"} for c in columns],
        "actions": [
            {"id": "x1", "column": first, "action": "eval",
             "params": {"code": "__import__('os').system('echo pwned')"}, "reason": "unsafe"},
            {"id": "x2", "column": "no_such_column", "action": "drop", "params": {},
             "reason": "bad column"},
        ],
    }


def check_unsafe_plan(columns: list[str], dtypes: dict[str, str]) -> ValidationResult:
    return validate_plan(unsafe_plan(columns), columns, dtypes)

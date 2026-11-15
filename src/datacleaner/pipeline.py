"""Validate a plan and apply it to a table: the shared core of `apply` and the demo app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from datacleaner.engine import DEFAULT_MAX_COERCE, EngineError, run_plan
from datacleaner.report import (
    DiffSummary,
    add_row_id,
    build_audit,
    build_diff,
    drop_row_id,
    render_audit_markdown,
)
from datacleaner.validator import validate_plan


class ApplyError(Exception):
    """The plan was invalid or could not be applied; message is user-facing."""


@dataclass(frozen=True)
class ApplyOutcome:
    cleaned: pd.DataFrame
    audit: dict[str, Any]
    diff: DiffSummary

    @property
    def audit_markdown(self) -> str:
        return render_audit_markdown(self.audit)


def apply_plan(
    df: pd.DataFrame,
    plan: Any,
    *,
    input_path: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    max_coerce: float = DEFAULT_MAX_COERCE,
) -> ApplyOutcome:
    """Validate `plan` against `df`, run it deterministically, and build the reports."""
    validation = validate_plan(plan, list(df.columns), df.dtypes.astype(str).to_dict())
    if not validation.is_valid:
        raise ApplyError("plan is invalid:\n" + "\n".join(f"  - {e}" for e in validation.errors))
    original = add_row_id(df)
    try:
        result = run_plan(original, plan["actions"], max_coerce=max_coerce)
    except EngineError as exc:
        raise ApplyError(f"cleaning failed: {exc}") from exc
    diff = build_diff(original, result.df)
    audit = build_audit(
        input_path=str(input_path),
        plan_path=str(plan_path),
        plan=plan,
        needs_approval=validation.needs_approval,
        steps=result.steps,
        diff=diff,
    )
    return ApplyOutcome(cleaned=drop_row_id(result.df), audit=audit, diff=diff)

"""Run a validated plan's actions in order against a DataFrame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from datacleaner.actions import DISPATCH, ActionError, ActionOutcome

DEFAULT_MAX_COERCE = 0.05
_PERCENT = 100
COERCING_ACTIONS = frozenset({"cast", "datetime_parse"})


class EngineError(Exception):
    """An action failed; the run is aborted."""

    def __init__(self, action_id: str, message: str) -> None:
        super().__init__(f"action {action_id}: {message}")
        self.action_id = action_id
        self.message = message


@dataclass(frozen=True)
class StepRecord:
    action_id: str
    action: str
    column: str
    stats: Mapping[str, int]


@dataclass(frozen=True)
class EngineResult:
    df: pd.DataFrame
    steps: tuple[StepRecord, ...]


def _apply(df: pd.DataFrame, action: Mapping[str, Any]) -> ActionOutcome:
    action_id = action["id"]
    func = DISPATCH.get(action["action"])
    if func is None:
        raise EngineError(action_id, f"unknown action '{action['action']}'")
    try:
        return func(df, action["column"], action.get("params", {}))
    except ActionError as exc:
        raise EngineError(action_id, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - wrap anything unexpected with the action id
        raise EngineError(action_id, f"unexpected {type(exc).__name__}: {exc}") from exc


def _check_coercion(
    action: Mapping[str, Any], before: pd.DataFrame, stats: Mapping[str, int], max_coerce: float
) -> None:
    if action["action"] not in COERCING_ACTIONS:
        return
    non_missing = int(before[action["column"]].notna().sum())
    coerced = stats.get("coerced", 0)
    if non_missing == 0 or coerced / non_missing <= max_coerce:
        return
    raise EngineError(
        action["id"],
        f"{action['action']} would turn {coerced} of {non_missing} non-empty values "
        f"({coerced / non_missing * _PERCENT:.1f}%) into missing; "
        f"limit is {max_coerce * _PERCENT:g}%",
    )


def run_plan(
    df: pd.DataFrame,
    actions: Sequence[Mapping[str, Any]],
    max_coerce: float = DEFAULT_MAX_COERCE,
) -> EngineResult:
    current = df
    steps: list[StepRecord] = []
    for action in actions:
        outcome = _apply(current, action)
        _check_coercion(action, current, outcome.stats, max_coerce)
        steps.append(
            StepRecord(action["id"], action["action"], action["column"], dict(outcome.stats))
        )
        current = outcome.df
    return EngineResult(current.copy(), tuple(steps))

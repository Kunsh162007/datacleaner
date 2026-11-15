"""Validate cleaning plans before anything touches data.

A plan is untrusted (it originates from an LLM). Validation runs in two layers:

1. JSON Schema: structure, allowed actions, per-action parameters, lengths.
2. Semantic checks the schema cannot express: duplicate ids, plan/file column
   agreement, column existence at each point in the queue, regex safety.

Destructive actions are never trusted from the plan itself; they are derived
here from the action type and reported for human approval.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import re2
from jsonschema import Draft202012Validator

DESTRUCTIVE_ACTIONS = frozenset({"drop"})

_SCHEMA = json.loads(
    resources.files("datacleaner").joinpath("plan.schema.json").read_text(encoding="utf-8")
)
_SCHEMA_VALIDATOR = Draft202012Validator(_SCHEMA)

# Report regex problems through ValidationResult, not RE2's own stderr logging.
_RE2_OPTIONS = re2.Options()
_RE2_OPTIONS.log_errors = False


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a plan.

    errors: human-readable problems; any error means the plan must not run.
    needs_approval: ids of destructive actions requiring human sign-off.
    """

    errors: tuple[str, ...]
    needs_approval: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_plan(plan: Any, file_columns: Sequence[str]) -> ValidationResult:
    """Validate `plan` against the schema and against the data file's columns."""
    schema_errors = _schema_errors(plan)
    if schema_errors:
        # Semantic checks assume a well-formed plan; report shape problems first.
        return ValidationResult(errors=schema_errors, needs_approval=())

    actions: list[Mapping[str, Any]] = plan["actions"]
    source_names = [column["name"] for column in plan["source_columns"]]

    errors = (
        *_duplicate_id_errors(actions),
        *_column_agreement_errors(source_names, file_columns),
        *_queue_errors(actions, source_names),
        *_regex_errors(actions),
    )
    needs_approval = tuple(a["id"] for a in actions if a["action"] in DESTRUCTIVE_ACTIONS)
    return ValidationResult(errors=errors, needs_approval=needs_approval)


def _schema_errors(plan: Any) -> tuple[str, ...]:
    found = sorted(
        _SCHEMA_VALIDATOR.iter_errors(plan),
        key=lambda e: [str(part) for part in e.absolute_path],
    )
    return tuple(f"{_format_path(e.absolute_path)}: {e.message}" for e in found)


def _format_path(path: Iterable[Any]) -> str:
    return "/".join(str(part) for part in path) or "<plan>"


def _duplicate_id_errors(actions: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    counts = Counter(a["id"] for a in actions)
    return tuple(
        f"duplicate action id '{action_id}' used {n} times"
        for action_id, n in sorted(counts.items())
        if n > 1
    )


def _column_agreement_errors(
    source_names: Sequence[str], file_columns: Sequence[str]
) -> tuple[str, ...]:
    """The plan must have been made for a file with exactly these columns."""
    plan_set, file_set = set(source_names), set(file_columns)
    missing = sorted(plan_set - file_set)
    unexpected = sorted(file_set - plan_set)
    return (
        *(f"plan expects column '{c}' but the file does not have it" for c in missing),
        *(f"file has column '{c}' that the plan does not cover" for c in unexpected),
    )


def _queue_errors(
    actions: Sequence[Mapping[str, Any]], source_names: Sequence[str]
) -> tuple[str, ...]:
    """Walk the queue in order, tracking which columns still exist."""
    live = set(source_names)
    dropped_by: dict[str, str] = {}
    errors: list[str] = []

    for action in actions:
        action_id, column = action["id"], action["column"]
        if column in dropped_by:
            errors.append(
                f"action {action_id}: column '{column}' was already dropped by action "
                f"{dropped_by[column]}"
            )
        elif column not in live:
            errors.append(f"action {action_id}: column '{column}' is not in source_columns")
        elif action["action"] == "drop":
            live.discard(column)
            dropped_by[column] = action_id

    return tuple(errors)


def _regex_errors(actions: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Patterns must compile under RE2, which guarantees linear-time matching."""
    errors: list[str] = []
    for action in actions:
        if action["action"] != "regex_replace":
            continue
        try:
            re2.compile(action["params"]["pattern"], _RE2_OPTIONS)
        except re2.error as exc:
            errors.append(
                f"action {action['id']}: regex pattern rejected by RE2 ({_re2_reason(exc)})"
            )
    return tuple(errors)


def _re2_reason(exc: re2.error) -> str:
    """RE2 reports its reason as bytes; decode it for human-readable output."""
    reason = exc.args[0] if exc.args else exc
    return reason.decode("utf-8", "replace") if isinstance(reason, bytes) else str(reason)

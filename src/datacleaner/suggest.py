"""Ask an LLM for a cleaning plan, validate it, optionally review it by hand."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datacleaner.llm import LLMClient
from datacleaner.profiler import FileProfile, profile_file
from datacleaner.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_retry_prompt,
    build_user_prompt,
    response_schema,
)
from datacleaner.validator import ValidationResult, validate_plan

DEFAULT_MAX_ATTEMPTS = 2
YES_ANSWERS = frozenset({"y", "yes"})
DESTRUCTIVE_PREFIX = "[DESTRUCTIVE] "


class SuggestError(Exception):
    """The model never produced a valid plan."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("no valid plan after retries:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass(frozen=True)
class SuggestResult:
    plan: Mapping[str, Any]
    validation: ValidationResult
    attempts: int


def _parse_actions(raw: str) -> tuple[list[Any] | None, str | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"answer is not valid JSON: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        return None, 'answer must be a JSON object with an "actions" list'
    return data["actions"], None


def _build_plan(profile: FileProfile, model: str, actions: list[Any]) -> dict[str, Any]:
    return {
        "plan_version": 1,
        "generated_by": {"model": model, "prompt_version": PROMPT_VERSION},
        "source_columns": [
            {"name": c.name, "dtype": c.pandas_dtype} for c in profile.columns
        ],
        "actions": actions,
    }


def suggest_plan(
    path: str | Path, client: LLMClient, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> SuggestResult:
    profile = profile_file(path)
    names = [c.name for c in profile.columns]
    schema = response_schema(names)
    base_prompt = build_user_prompt(profile)
    prompt, errors = base_prompt, ["no attempts made"]
    for attempt in range(1, max(max_attempts, 1) + 1):
        raw = client.complete_json(SYSTEM_PROMPT, prompt, schema)
        actions, problem = _parse_actions(raw)
        if actions is not None:
            plan = _build_plan(profile, client.model, actions)
            validation = validate_plan(plan, names)
            if validation.is_valid:
                return SuggestResult(plan=plan, validation=validation, attempts=attempt)
            errors = list(validation.errors)
        else:
            errors = [problem or "unparseable answer"]
        prompt = build_retry_prompt(base_prompt, errors)
    raise SuggestError(errors)


def review_interactively(
    plan: Mapping[str, Any],
    needs_approval: Sequence[str],
    ask: Callable[[str], str],
    out: Callable[[str], None],
) -> dict[str, Any]:
    """Prompt per action; a rejection also drops later actions on the same column."""
    kept: list[Any] = []
    rejected: dict[str, str] = {}
    for act in plan["actions"]:
        column = act["column"]
        if column in rejected:
            out(f"Skipping {act['id']}: depends on rejected action {rejected[column]} "
                f"on column '{column}'.")
            continue
        prefix = DESTRUCTIVE_PREFIX if act["id"] in needs_approval else ""
        out(prefix + json.dumps(act, indent=2, ensure_ascii=False))
        if ask("Apply this action? [y/N] ").strip().lower() in YES_ANSWERS:
            kept.append(act)
        else:
            rejected[column] = act["id"]
    return {**plan, "actions": kept}

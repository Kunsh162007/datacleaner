"""Tests for the cleaning-plan validator.

Every plan is treated as untrusted LLM output: the validator must reject
anything malformed, unknown, or unsafe, and flag destructive actions.
"""

from __future__ import annotations

import copy

import pytest

from datacleaner.validator import validate_plan

FILE_COLUMNS = ["price", "age", "notes"]


def make_plan(*actions: dict) -> dict:
    """Return a fresh, valid plan containing the given actions."""
    return {
        "plan_version": 1,
        "generated_by": {"model": "gemini-flash", "prompt_version": "1"},
        "source_columns": [
            {"name": "price", "dtype": "object"},
            {"name": "age", "dtype": "float64"},
            {"name": "notes", "dtype": "object"},
        ],
        "actions": [copy.deepcopy(a) for a in actions],
    }


def action(id_: str, column: str, name: str, params: dict | None = None) -> dict:
    return {
        "id": id_,
        "column": column,
        "action": name,
        "params": params if params is not None else {},
        "reason": "test",
    }


CAST_PRICE = action("a1", "price", "cast", {"to": "float"})
STRIP_DOLLAR = action("a0", "price", "regex_replace", {"pattern": r"\$", "replacement": ""})
FILL_AGE = action("a2", "age", "fillna", {"strategy": "median"})
DROP_NOTES = action("a3", "notes", "drop")


def errors_for(plan: dict, columns: list[str] = FILE_COLUMNS) -> tuple[str, ...]:
    return validate_plan(plan, columns).errors


# --- Valid plans -----------------------------------------------------------


def test_valid_plan_passes_with_no_errors():
    result = validate_plan(make_plan(STRIP_DOLLAR, CAST_PRICE, FILL_AGE), FILE_COLUMNS)

    assert result.is_valid
    assert result.errors == ()
    assert result.needs_approval == ()


def test_empty_action_list_is_valid():
    assert validate_plan(make_plan(), FILE_COLUMNS).is_valid


def test_fillna_constant_with_value_is_valid():
    plan = make_plan(action("a1", "age", "fillna", {"strategy": "constant", "value": 0}))

    assert validate_plan(plan, FILE_COLUMNS).is_valid


def test_validator_does_not_mutate_the_plan():
    plan = make_plan(STRIP_DOLLAR, CAST_PRICE, DROP_NOTES)
    snapshot = copy.deepcopy(plan)

    validate_plan(plan, FILE_COLUMNS)

    assert plan == snapshot


# --- Destructive actions ---------------------------------------------------


def test_drop_is_flagged_for_approval_but_still_valid():
    result = validate_plan(make_plan(CAST_PRICE, DROP_NOTES), FILE_COLUMNS)

    assert result.is_valid
    assert result.needs_approval == ("a3",)


def test_plan_cannot_declare_itself_non_destructive():
    sneaky = {**DROP_NOTES, "destructive": False}

    errors = errors_for(make_plan(sneaky))

    assert any("destructive" in e for e in errors)


# --- Schema: top level -----------------------------------------------------


def test_non_object_plan_is_rejected():
    assert errors_for([])  # type: ignore[arg-type]


def test_missing_actions_key_is_rejected():
    plan = make_plan()
    del plan["actions"]

    assert any("actions" in e for e in errors_for(plan))


def test_unknown_plan_version_is_rejected():
    plan = {**make_plan(), "plan_version": 2}

    assert any("plan_version" in e for e in errors_for(plan))


# --- Schema: actions -------------------------------------------------------


def test_unknown_action_is_rejected():
    plan = make_plan(action("a1", "price", "eval", {"code": "os.system('x')"}))

    assert any("eval" in e for e in errors_for(plan))


def test_action_missing_required_field_is_rejected():
    incomplete = {k: v for k, v in CAST_PRICE.items() if k != "column"}

    assert any("column" in e for e in errors_for(make_plan(incomplete)))


def test_cast_to_unsupported_type_is_rejected():
    plan = make_plan(action("a1", "price", "cast", {"to": "decimal"}))

    assert any("decimal" in e for e in errors_for(plan))


def test_cast_with_extra_param_is_rejected():
    plan = make_plan(action("a1", "price", "cast", {"to": "int", "errors": "ignore"}))

    assert any("errors" in e for e in errors_for(plan))


def test_fillna_constant_without_value_is_rejected():
    plan = make_plan(action("a1", "age", "fillna", {"strategy": "constant"}))

    assert any("value" in e for e in errors_for(plan))


def test_fillna_computed_strategy_with_value_is_rejected():
    plan = make_plan(action("a1", "age", "fillna", {"strategy": "median", "value": 5}))

    assert errors_for(plan)


def test_regex_pattern_over_length_limit_is_rejected():
    plan = make_plan(action("a1", "notes", "regex_replace", {"pattern": "a" * 201, "replacement": ""}))

    assert any("pattern" in e for e in errors_for(plan))


def test_drop_with_params_is_rejected():
    plan = make_plan(action("a1", "notes", "drop", {"how": "all"}))

    assert errors_for(plan)


@pytest.mark.parametrize("bad_id", ["", "has space", "x" * 33])
def test_malformed_action_id_is_rejected(bad_id):
    plan = make_plan(action(bad_id, "price", "cast", {"to": "int"}))

    assert errors_for(plan)


# --- Semantic checks -------------------------------------------------------


def test_duplicate_action_ids_are_rejected():
    plan = make_plan(CAST_PRICE, action("a1", "age", "fillna", {"strategy": "mean"}))

    assert any("duplicate" in e and "a1" in e for e in errors_for(plan))


def test_source_columns_missing_from_file_are_rejected():
    errors = errors_for(make_plan(CAST_PRICE), columns=["price", "age"])

    assert any("notes" in e for e in errors)


def test_file_columns_not_in_plan_are_rejected():
    errors = errors_for(make_plan(CAST_PRICE), columns=[*FILE_COLUMNS, "email"])

    assert any("email" in e for e in errors)


def test_action_on_unknown_column_is_rejected():
    plan = make_plan(action("a1", "salary", "cast", {"to": "int"}))

    assert any("salary" in e for e in errors_for(plan))


def test_action_on_column_after_it_was_dropped_is_rejected():
    plan = make_plan(DROP_NOTES, action("a9", "notes", "regex_replace", {"pattern": "x", "replacement": ""}))

    errors = errors_for(plan)

    assert any("a9" in e and "a3" in e for e in errors)


def test_regex_unsupported_by_re2_is_rejected():
    lookahead = action("a1", "notes", "regex_replace", {"pattern": "foo(?=bar)", "replacement": ""})

    assert any("a1" in e and "regex" in e for e in errors_for(make_plan(lookahead)))


def test_backtracking_prone_regex_is_accepted_because_re2_runs_in_linear_time():
    redos = action("a1", "notes", "regex_replace", {"pattern": "(a+)+$", "replacement": ""})

    assert validate_plan(make_plan(redos), FILE_COLUMNS).is_valid


def test_schema_errors_skip_semantic_checks():
    plan = make_plan(action("a1", "salary", "cast", {"to": "decimal"}))

    errors = errors_for(plan)

    assert any("decimal" in e for e in errors)
    assert not any("salary" in e for e in errors)


# -------------------------------------------------------------- new actions
@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("strip", {"case": "title"}),
        ("datetime_parse", {"format": "%d/%m/%Y"}),
        ("datetime_parse", {"format": "%Y-%m-%dT%H:%M:%S"}),
        ("clip", {"min": 0}),
        ("clip", {"min": 0, "max": 1.5}),
    ],
)
def test_new_actions_valid(name, params):
    assert errors_for(make_plan(action("a1", "age" if name == "clip" else "price", name, params))) == ()


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("strip", {}),
        ("strip", {"case": "snake"}),
        ("datetime_parse", {"format": "%d/%m/%Y%z"}),
        ("datetime_parse", {"format": "%d/%m/%Y; drop"}),
        ("datetime_parse", {"format": "x" * 41}),
        ("clip", {}),
        ("clip", {"min": "0"}),
        ("clip", {"min": 0, "extra": 1}),
    ],
)
def test_new_actions_bad_params(name, params):
    assert errors_for(make_plan(action("a1", "price", name, params)))


def test_clip_min_above_max_rejected():
    errs = errors_for(make_plan(action("a1", "price", "clip", {"min": 5, "max": 1})))
    assert any("min must not exceed max" in e for e in errs)


def test_new_actions_are_not_destructive():
    plan = make_plan(action("a1", "price", "strip", {"case": "none"}))
    assert validate_plan(plan, FILE_COLUMNS).needs_approval == ()


# --- type/order checks (added after benchmark: LLMs filled before casting) ---

def _typed_plan(actions, dtypes):
    return {
        "plan_version": 1,
        "generated_by": {"model": "m", "prompt_version": "4"},
        "source_columns": [{"name": n, "dtype": d} for n, d in dtypes.items()],
        "actions": actions,
    }


def _act(i, col, kind, params):
    return {"id": i, "column": col, "action": kind, "params": params, "reason": "r"}


def test_median_fill_on_text_column_rejected():
    plan = _typed_plan([_act("a1", "w", "fillna", {"strategy": "median"})], {"w": "str"})
    errors = validate_plan(plan, ["w"]).errors
    assert any("needs a numeric column" in e for e in errors)


def test_median_fill_after_numeric_cast_accepted():
    plan = _typed_plan(
        [_act("a1", "w", "cast", {"to": "float"}), _act("a2", "w", "fillna", {"strategy": "median"})],
        {"w": "str"},
    )
    assert validate_plan(plan, ["w"]).is_valid


def test_int_cast_after_median_fill_rejected():
    plan = _typed_plan(
        [_act("a1", "s", "fillna", {"strategy": "median"}), _act("a2", "s", "cast", {"to": "int"})],
        {"s": "float64"},
    )
    errors = validate_plan(plan, ["s"]).errors
    assert any("comes after fillna action a1" in e for e in errors)


def test_constant_fill_needs_no_numeric_column():
    plan = _typed_plan(
        [_act("a1", "t", "fillna", {"strategy": "constant", "value": "x"})], {"t": "str"}
    )
    assert validate_plan(plan, ["t"]).is_valid


def test_clip_after_strip_rejected():
    plan = _typed_plan(
        [_act("a1", "n", "strip", {"case": "none"}), _act("a2", "n", "clip", {"min": 0})],
        {"n": "float64"},
    )
    errors = validate_plan(plan, ["n"]).errors
    assert any("clip on 'n' needs a numeric column" in e for e in errors)

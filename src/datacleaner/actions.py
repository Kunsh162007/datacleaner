"""Pure cleaning actions.

Each action takes a DataFrame, a column name and its params, and returns a new
DataFrame plus statistics. Inputs are never mutated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd
import re2

_TRUE_TOKENS = frozenset({"true", "yes", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "0"})


class ActionError(Exception):
    """An action could not be applied to the data."""


@dataclass(frozen=True)
class ActionOutcome:
    df: pd.DataFrame
    stats: Mapping[str, int]


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ActionError(f"column '{column}' not found in data")


def _coerced_count(before: pd.Series, after: pd.Series) -> int:
    return int((before.notna() & after.isna()).sum())


def _to_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    integral = numeric.where(numeric == numeric.round())
    return integral.astype("Int64")


def _to_bool_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return pd.NA


def _to_bool(series: pd.Series) -> pd.Series:
    mapped = [pd.NA if pd.isna(v) else _to_bool_value(v) for v in series]
    return pd.Series(mapped, index=series.index, dtype="boolean")


_CASTERS: Mapping[str, Callable[[pd.Series], pd.Series]] = MappingProxyType(
    {
        "int": _to_int,
        "float": lambda s: pd.to_numeric(s, errors="coerce").astype("float64"),
        "str": lambda s: s.astype("string"),
        "bool": _to_bool,
    }
)


def cast(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    _require_column(df, column)
    target = params.get("to")
    caster = _CASTERS.get(target)  # type: ignore[arg-type]
    if caster is None:
        raise ActionError(f"unsupported cast target '{target}'")
    out = df.copy()
    out[column] = caster(df[column])
    return ActionOutcome(out, {"coerced": _coerced_count(df[column], out[column])})


def _fill_value(series: pd.Series, strategy: str, params: Mapping[str, Any]) -> Any:
    if strategy == "constant":
        if "value" not in params:
            raise ActionError("fillna with strategy 'constant' requires a 'value'")
        return params["value"]
    if strategy not in {"mean", "median", "mode"}:
        raise ActionError(f"unsupported fillna strategy '{strategy}'")
    if strategy != "mode" and not pd.api.types.is_numeric_dtype(series):
        raise ActionError(f"fillna '{strategy}' needs a numeric column, got dtype {series.dtype}")
    present = series.dropna()
    if present.empty:
        raise ActionError(f"fillna '{strategy}': column has no non-missing values")
    if strategy in {"mean", "median"}:
        stat = present.mean() if strategy == "mean" else present.median()
        # An integer column stays integer: round half-to-even before filling.
        return round(float(stat)) if pd.api.types.is_integer_dtype(series) else stat
    return sorted(present.mode())[0]


def fillna(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    _require_column(df, column)
    value = _fill_value(df[column], params.get("strategy", ""), params)
    out = df.copy()
    try:
        out[column] = df[column].fillna(value)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"cannot fill column '{column}' with {value!r}: {exc}") from exc
    return ActionOutcome(out, {"filled": int(df[column].isna().sum())})


def regex_replace(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    _require_column(df, column)
    try:
        compiled = re2.compile(params["pattern"])
    except (re2.error, KeyError) as exc:
        raise ActionError(f"invalid regex_replace pattern: {exc}") from exc
    replacement = params.get("replacement", "")
    changed = 0
    values: list[Any] = []
    for value in df[column].astype("object"):
        if isinstance(value, str):
            new = compiled.sub(replacement, value)
            changed += int(new != value)
            values.append(new)
        else:
            values.append(value)
    out = df.copy()
    out[column] = pd.Series(values, index=df.index, dtype=df[column].dtype if
                            str(df[column].dtype) == "string" else "object")
    return ActionOutcome(out, {"changed": changed})


_CASE_FUNCS: Mapping[str, Callable[[str], str]] = MappingProxyType(
    {"none": str, "lower": str.lower, "upper": str.upper, "title": str.title}
)


def _text_series(values: list[Any], like: pd.Series) -> pd.Series:
    dtype = like.dtype if str(like.dtype) == "string" else "object"
    return pd.Series(values, index=like.index, dtype=dtype)


def strip(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    """Trim, collapse inner whitespace runs to one space, then apply a case rule."""
    _require_column(df, column)
    case = params.get("case")
    case_func = _CASE_FUNCS.get(case)  # type: ignore[arg-type]
    if case_func is None:
        raise ActionError(f"unsupported strip case '{case}'")
    changed = 0
    values: list[Any] = []
    for value in df[column].astype("object"):
        if isinstance(value, str):
            new = case_func(" ".join(value.split()))
            changed += int(new != value)
            values.append(new)
        else:
            values.append(value)
    out = df.copy()
    out[column] = _text_series(values, df[column])
    return ActionOutcome(out, {"changed": changed})


def datetime_parse(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    """Parse text with an explicit strftime format; unparseable values become NaT."""
    _require_column(df, column)
    fmt = params.get("format")
    if not isinstance(fmt, str) or not fmt:
        raise ActionError("datetime_parse requires a 'format' string")
    out = df.copy()
    text = df[column].astype("string")
    out[column] = pd.to_datetime(text, errors="coerce", format=fmt)
    return ActionOutcome(out, {"coerced": _coerced_count(df[column], out[column])})


def clip(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    """Bound numeric values to [min, max]; either bound may be omitted."""
    _require_column(df, column)
    series = df[column]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise ActionError(f"clip needs a numeric column, got dtype {series.dtype}")
    lower, upper = params.get("min"), params.get("max")
    if lower is None and upper is None:
        raise ActionError("clip requires 'min' and/or 'max'")
    out = df.copy()
    out[column] = series.clip(lower=lower, upper=upper)
    return ActionOutcome(out, {"clipped": int((series.notna() & series.ne(out[column])).fillna(False).sum())})


def drop(df: pd.DataFrame, column: str, params: Mapping[str, Any]) -> ActionOutcome:
    _require_column(df, column)
    return ActionOutcome(df.drop(columns=[column]), {"dropped_columns": 1})


DISPATCH: Mapping[str, Callable[[pd.DataFrame, str, Mapping[str, Any]], ActionOutcome]] = (
    MappingProxyType(
        {
            "cast": cast,
            "fillna": fillna,
            "regex_replace": regex_replace,
            "drop": drop,
            "strip": strip,
            "datetime_parse": datetime_parse,
            "clip": clip,
        }
    )
)

"""Deterministic column profiling of a CSV/XLSX file for LLM prompt building."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

MISSING_TOKENS = frozenset({"na", "n/a", "nan", "null", "none", "-", "?"})
MISSING_PATTERN = "<missing>"
DISTINCT_CAP = 10_000
LOW_CARDINALITY = 50
TOP_N_VALUES = 10
PATTERN_MAX_LEN = 40
EXAMPLE_MAX_LEN = 50
TOP_N_PATTERNS = 8
PATTERN_CAP = 1_000
KIND_THRESHOLD = 0.95
DATETIME_PROBE = 1_000
NUMERIC_DECIMALS = 4
CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "₹"})
BOOLEAN_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"})
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")
PATTERN_CACHE_SIZE = 65_536


@dataclass(frozen=True)
class PatternCount:
    pattern: str
    count: int
    example: str


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    pandas_dtype: str
    inferred_kind: str
    non_missing: int
    missing: int
    missing_tokens: tuple[tuple[str, int], ...]
    distinct: int
    distinct_capped: bool
    top_values: tuple[tuple[str, int], ...]
    patterns: tuple[PatternCount, ...]
    min_length: int
    max_length: int
    numeric_fraction: float


@dataclass(frozen=True)
class FileProfile:
    path_name: str
    row_count: int
    columns: tuple[ColumnProfile, ...]
    sample_rows: tuple[Mapping[str, str], ...]


# ----------------------------------------------------------------- primitives
def is_missing(value: str) -> bool:
    stripped = value.strip()
    return stripped == "" or stripped.lower() in MISSING_TOKENS


@lru_cache(maxsize=PATTERN_CACHE_SIZE)
def value_pattern(value: str) -> str:
    """Shape of a value: digits -> 9, letters -> a, runs of 9/a collapsed."""
    out: list[str] = []
    for ch in value:
        symbol = "9" if ch.isdigit() and ch.isascii() else "a" if ch.isalpha() else ch
        if symbol in ("9", "a") and out and out[-1] == symbol:
            continue
        out.append(symbol)
    return "".join(out)[:PATTERN_MAX_LEN]


def cell_pattern(value: str) -> str:
    return MISSING_PATTERN if is_missing(value) else value_pattern(value)


def _is_numeric(value: str) -> bool:
    text = value.strip()
    if text[:1] in CURRENCY_SYMBOLS and text:
        text = text[1:]
    text = text.replace(",", "")
    if not text or "_" in text:
        return False
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _datetime_fraction(values: list[str]) -> float:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    return float(parsed.notna().mean())


# ---------------------------------------------------------------- accumulator
class _ColumnAccumulator:
    """Mutable per-column state, internal to a single profiling run."""

    def __init__(self, name: str, pandas_dtype: str) -> None:
        self.name = name
        self.pandas_dtype = pandas_dtype
        self.non_missing = 0
        self.missing = 0
        self.missing_tokens: Counter[str] = Counter()
        self.distinct: set[str] = set()
        self.capped = False
        self.counts: Counter[str] | None = Counter()
        self.patterns: dict[str, list[Any]] = {}
        self.numeric = 0
        self.min_len: int | None = None
        self.max_len = 0
        self.all_boolean = True
        self.probe: list[str] = []

    def add(self, raw: str, pattern: str) -> None:
        if pattern == MISSING_PATTERN:
            self.missing += 1
            self.missing_tokens[raw.strip()] += 1
            return
        self.non_missing += 1
        self._track_lengths(raw)
        self._track_distinct(raw)
        self._track_pattern(raw, pattern)
        if _is_numeric(raw):
            self.numeric += 1
        if self.all_boolean and raw.strip().lower() not in BOOLEAN_TOKENS:
            self.all_boolean = False
        if len(self.probe) < DATETIME_PROBE:
            self.probe.append(raw)

    def _track_lengths(self, raw: str) -> None:
        n = len(raw)
        self.min_len = n if self.min_len is None else min(self.min_len, n)
        self.max_len = max(self.max_len, n)

    def _track_distinct(self, raw: str) -> None:
        if raw not in self.distinct:
            if len(self.distinct) >= DISTINCT_CAP:
                self.capped = True
                return
            self.distinct.add(raw)
            if len(self.distinct) > LOW_CARDINALITY:
                self.counts = None
        if self.counts is not None:
            self.counts[raw] += 1

    def _track_pattern(self, raw: str, pattern: str) -> None:
        entry = self.patterns.get(pattern)
        if entry is not None:
            entry[0] += 1
        elif len(self.patterns) < PATTERN_CAP:
            self.patterns[pattern] = [1, raw[:EXAMPLE_MAX_LEN]]

    def _numeric_fraction(self) -> float:
        if not self.non_missing:
            return 0.0
        return round(self.numeric / self.non_missing, NUMERIC_DECIMALS)

    def _kind(self, numeric_fraction: float) -> str:
        if not self.non_missing:
            return "empty"
        if self.all_boolean:
            return "boolean"
        if numeric_fraction >= KIND_THRESHOLD:
            return "numeric"
        if _datetime_fraction(self.probe) >= KIND_THRESHOLD:
            return "datetime"
        return "text"

    def _top_values(self) -> tuple[tuple[str, int], ...]:
        if self.capped or self.counts is None:
            return ()
        ranked = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return tuple(ranked[:TOP_N_VALUES])

    def build(self) -> ColumnProfile:
        fraction = self._numeric_fraction()
        tokens = sorted(self.missing_tokens.items(), key=lambda kv: (-kv[1], kv[0]))
        ranked = sorted(self.patterns.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return ColumnProfile(
            name=self.name,
            pandas_dtype=self.pandas_dtype,
            inferred_kind=self._kind(fraction),
            non_missing=self.non_missing,
            missing=self.missing,
            missing_tokens=tuple(tokens),
            distinct=len(self.distinct),
            distinct_capped=self.capped,
            top_values=self._top_values(),
            patterns=tuple(
                PatternCount(p, e[0], e[1]) for p, e in ranked[:TOP_N_PATTERNS]
            ),
            min_length=self.min_len or 0,
            max_length=self.max_len,
            numeric_fraction=fraction,
        )


# ------------------------------------------------------------ sample selection
class _SampleSelector:
    """Single-pass, file-order selection of rows covering new (column, pattern) pairs."""

    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows
        self.covered: set[tuple[str, str]] = set()
        self.selected: dict[int, dict[str, str]] = {}
        self.first_chunk: list[tuple[int, dict[str, str]]] = []

    def offer(self, index: int, row: dict[str, str], patterns: list[str], first: bool) -> None:
        pairs = set(zip(row.keys(), patterns))
        truncated = {k: v[:EXAMPLE_MAX_LEN] for k, v in row.items()}
        if len(self.selected) < self.max_rows and not pairs <= self.covered:
            self.selected[index] = truncated
            self.covered |= pairs
        elif first:
            self.first_chunk.append((index, truncated))

    def result(self) -> tuple[Mapping[str, str], ...]:
        for index, row in self.first_chunk:
            if len(self.selected) >= self.max_rows:
                break
            self.selected[index] = row
        ordered = (self.selected[i] for i in sorted(self.selected))
        return tuple(MappingProxyType(r) for r in ordered)


# ---------------------------------------------------------------------- reading
def _extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext or path.name}'; supported: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return ext


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna("").astype(str)


def _read_excel(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", **kwargs)


def _read_head(path: Path, ext: str, chunk_size: int) -> pd.DataFrame:
    try:
        if ext == ".csv":
            return pd.read_csv(path, nrows=chunk_size)
        return _read_excel(path).head(chunk_size)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File is empty: {path.name}") from exc


def _iter_chunks(path: Path, ext: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    if ext == ".csv":
        reader = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size)
        for chunk in reader:
            yield _normalize(chunk)
        return
    frame = _normalize(_read_excel(path, dtype=str, keep_default_na=False))
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size]


# ------------------------------------------------------------------- public API
def profile_file(
    path: str | Path, *, chunk_size: int = 50_000, max_sample_rows: int = 10
) -> FileProfile:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if max_sample_rows < 0:
        raise ValueError("max_sample_rows must be >= 0")
    path = Path(path)
    ext = _extension(path)
    head = _read_head(path, ext, chunk_size)
    if len(head.columns) == 0:
        raise ValueError(f"No columns found in {path.name}")
    names = [str(c) for c in head.columns]
    accs = [_ColumnAccumulator(n, str(t)) for n, t in zip(names, head.dtypes)]
    selector = _SampleSelector(max_sample_rows)
    row_count = 0
    for chunk_no, chunk in enumerate(_iter_chunks(path, ext, chunk_size)):
        for values in chunk.values.tolist():
            patterns = [cell_pattern(v) for v in values]
            for acc, value, pattern in zip(accs, values, patterns):
                acc.add(value, pattern)
            selector.offer(row_count, dict(zip(names, values)), patterns, chunk_no == 0)
            row_count += 1
    return FileProfile(
        path_name=path.name,
        row_count=row_count,
        columns=tuple(a.build() for a in accs),
        sample_rows=selector.result(),
    )


def _column_to_dict(c: ColumnProfile) -> dict[str, Any]:
    return {
        "name": c.name,
        "pandas_dtype": c.pandas_dtype,
        "inferred_kind": c.inferred_kind,
        "non_missing": c.non_missing,
        "missing": c.missing,
        "missing_tokens": [[t, n] for t, n in c.missing_tokens],
        "distinct": c.distinct,
        "distinct_capped": c.distinct_capped,
        "top_values": [[v, n] for v, n in c.top_values],
        "patterns": [
            {"pattern": p.pattern, "count": p.count, "example": p.example}
            for p in c.patterns
        ],
        "min_length": c.min_length,
        "max_length": c.max_length,
        "numeric_fraction": c.numeric_fraction,
    }


def profile_to_dict(profile: FileProfile) -> dict[str, Any]:
    """JSON-serializable form with stable key order."""
    return {
        "path_name": profile.path_name,
        "row_count": profile.row_count,
        "columns": [_column_to_dict(c) for c in profile.columns],
        "sample_rows": [dict(r) for r in profile.sample_rows],
    }

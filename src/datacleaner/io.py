"""Read and write tabular files (CSV and XLSX)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")
ISO_DATE = "%Y-%m-%d"
ISO_DATETIME = "%Y-%m-%dT%H:%M:%S"


def _extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext or path.name}'; supported: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return ext


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = _extension(path)
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def _iso_text(series: pd.Series) -> pd.Series:
    """Render datetimes as ISO text: date-only if every value is midnight."""
    present = series.dropna()
    date_only = bool((present == present.dt.normalize()).all())
    return series.dt.strftime(ISO_DATE if date_only else ISO_DATETIME)


def _csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Write datetime columns deterministically, independent of pandas defaults."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = _iso_text(out[column])
    return out


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ext = _extension(path)
    if ext == ".csv":
        _csv_ready(df).to_csv(path, index=False)
    else:
        df.to_excel(path, index=False, engine="openpyxl")

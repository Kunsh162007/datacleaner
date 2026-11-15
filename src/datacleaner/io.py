"""Read and write tabular files (CSV and XLSX)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


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


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ext = _extension(path)
    if ext == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False, engine="openpyxl")

"""Generate the benchmark datasets: `python -m bench.generate [--suite tuning|holdout|holdout2]`.

All suites by default: tuning in bench/data/, held-out in bench/holdout/ and bench/holdout2/. Per dataset, in
<suite dir>/<name>/: raw.csv (messy), expected.csv (ground truth
after ideal cleaning), spec.json (column kinds) and reference_plan.json.
Deterministic: fixed seeds, stdlib random, LF line endings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bench.datasets import DATASET_NAMES, Dataset, build
from bench.holdout2_datasets import HOLDOUT2_NAMES, build_holdout2
from bench.holdout_datasets import HOLDOUT_NAMES, build_holdout
from datacleaner.io import read_table, write_table

DATA_DIR = Path(__file__).parent / "data"
HOLDOUT_DIR = Path(__file__).parent / "holdout"
HOLDOUT2_DIR = Path(__file__).parent / "holdout2"
SUITES = {"tuning": (DATASET_NAMES, DATA_DIR), "holdout": (HOLDOUT_NAMES, HOLDOUT_DIR),
          "holdout2": (HOLDOUT2_NAMES, HOLDOUT2_DIR)}
PLAN_VERSION = 1
REFERENCE_MODEL = "reference/hand-written"
REFERENCE_PROMPT_VERSION = "n/a"


def _typed_column(values: list[Any], kind: str) -> pd.Series:
    if kind == "int":
        return pd.Series(pd.array(values, dtype="Int64"))
    if kind == "float":
        return pd.Series(values, dtype="float64")
    if kind == "bool":
        return pd.Series(pd.array(values, dtype="boolean"))
    if kind == "datetime":
        return pd.to_datetime(pd.Series(values, dtype="object"), format="ISO8601")
    return pd.Series(values, dtype="object")


def expected_frame(data: Dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _typed_column([row[column] for row in data.clean_rows], kind)
            for column, kind in data.kinds.items()
        }
    )


def reference_plan(data: Dataset, raw_path: Path) -> dict[str, Any]:
    raw = read_table(raw_path)
    return {
        "plan_version": PLAN_VERSION,
        "generated_by": {"model": REFERENCE_MODEL, "prompt_version": REFERENCE_PROMPT_VERSION},
        "source_columns": [{"name": c, "dtype": str(raw[c].dtype)} for c in raw.columns],
        "actions": [
            {"id": f"a{i}", "column": column, "action": action, "params": params,
             "reason": reason}
            for i, (column, action, params, reason) in enumerate(data.steps, start=1)
        ],
    }


def _write_raw(data: Dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data.kinds), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data.raw_rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _normalise_newlines(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def generate_dataset(name: str, out_dir: Path | None = None) -> Path:
    if name in HOLDOUT2_NAMES:
        data, default_dir = build_holdout2(name), HOLDOUT2_DIR
    elif name in HOLDOUT_NAMES:
        data, default_dir = build_holdout(name), HOLDOUT_DIR
    else:
        data, default_dir = build(name), DATA_DIR
    out_dir = out_dir or default_dir
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    _write_raw(data, target / "raw.csv")
    write_table(expected_frame(data), target / "expected.csv")
    _normalise_newlines(target / "expected.csv")
    _write_json(target / "spec.json", {"name": name, "columns": data.kinds,
                                       "injection": data.injection})
    _write_json(target / "reference_plan.json", reference_plan(data, target / "raw.csv"))
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.generate")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    args = parser.parse_args(argv)
    for suite, (names, folder) in SUITES.items():
        if args.suite in (suite, "all"):
            for name in names:
                generate_dataset(name, folder)
                print(f"generated bench/{folder.name}/{name}")


if __name__ == "__main__":
    main()

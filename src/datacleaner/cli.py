"""Command-line interface: `datacleaner apply` and `datacleaner suggest`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from datacleaner.engine import DEFAULT_MAX_COERCE
from datacleaner.io import read_table, write_table
from datacleaner.pipeline import ApplyError, apply_plan
from datacleaner.llm import LLMClient, LLMError
from datacleaner.providers import AUTO_ORDER, AUTO_PROVIDER, DEFAULT_PROVIDER, PROVIDERS, client_for
from datacleaner.suggest import SuggestError, review_interactively, suggest_plan
from datacleaner.validator import validate_plan

EXIT_OK = 0
EXIT_FAILURE = 1
DEFAULT_UI_PORT = 8501


class CliError(Exception):
    """A user-facing failure; message is printed to stderr."""


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datacleaner")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_p = sub.add_parser("apply", help="apply a validated cleaning plan to a file")
    apply_p.add_argument("--in", dest="input", required=True, type=Path)
    apply_p.add_argument("--plan", required=True, type=Path)
    apply_p.add_argument("--out-dir", required=True, type=Path)
    apply_p.add_argument("--max-coerce", type=_fraction, default=DEFAULT_MAX_COERCE)
    suggest_p = sub.add_parser("suggest", help="ask an LLM to draft a cleaning plan")
    suggest_p.add_argument("--in", dest="input", required=True, type=Path)
    suggest_p.add_argument("--out", type=Path, default=Path("cleaning_plan.json"))
    suggest_p.add_argument("--provider", choices=[*sorted(PROVIDERS), AUTO_PROVIDER],
                           default=DEFAULT_PROVIDER)
    suggest_p.add_argument("--model", default=None, help="default: the provider's own default")
    suggest_p.add_argument("--interactive", action="store_true")
    suggest_p.add_argument("--force", action="store_true")
    ui_p = sub.add_parser("ui", help="open the local web app in your browser")
    ui_p.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    return parser


def _launch_ui(args: argparse.Namespace) -> None:
    from datacleaner.ui.helpers import APP_PATH

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
           "--server.headless", "false", "--server.port", str(args.port)]
    try:
        completed = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        return
    if completed.returncode:
        raise CliError(f"web app exited with code {completed.returncode}")


def _load_plan(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read plan '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"plan '{path}' is not valid JSON: {exc}") from exc


def _read_input(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except FileNotFoundError as exc:
        raise CliError(f"input file not found: {path}") from exc
    except (ValueError, OSError) as exc:
        raise CliError(f"cannot read input '{path}': {exc}") from exc


def _write_all(out_dir: Path, outputs: dict[str, Callable[[Path], None]]) -> None:
    """Write every output to a temp name, then rename them all into place."""
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".datacleaner-", dir=out_dir))
    try:
        for name, writer in outputs.items():
            writer(staging / name)
        for name in outputs:
            os.replace(staging / name, out_dir / name)
    finally:
        for leftover in staging.glob("*"):
            leftover.unlink(missing_ok=True)
        staging.rmdir()


def _text_writer(text: str) -> Callable[[Path], None]:
    return lambda path: path.write_text(text, encoding="utf-8")


def _data_notice(provider: str) -> str:
    """Say where file data goes before anything is sent."""
    targets = " -> ".join(AUTO_ORDER) + " (fallback order)" if provider == AUTO_PROVIDER else provider
    return f"datacleaner: sending column profiles and sample rows to: {targets}"


def _apply(args: argparse.Namespace) -> None:
    df = _read_input(args.input)
    plan = _load_plan(args.plan)
    try:
        outcome = apply_plan(
            df, plan, input_path=args.input, plan_path=args.plan, max_coerce=args.max_coerce
        )
    except ApplyError as exc:
        raise CliError(str(exc)) from exc
    cleaned_name = f"cleaned_{args.input.stem}{args.input.suffix.lower()}"
    _write_all(
        args.out_dir,
        {
            cleaned_name: lambda path: write_table(outcome.cleaned, path),
            "audit_report.md": _text_writer(outcome.audit_markdown),
            "audit_report.json": _text_writer(json.dumps(outcome.audit, indent=2)),
            "diff_report.txt": _text_writer(outcome.diff.report_text),
        },
    )
    print(f"datacleaner: wrote {cleaned_name} and reports to {args.out_dir}")


def _write_plan(path: Path, plan: Any) -> None:
    text = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".plan-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _suggest(
    args: argparse.Namespace,
    client_factory: Callable[[str, str | None], LLMClient],
    ask: Callable[[str], str],
) -> None:
    if args.out.exists() and not args.force:
        raise CliError(f"'{args.out}' already exists; use --force to overwrite")
    try:
        client = client_factory(args.provider, args.model)
        print(_data_notice(args.provider), file=sys.stderr)
        result = suggest_plan(args.input, client)
    except (LLMError, SuggestError) as exc:
        raise CliError(str(exc)) from exc
    plan, needs_approval = result.plan, result.validation.needs_approval
    if args.interactive:
        plan = review_interactively(plan, needs_approval, ask, print)
        columns = [c["name"] for c in plan["source_columns"]]
        revalidated = validate_plan(plan, columns)
        if not revalidated.is_valid:
            raise CliError("reviewed plan is invalid:\n" + "\n".join(revalidated.errors))
        needs_approval = revalidated.needs_approval
    _write_plan(args.out, plan)
    review = ", ".join(needs_approval) or "none"
    print(f"datacleaner: {len(plan['actions'])} actions; needs review (destructive): {review}")
    print(f"datacleaner: wrote plan to {args.out}")


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[str, str | None], LLMClient] = client_for,
    ask: Callable[[str], str] = input,
) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_FAILURE
    try:
        if args.command == "ui":
            _launch_ui(args)
        elif args.command == "suggest":
            _suggest(args, client_factory, ask)
        else:
            _apply(args)
    except (CliError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


def run() -> None:
    sys.exit(main())

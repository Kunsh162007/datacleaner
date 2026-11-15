"""Run the benchmark: `python -m bench.run --provider groq [--model M]`.

Calls suggest_plan with a real client per dataset, saves each plan, scores it
and writes a Markdown summary. `--suite holdout` uses the held-out datasets
(bench/holdout, results in bench/results/holdout); `--repeat N` runs each
dataset N times and reports mean and min. `--compare` aggregates all runs into
<results>/leaderboard.md. Rate limits are retried once (429 after 60 s,
503 after 20 s); there is never a loop. `--delay` pauses between datasets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bench.generate import SUITES
from bench.score import Score, score_plan
from datacleaner.llm import LLMClient, LLMError
from datacleaner.providers import AUTO_ORDER, PROVIDERS, client_for
from datacleaner.suggest import SuggestError, suggest_plan

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_SUITE = "tuning"
HOLDOUT_RESULTS_DIR = RESULTS_DIR / "holdout"
HOLDOUT2_RESULTS_DIR = RESULTS_DIR / "holdout2"
DATA_DIR = SUITES[DEFAULT_SUITE][1]
RETRY_WAIT_S = 20
RATE_LIMIT_WAIT_S = 60
DEFAULT_DELAY_S = 5.0
RETRYABLE = re.compile(r"\b(429|503)\b")
RATE_LIMITED = re.compile(r"\b429\b")
SUMMARY_JSON = "summary.json"
PERCENT = 100


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def run_dir_name(provider: str, model: str) -> str:
    return f"{provider}__{sanitize(model)}"


def _plan_with_retry(
    raw: Path, client: LLMClient, sleep: Callable[[float], None]
) -> tuple[Any, int]:
    try:
        result = suggest_plan(raw, client)
    except LLMError as exc:
        if not RETRYABLE.search(str(exc)):
            raise
        sleep(RATE_LIMIT_WAIT_S if RATE_LIMITED.search(str(exc)) else RETRY_WAIT_S)
        result = suggest_plan(raw, client)
    return result.plan, result.attempts


def run_dataset(
    name: str,
    client: LLMClient,
    out_dir: Path,
    *,
    data_dir: Path = DATA_DIR,
    sleep: Callable[[float], None] = time.sleep,
    tag: str = "",
) -> Score:
    source = data_dir / name
    started = time.monotonic()
    try:
        plan, attempts = _plan_with_retry(source / "raw.csv", client, sleep)
    except (LLMError, SuggestError) as exc:
        latency = time.monotonic() - started
        return Score(False, f"no plan: {str(exc)[:200]}", 0.0, latency_s=latency,
                     has_plan=False)
    latency = time.monotonic() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}{tag}.plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return score_plan(source / "raw.csv", plan, source / "expected.csv",
                      attempts=attempts, latency_s=latency)


def _weak_columns(score: Score) -> str:
    weak = [f"{c} {v:.2f}" for c, v in score.column_scores.items() if v < 1]
    return ", ".join(weak) or "-"


def _row(name: str, score: Score) -> str:
    latency = "-" if score.latency_s is None else f"{score.latency_s:.1f}s"
    retries = "-" if score.retries is None else str(score.retries)
    resisted = {None: "-", True: "yes", False: "NO"}[score.injection_resisted]
    detail = score.reason or _weak_columns(score)
    return (f"| {name} | {'yes' if score.valid else 'NO'} | {score.overall * PERCENT:.1f}% | "
            f"{retries} | {score.destructive} | {resisted} | {latency} | {detail} |")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _agg_row(name: str, runs: list[Score]) -> str:
    n = len(runs)
    overall = [r.overall for r in runs]
    valid = sum(r.valid for r in runs)
    resisted = sum(bool(r.injection_resisted) for r in runs)
    latencies = [r.latency_s for r in runs if r.latency_s is not None]
    latency = f"{_mean(latencies):.1f}s" if latencies else "-"
    worst = min(runs, key=lambda r: r.overall)
    detail = worst.reason or _weak_columns(worst)
    return (f"| {name} | {valid}/{n} | {_mean(overall) * PERCENT:.1f}% | "
            f"{min(overall) * PERCENT:.1f}% | {sum(r.destructive for r in runs)} | "
            f"{resisted}/{n} | {latency} | {detail} |")


def dataset_means(runs: dict[str, list[Score]]) -> dict[str, float]:
    return {n: _mean([r.overall for r in rs]) for n, rs in runs.items()}


def render_summary(title: str, runs: dict[str, list[Score]]) -> str:
    repeated = any(len(rs) > 1 for rs in runs.values())
    mean = _mean(list(dataset_means(runs).values()))
    if repeated:
        header = ["| dataset | valid | mean score | min score | destructive | "
                  "injection resisted | latency | notes (worst run) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        rows = [_agg_row(name, rs) for name, rs in runs.items()]
    else:
        header = ["| dataset | valid | score | retries | destructive | injection resisted | "
                  "latency | notes |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        rows = [_row(name, rs[0]) for name, rs in runs.items()]
    return "\n".join([f"# {title}", "", *header, *rows, "",
                      f"Mean score: {mean * PERCENT:.1f}%", ""])


def _summary_payload(title: str, runs: dict[str, list[Score]]) -> dict[str, Any]:
    flat = [s for rs in runs.values() for s in rs]
    return {"run": title, "mean": _mean(list(dataset_means(runs).values())),
            "datasets": dataset_means(runs),
            "datasets_min": {n: min(r.overall for r in rs) for n, rs in runs.items()},
            "repeat": max((len(rs) for rs in runs.values()), default=1),
            "destructive": sum(s.destructive for s in flat),
            "injection_resisted": {n: (rs[0].injection_resisted if len(rs) == 1 else
                                       all(bool(r.injection_resisted) for r in rs))
                                   for n, rs in runs.items()}}


def write_summary(out_dir: Path, title: str, runs: dict[str, list[Score]]) -> str:
    text = render_summary(title, runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(_summary_payload(title, runs), indent=2) + "\n", encoding="utf-8")
    return text


def write_leaderboard(results_dir: Path) -> str:
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(results_dir.glob(f"*/{SUMMARY_JSON}"))]
    runs.sort(key=lambda r: (-r["mean"], r["run"]))
    names = sorted({n for r in runs for n in r["datasets"]})
    lines = ["# Leaderboard", "",
             "| run | mean | " + " | ".join(names) + " | destructive |",
             "| --- | --- | " + " | ".join("---" for _ in names) + " | --- |"]
    for r in runs:
        cells = " | ".join(f"{r['datasets'].get(n, 0) * PERCENT:.1f}%" for n in names)
        lines.append(f"| {r['run']} | {r['mean'] * PERCENT:.1f}% | {cells} | {r['destructive']} |")
    text = "\n".join(lines) + "\n"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "leaderboard.md").write_text(text, encoding="utf-8")
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.run")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=AUTO_ORDER[0])  # one concrete provider per run
    parser.add_argument("--model", default=None)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S,
                        help="seconds to sleep between datasets (default 5)")
    parser.add_argument("--suite", choices=sorted(SUITES), default=DEFAULT_SUITE,
                        help="tuning (bench/data), holdout (bench/holdout) or holdout2 (bench/holdout2)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per dataset; summaries show mean and min (default 1)")
    parser.add_argument("--compare", action="store_true",
                        help="only aggregate existing results into leaderboard.md")
    return parser


def _selected(names: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    if "all" in names:
        return tuple(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return tuple(names)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[str, str | None], LLMClient] = client_for,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    names_in_suite, suite_data = SUITES[args.suite]
    data_dir = data_dir or suite_data
    suite_results = {"holdout": HOLDOUT_RESULTS_DIR, "holdout2": HOLDOUT2_RESULTS_DIR}
    results_dir = results_dir or suite_results.get(args.suite, RESULTS_DIR)
    if args.compare:
        print(write_leaderboard(results_dir))
        return 0
    try:
        if args.repeat < 1:
            raise ValueError("--repeat must be at least 1")
        names = _selected(args.datasets, names_in_suite)
        client = client_factory(args.provider, args.model)
    except (ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    model = args.model or PROVIDERS[args.provider].default_model
    title = run_dir_name(args.provider, model)
    runs: dict[str, list[Score]] = {name: [] for name in names}
    first = True
    for name in names:
        for repeat in range(args.repeat):
            if not first:
                sleep(args.delay)
            first = False
            tag = f".r{repeat + 1}" if args.repeat > 1 else ""
            runs[name].append(run_dataset(name, client, results_dir / title,
                                          data_dir=data_dir, sleep=sleep, tag=tag))
    print(write_summary(results_dir / title, title, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

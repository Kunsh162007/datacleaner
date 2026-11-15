# datacleaner

LLM-assisted, **validated**, **deterministic** cleaning for CSV/Excel files.

An LLM (Google Gemini) *suggests* a cleaning plan. A human reviews and commits it.
CI then *applies* that plan with plain Pandas — no LLM, same input → byte-identical output —
and publishes the cleaned file plus a diff and an audit log.

```
            ┌──────────── suggest (local, uses Gemini) ────────────┐
messy.csv → profile → prompt → Gemini → JSON Schema + semantic checks → cleaning_plan.json
                                                                              │  review + commit
            ┌──────────── apply (CI, no LLM) ──────────────────────┐          ▼
messy.csv + cleaning_plan.json → validate → whitelisted actions → cleaned file + diff + audit
```

## Quick start

```bash
git clone https://github.com/Kunsh162007/datacleaner.git
cd datacleaner
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install .
datacleaner ui                 # opens the web app in your browser
```

The app walks through: choose a file (a sample is built in, or upload CSV/XLSX), profile it,
ask an LLM for a cleaning plan, apply it deterministically, and review changed cells plus an audit.

**API keys.** Suggesting a plan needs a free key from one provider:
[NVIDIA](https://build.nvidia.com) (default), [Google AI Studio](https://aistudio.google.com/apikey)
(Gemini) or the [Groq console](https://console.groq.com). Paste it into the app sidebar (used only
in that browser session, never saved), or copy `.env.example` to `.env` and fill in the matching
variable. **No key?** Click "Use saved demo plan" to try the whole flow offline.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

### 1. Suggest a plan (needs a Gemini key)

Put the key in a `.env` file in the project root (already git-ignored):

```
GEMINI_API_KEY=your-key-here
```

Other providers use `GROQ_API_KEY` / `NVIDIA_API_KEY` the same way (non-Gemini plans record `generated_by.model` as `<provider>/<model>`).

`--provider auto` (opt-in) tries nvidia (MiniMax), then gemini, then groq — ordered by benchmark results (each with its default model; providers without an API key are skipped; `--model` is not allowed). It moves to the next provider only on availability errors (HTTP 429/500/502/503/504, timeouts, connection errors, 404/410 model unavailable) and prints one line to stderr per fallback, e.g. `gemini unavailable (503), trying nvidia`. Validation failures are retried by `suggest` itself, and 400/401/403 (configuration errors) are raised, never skipped. `generated_by.model` records the provider/model that actually answered. **Privacy:** every attempt sends column profiles and sample rows to that provider, so one `auto` run can share data with up to three companies. The default is a single provider (`nvidia`); use `auto` only for data you are comfortable sharing that widely. `suggest` prints where data is sent before sending.

A key in `.env` takes precedence over a `GEMINI_API_KEY` set in your shell, so a machine-wide key
for another project is never used by accident. Without `.env`, the shell variable is used.

```bash
datacleaner suggest --in data.csv --out cleaning_plan.json
```

| Flag | Meaning |
|------|---------|
| `--in` | CSV or XLSX input (required) |
| `--out` | plan path (default `cleaning_plan.json`) |
| `--provider` | `nvidia` (default, MiniMax), `gemini`, `groq`, or `auto` (opt-in fallback chain, see below) |
| `--model` | model ID (default per provider: `gemini-2.5-flash`, `openai/gpt-oss-120b`, `minimaxai/minimax-m3.1-flash`) |
| `--interactive` | approve each action; rejecting one also drops later actions on that column |
| `--force` | overwrite an existing plan file |

The LLM only sees column profiles and a few representative sample rows, never the full file.
Its answer is validated; on failure it gets one retry with the validation errors.

### 2. Apply the plan (deterministic, no LLM)

```bash
datacleaner apply --in data.csv --plan cleaning_plan.json --out-dir cleaned
```

| Flag | Meaning |
|------|---------|
| `--max-coerce` | max fraction of values a `cast` may turn into NaN before the run aborts (default `0.05`) |

Outputs in `--out-dir` (written only if every step succeeds — nothing is written on failure):

- `cleaned_<name>.<ext>` — the cleaned data
- `diff_report.txt` — tabular diff (datacompy), changed cells sorted by row, first 20 shown
- `audit_report.md` / `audit_report.json` — every step, its stats, and any destructive actions

Exit code `0` on success, `1` on any validation or execution error.

## The plan file

```json
{
  "plan_version": 1,
  "generated_by": {"model": "gemini-2.5-flash", "prompt_version": "3"},
  "source_columns": [{"name": "price", "dtype": "object"}],
  "actions": [
    {"id": "a1", "column": "price", "action": "regex_replace",
     "params": {"pattern": "[$,]", "replacement": ""}, "reason": "strip currency symbols"},
    {"id": "a2", "column": "price", "action": "cast", "params": {"to": "float"},
     "reason": "numeric column stored as text"}
  ]
}
```

Actions run in order, as one queue:

| Action | Params | Notes |
|--------|--------|-------|
| `cast` | `to`: `int` / `float` / `str` / `bool` | bad values → NaN (threshold checked); `int` → nullable `Int64` |
| `fillna` | `strategy`: `mean` / `median` / `mode` / `constant` (+ `value`) | mean/median numeric only; mode ties → smallest |
| `regex_replace` | `pattern`, `replacement` | RE2 syntax (no catastrophic backtracking); text values only |
| `drop` | none | destructive — always flagged in the audit |
| `strip` | `case`: `none` / `lower` / `upper` / `title` | trims, collapses inner whitespace to one space, then applies case; text values only |
| `datetime_parse` | `format`: strftime string (`%Y %m %d %H %M %S %b %B %y %j`, separators `- / space . : T ,`) | failures become NaT and count toward `--max-coerce`; CSV output is ISO (`YYYY-MM-DD`, or `YYYY-MM-DDTHH:MM:SS` if any time is present) |
| `clip` | `min` and/or `max` (min <= max) | numeric columns only |

## How LLM output is guarded

1. **Structured output** — Gemini is called with a JSON response schema and `temperature=0`.
2. **JSON Schema** (`plan.schema.json`) — strict types, `additionalProperties: false`, per-action params.
3. **Semantic checks** — unique IDs, plan columns match the file, each column exists at its point in the queue, regexes compile under RE2.
4. **Whitelisted dispatch** — action names map to pre-written functions; nothing from the plan is ever `eval`'d.
5. **Destructive flag is computed by code**, never trusted from the plan.
6. **Prompt-injection hygiene** — cell values are marked as untrusted data in the system prompt; prompt size is capped.

## CI (GitHub Action)

`.github/workflows/data-hygiene.yml` runs on push / PR / manual dispatch:

1. `test` — pytest with coverage.
2. `data-hygiene` — `datacleaner apply` on `DATA_FILE` with `PLAN_FILE`, writes the audit into the job summary,
   and uploads the `cleaned-data` artifact. The job fails if the plan doesn't validate or a step fails.

Change `DATA_FILE`, `PLAN_FILE` and `MAX_COERCE` in the workflow's `env:` block. CI never needs an API key.

## Try it

```bash
datacleaner apply --in examples/messy.csv --plan examples/cleaning_plan.json --out-dir cleaned
```

## Benchmark

- `python -m bench.generate` writes three seeded messy datasets to `bench/data/` (raw, ground truth,
  column kinds, hand-written reference plan).
- `python -m bench.run --provider groq [--model M] [--delay 5]` asks the LLM for a plan per dataset,
  scores it with the real engine and writes `bench/results/<provider>__<model>/summary.md`. `--delay`
  is the pause between datasets; a 429 is retried once after 60 s, a 503 after 20 s.
- `python -m bench.run --compare` builds `bench/results/leaderboard.md`.

### Tuning vs held-out suites

- **tuning** (`bench/data/`: sales, patients, products) is the suite the prompt was iterated on.
- **holdout** (`bench/holdout/`: employees, sensors) uses different domains, column names and
  messiness (euro amounts with an `EUR` suffix, unit suffixes like `21.4 °C`, dotted dates,
  timestamps with a time, percent values with sensor glitches above 100). The prompt was never
  tuned on it, so it measures overfitting.
- **Do not edit the prompt based on holdout results.** Doing so turns the holdout into a second
  tuning set and it stops being held-out. Use it only to compare tuning vs holdout scores; if a
  new suite is needed, add fresh datasets.
- `python -m bench.generate [--suite tuning|holdout|holdout2]` regenerates all by default (deterministic).
- **holdout2** (`bench/holdout2/`: invoices, fitness) is the fresh, never-tuned set, created after prompt v5 (whole-number hint) with new domains and seeds. Run it with `--suite holdout2` (results in `bench/results/holdout2/`); the same rule applies: do not tune the prompt on it.
- `python -m bench.run --suite holdout ...` writes to `bench/results/holdout/<run>/`;
  `python -m bench.run --suite holdout --compare` builds `bench/results/holdout/leaderboard.md`.
- `--repeat N` runs each dataset N times (plans saved as `<name>.rK.plan.json`); the summary then
  shows mean and min score per dataset so run-to-run variance is visible. Default is 1.

Missing-value policy (what the ground truth and the prompt both assume): numeric measurement, count
and score columns are filled with the median (integer columns: median rounded half-to-even, so they
stay integer); money columns (price, amount, income, ...), ids, dates, booleans and text stay missing.
Text-normalised columns are scored case-insensitively.

Missing tokens used are only those pandas reads as NaN (`""`, `N/A`, `null`, `NaN`).

## Development

```bash
pytest --cov=datacleaner
```

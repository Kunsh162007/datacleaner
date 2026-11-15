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

A key in `.env` takes precedence over a `GEMINI_API_KEY` set in your shell, so a machine-wide key
for another project is never used by accident. Without `.env`, the shell variable is used.

```bash
datacleaner suggest --in data.csv --out cleaning_plan.json
```

| Flag | Meaning |
|------|---------|
| `--in` | CSV or XLSX input (required) |
| `--out` | plan path (default `cleaning_plan.json`) |
| `--model` | Gemini model ID (default `gemini-2.5-flash`, pinned for reproducibility) |
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
  "generated_by": {"model": "gemini-2.5-flash", "prompt_version": "2"},
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

## Development

```bash
pytest --cov=datacleaner
```

"""End-to-end tests for the `apply` command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datacleaner.cli import main

COLUMNS = ["price", "age", "notes"]


def make_plan(actions: list[dict], columns: list[str] = COLUMNS) -> dict:
    return {
        "plan_version": 1,
        "generated_by": {"model": "test", "prompt_version": "1"},
        "source_columns": [{"name": c, "dtype": "object"} for c in columns],
        "actions": actions,
    }


def act(id_: str, column: str, name: str, params: dict | None = None) -> dict:
    return {"id": id_, "column": column, "action": name, "params": params or {}, "reason": "t"}


GOOD_ACTIONS = [
    act("a1", "price", "regex_replace", {"pattern": r"\$", "replacement": ""}),
    act("a2", "price", "cast", {"to": "float"}),
    act("a3", "age", "fillna", {"strategy": "median"}),
    act("a4", "notes", "drop"),
]


def setup(tmp_path: Path, actions: list[dict], ext: str = "csv") -> tuple[Path, Path, Path]:
    df = pd.DataFrame(
        {
            "price": ["$12.50", "$3.00", "N/A", "$7.25"] * 5,
            "age": [20.0, None, 40.0, 30.0] * 5,
            "notes": ["a", "b", "c", "d"] * 5,
        }
    )
    src = tmp_path / f"data.{ext}"
    if ext == "csv":
        df.to_csv(src, index=False)
    else:
        df.to_excel(src, index=False, engine="openpyxl")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(make_plan(actions)), encoding="utf-8")
    return src, plan, tmp_path / "out"


def run(src: Path, plan: Path, out: Path, *extra: str) -> int:
    return main(["apply", "--in", str(src), "--plan", str(plan), "--out-dir", str(out), *extra])


def test_apply_end_to_end(tmp_path, capsys) -> None:
    src, plan, out = setup(tmp_path, GOOD_ACTIONS)
    assert run(src, plan, out, "--max-coerce", "0.5") == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["audit_report.json", "audit_report.md", "cleaned_data.csv", "diff_report.txt"]
    cleaned = pd.read_csv(out / "cleaned_data.csv")
    assert list(cleaned.columns) == ["price", "age"]
    assert cleaned["price"].iloc[0] == 12.5 and pd.isna(cleaned["price"].iloc[2])
    assert cleaned["age"].iloc[1] == 30.0
    audit = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))
    assert audit["destructive_actions_run"] == ["a4"]
    assert "__datacleaner_row_id__" not in (out / "cleaned_data.csv").read_text(encoding="utf-8")
    assert "cleaned_data.csv" in capsys.readouterr().out


def test_invalid_plan_writes_nothing(tmp_path, capsys) -> None:
    src, plan, out = setup(tmp_path, [act("a1", "ghost", "drop")])
    assert run(src, plan, out) == 1
    assert not out.exists()
    assert "ghost" in capsys.readouterr().err


def test_coercion_over_threshold_writes_nothing(tmp_path, capsys) -> None:
    src, plan, out = setup(tmp_path, [act("a1", "price", "cast", {"to": "float"})])
    assert run(src, plan, out) != 0
    assert not out.exists()
    assert "a1" in capsys.readouterr().err


def test_missing_input_and_bad_json(tmp_path) -> None:
    src, plan, out = setup(tmp_path, GOOD_ACTIONS)
    assert run(tmp_path / "nope.csv", plan, out) != 0
    plan.write_text("{not json", encoding="utf-8")
    assert run(src, plan, out) != 0
    assert not out.exists()


def test_unsupported_format(tmp_path) -> None:
    _, plan, out = setup(tmp_path, GOOD_ACTIONS)
    odd = tmp_path / "data.txt"
    odd.write_text("x", encoding="utf-8")
    assert run(odd, plan, out) != 0
    assert not out.exists()


def test_max_coerce_out_of_range(tmp_path) -> None:
    src, plan, out = setup(tmp_path, GOOD_ACTIONS)
    assert run(src, plan, out, "--max-coerce", "1.5") != 0
    assert not out.exists()


def test_excel_round_trip(tmp_path) -> None:
    src, plan, out = setup(tmp_path, GOOD_ACTIONS, ext="xlsx")
    assert run(src, plan, out, "--max-coerce", "0.5") == 0
    cleaned = pd.read_excel(out / "cleaned_data.xlsx", engine="openpyxl")
    assert list(cleaned.columns) == ["price", "age"]
    assert cleaned["price"].iloc[1] == 3.0


def test_row_id_collision_refused(tmp_path) -> None:
    df = pd.DataFrame({"__datacleaner_row_id__": [1], "price": [1]})
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)
    plan = tmp_path / "p.json"
    plan.write_text(
        json.dumps(make_plan([], ["__datacleaner_row_id__", "price"])), encoding="utf-8"
    )
    assert run(src, plan, tmp_path / "out") != 0
    assert not (tmp_path / "out").exists()


def test_outputs_are_byte_identical_across_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame(
        {
            "price": [f"${i}.50" if i % 7 else "N/A" for i in range(30)],
            "age": [float(i) if i % 3 else None for i in range(30)],
            "notes": [f"n{i}" for i in range(30)],
        }
    )
    df.to_csv("data.csv", index=False)
    Path("plan.json").write_text(json.dumps(make_plan(GOOD_ACTIONS)), encoding="utf-8")
    outs = [Path("out1"), Path("out2")]
    for out in outs:
        assert run(Path("data.csv"), Path("plan.json"), out, "--max-coerce", "0.5") == 0
    names = sorted(p.name for p in outs[0].iterdir())
    assert names == sorted(p.name for p in outs[1].iterdir())
    assert "audit_report.json" in names
    for name in names:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    audit = json.loads((outs[0] / "audit_report.json").read_text(encoding="utf-8"))
    assert audit["input_path"] == "data.csv" and audit["plan_path"] == "plan.json"


# ------------------------------------------------------------------ suggest
import shutil  # noqa: E402

from fakes import FakeClient, action as fake_action  # noqa: E402

from datacleaner.llm import LLMError  # noqa: E402

MESSY = Path(__file__).parent.parent / "examples" / "messy.csv"
SUGGESTED = {
    "actions": [
        fake_action("a1", "price", "regex_replace", {"pattern": r"\$", "replacement": ""}),
        fake_action("a2", "price", "cast", {"to": "float"}),
        fake_action("a3", "age", "fillna", {"strategy": "median"}),
        fake_action("a4", "notes", "drop"),
    ]
}


def _suggest(tmp_path, client, *extra, ask=input):
    src = tmp_path / "messy.csv"
    if not src.exists():
        shutil.copy(MESSY, src)
    out = tmp_path / "plan.json"
    argv = ["suggest", "--in", str(src), "--out", str(out), *extra]
    code = main(argv, client_factory=lambda provider, model: client, ask=ask)
    return code, src, out


def test_suggest_writes_plan_that_apply_accepts(tmp_path, capsys) -> None:
    client = FakeClient(SUGGESTED)
    code, src, out = _suggest(tmp_path, client)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n") and json.loads(text)["generated_by"]["model"] == "fake-model"
    stdout = capsys.readouterr().out
    assert "4 actions" in stdout and "a4" in stdout and str(out) in stdout
    assert run(src, out, tmp_path / "o", "--max-coerce", "0.5") == 0


def test_suggest_refuses_existing_out_without_force(tmp_path, capsys) -> None:
    client = FakeClient(SUGGESTED)
    (tmp_path / "plan.json").write_text("old", encoding="utf-8")
    code, _, out = _suggest(tmp_path, client)
    assert code == 1 and client.calls == [] and out.read_text(encoding="utf-8") == "old"
    assert "--force" in capsys.readouterr().err
    code, _, _ = _suggest(tmp_path, client, "--force")
    assert code == 0 and out.read_text(encoding="utf-8") != "old"


def test_suggest_llm_failure_writes_nothing(tmp_path, capsys) -> None:
    code, _, out = _suggest(tmp_path, FakeClient(LLMError("boom")))
    assert code == 1 and not out.exists()
    assert "boom" in capsys.readouterr().err


def test_suggest_invalid_answers_write_nothing(tmp_path, capsys) -> None:
    bad = {"actions": [fake_action("a1", "ghost", "drop")]}
    code, _, out = _suggest(tmp_path, FakeClient(bad, bad))
    assert code == 1 and not out.exists()
    assert "ghost" in capsys.readouterr().err


def test_suggest_client_factory_failure(tmp_path, capsys) -> None:
    def factory(provider, model):
        raise LLMError("GEMINI_API_KEY is not set")

    shutil.copy(MESSY, tmp_path / "messy.csv")
    code = main(["suggest", "--in", str(tmp_path / "messy.csv"), "--out",
                 str(tmp_path / "p.json")], client_factory=factory)
    assert code == 1 and "GEMINI_API_KEY" in capsys.readouterr().err


def test_suggest_missing_input(tmp_path, capsys) -> None:
    code = main(["suggest", "--in", str(tmp_path / "nope.csv"), "--out",
                 str(tmp_path / "p.json")], client_factory=lambda p, m: FakeClient())
    assert code == 1


def test_suggest_interactive_rejection(tmp_path, capsys) -> None:
    answers = iter(["y", "n", "y", "n"])  # a1 yes, a2 no, a3 yes, a4 (drop) no
    code, _, out = _suggest(tmp_path, FakeClient(SUGGESTED), "--interactive",
                            ask=lambda _p: next(answers))
    assert code == 0
    ids = [a["id"] for a in json.loads(out.read_text(encoding="utf-8"))["actions"]]
    assert ids == ["a1", "a3"]


def test_apply_new_actions_end_to_end_with_dates(tmp_path) -> None:
    df = pd.DataFrame(
        {"city": ["  new  york", "NEW YORK"], "when": ["05/03/2024", "31/12/2023"], "n": [-5, 500]}
    )
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)
    actions = [
        act("a1", "city", "strip", {"case": "title"}),
        act("a2", "when", "datetime_parse", {"format": "%d/%m/%Y"}),
        act("a3", "n", "clip", {"min": 0, "max": 100}),
    ]
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps(make_plan(actions, ["city", "when", "n"])), encoding="utf-8")
    out = tmp_path / "out"
    assert run(src, plan, out) == 0
    cleaned = (out / "cleaned_d.csv").read_text(encoding="utf-8").splitlines()
    assert cleaned == ["city,when,n", "New York,2024-03-05,0", "New York,2023-12-31,100"]
    audit = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))
    assert audit["destructive_actions_run"] == []
    assert (out / "diff_report.txt").read_text(encoding="utf-8")

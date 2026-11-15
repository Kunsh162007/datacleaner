"""Tests for the benchmark data generator and its reference plans."""

from __future__ import annotations

import json

import pytest

from bench.datasets import DATASET_NAMES
from bench.generate import DATA_DIR, generate_dataset
from bench.noise import INJECTION_TEXT, MISSING_TOKENS
from bench.score import score_plan


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_generation_is_deterministic(tmp_path, name) -> None:
    first = generate_dataset(name, tmp_path / "a")
    second = generate_dataset(name, tmp_path / "b")
    for file in ("raw.csv", "expected.csv", "spec.json", "reference_plan.json"):
        assert (first / file).read_bytes() == (second / file).read_bytes()


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_committed_data_matches_generator(tmp_path, name) -> None:
    fresh = generate_dataset(name, tmp_path)
    for file in ("raw.csv", "expected.csv", "reference_plan.json"):
        assert (fresh / file).read_bytes() == (DATA_DIR / name / file).read_bytes()


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_reference_plan_reproduces_expected(tmp_path, name) -> None:
    folder = generate_dataset(name, tmp_path)
    plan = json.loads((folder / "reference_plan.json").read_text(encoding="utf-8"))
    score = score_plan(folder / "raw.csv", plan, folder / "expected.csv")
    assert score.valid and score.overall == 1.0 and score.destructive == 0


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_shape_and_injection(tmp_path, name) -> None:
    folder = generate_dataset(name, tmp_path)
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    text = (folder / "raw.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 150 <= len(lines) - 1 <= 300
    assert lines[0].split(",") == list(spec["columns"])
    assert text.count(INJECTION_TEXT) == 1
    assert {"id", "text-untouched"} <= set(spec["columns"].values())


def test_only_pandas_representable_missing_tokens() -> None:
    assert "-" not in MISSING_TOKENS and "" in MISSING_TOKENS


# --- held-out suite -------------------------------------------------------

from bench.generate import HOLDOUT_DIR, SUITES, main as generate_main  # noqa: E402
from bench.holdout_datasets import HOLDOUT_INJECTION, HOLDOUT_NAMES  # noqa: E402


@pytest.mark.parametrize("name", HOLDOUT_NAMES)
def test_holdout_generation_is_deterministic(tmp_path, name) -> None:
    first = generate_dataset(name, tmp_path / "a")
    second = generate_dataset(name, tmp_path / "b")
    for file in ("raw.csv", "expected.csv", "spec.json", "reference_plan.json"):
        assert (first / file).read_bytes() == (second / file).read_bytes()


@pytest.mark.parametrize("name", HOLDOUT_NAMES)
def test_committed_holdout_matches_generator(tmp_path, name) -> None:
    fresh = generate_dataset(name, tmp_path)
    for file in ("raw.csv", "expected.csv", "spec.json", "reference_plan.json"):
        assert (fresh / file).read_bytes() == (HOLDOUT_DIR / name / file).read_bytes()


@pytest.mark.parametrize("name", HOLDOUT_NAMES)
def test_holdout_reference_plan_reproduces_expected(tmp_path, name) -> None:
    folder = generate_dataset(name, tmp_path)
    plan = json.loads((folder / "reference_plan.json").read_text(encoding="utf-8"))
    score = score_plan(folder / "raw.csv", plan, folder / "expected.csv")
    assert score.valid and score.overall == 1.0 and score.destructive == 0


@pytest.mark.parametrize("name", HOLDOUT_NAMES)
def test_holdout_shape(tmp_path, name) -> None:
    folder = generate_dataset(name, tmp_path)
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    text = (folder / "raw.csv").read_text(encoding="utf-8")
    assert 150 <= len(text.splitlines()) - 1 <= 300
    assert {"id", "text-untouched"} <= set(spec["columns"].values())
    assert text.count(INJECTION_TEXT) == 0


def test_holdout_injection_only_in_employees(tmp_path) -> None:
    employees = generate_dataset("employees", tmp_path)
    sensors = generate_dataset("sensors", tmp_path)
    assert (employees / "raw.csv").read_text(encoding="utf-8").count(HOLDOUT_INJECTION) == 1
    assert HOLDOUT_INJECTION not in (sensors / "raw.csv").read_text(encoding="utf-8")


def test_holdout_sensors_have_glitches_and_time(tmp_path) -> None:
    folder = generate_dataset("sensors", tmp_path)
    raw = (folder / "raw.csv").read_text(encoding="utf-8")
    expected = (folder / "expected.csv").read_text(encoding="utf-8")
    assert any(f",{v}%," in raw for v in range(101, 106))
    assert "T" in expected.splitlines()[1] and not any(f",{v}.0," in expected for v in range(101, 106))


def test_generate_main_suites(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(SUITES, "holdout", (HOLDOUT_NAMES, tmp_path))
    generate_main(["--suite", "holdout"])
    assert (tmp_path / "employees" / "raw.csv").exists()
    assert "generated bench/" in capsys.readouterr().out


# --- fresh held-out suite (holdout2) ----------------------------------------

from bench.generate import HOLDOUT2_DIR  # noqa: E402
from bench.holdout2_datasets import HOLDOUT2_INJECTION, HOLDOUT2_NAMES  # noqa: E402

H2_FILES = ("raw.csv", "expected.csv", "spec.json", "reference_plan.json")


@pytest.mark.parametrize("name", HOLDOUT2_NAMES)
def test_holdout2_generation_is_deterministic(tmp_path, name) -> None:
    first = generate_dataset(name, tmp_path / "a")
    second = generate_dataset(name, tmp_path / "b")
    for file in H2_FILES:
        assert (first / file).read_bytes() == (second / file).read_bytes()


@pytest.mark.parametrize("name", HOLDOUT2_NAMES)
def test_committed_holdout2_matches_generator(tmp_path, name) -> None:
    fresh = generate_dataset(name, tmp_path)
    for file in H2_FILES:
        assert (fresh / file).read_bytes() == (HOLDOUT2_DIR / name / file).read_bytes()


@pytest.mark.parametrize("name", HOLDOUT2_NAMES)
def test_holdout2_reference_plan_reproduces_expected(tmp_path, name) -> None:
    from datacleaner.validator import validate_plan

    folder = generate_dataset(name, tmp_path)
    plan = json.loads((folder / "reference_plan.json").read_text(encoding="utf-8"))
    score = score_plan(folder / "raw.csv", plan, folder / "expected.csv")
    assert score.valid and score.overall == 1.0 and score.destructive == 0
    columns = list(json.loads((folder / "spec.json").read_text(encoding="utf-8"))["columns"])
    assert validate_plan(plan, columns).is_valid


@pytest.mark.parametrize("name", HOLDOUT2_NAMES)
def test_holdout2_shape(tmp_path, name) -> None:
    folder = generate_dataset(name, tmp_path)
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    text = (folder / "raw.csv").read_text(encoding="utf-8")
    assert 150 <= len(text.splitlines()) - 1 <= 300
    assert {"id", "text-untouched"} <= set(spec["columns"].values())


def test_holdout2_injection_only_in_invoices_and_whole_numbers(tmp_path) -> None:
    invoices = generate_dataset("invoices", tmp_path)
    fitness = generate_dataset("fitness", tmp_path)
    assert (invoices / "raw.csv").read_text(encoding="utf-8").count(HOLDOUT2_INJECTION) == 1
    assert HOLDOUT2_INJECTION not in (fitness / "raw.csv").read_text(encoding="utf-8")
    from datacleaner.profiler import profile_file

    for folder, column in ((invoices, "qty"), (fitness, "heart_rate")):
        profile = profile_file(folder / "raw.csv")
        assert next(c for c in profile.columns if c.name == column).whole_numbers


def test_holdout2_fitness_has_glitches(tmp_path) -> None:
    folder = generate_dataset("fitness", tmp_path)
    raw = (folder / "raw.csv").read_text(encoding="utf-8")
    assert any(f",{v}%," in raw for v in range(101, 111))


def test_generate_main_holdout2(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(SUITES, "holdout2", (HOLDOUT2_NAMES, tmp_path))
    generate_main(["--suite", "holdout2"])
    assert (tmp_path / "invoices" / "raw.csv").exists()
    assert "generated bench/" in capsys.readouterr().out

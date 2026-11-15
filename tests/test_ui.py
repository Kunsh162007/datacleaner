"""Tests for the shared apply pipeline, the UI helpers and the `ui` subcommand."""

from __future__ import annotations

import subprocess

import pytest

from datacleaner import cli
from datacleaner.io import read_table
from datacleaner.pipeline import ApplyError, apply_plan
from datacleaner.profiler import profile_file
from datacleaner.providers import client_for
from datacleaner.ui import helpers


def _outcome():
    plan = helpers.load_demo_plan()
    return plan, apply_plan(
        read_table(helpers.SAMPLE_PATH), plan, input_path="messy.csv", plan_path="p.json"
    )


def test_packaged_resources_exist() -> None:
    assert helpers.APP_PATH.is_file()
    assert helpers.SAMPLE_PATH.is_file()
    assert helpers.DEMO_PLAN_PATH.is_file()


def test_demo_plan_applies_to_sample() -> None:
    plan, outcome = _outcome()
    assert len(outcome.cleaned) == len(read_table(helpers.SAMPLE_PATH))
    assert outcome.audit["steps"]
    assert "audit" in outcome.audit_markdown.lower()
    assert len(helpers.plan_actions_table(plan)) == len(plan["actions"])
    assert not helpers.changed_cells_table(outcome).empty
    assert len(helpers.steps_table(outcome)) == len(plan["actions"])
    assert helpers.cleaned_csv_bytes(outcome).startswith(b"customer_id")


def test_apply_plan_rejects_invalid_plan() -> None:
    df = read_table(helpers.SAMPLE_PATH)
    bad = helpers.unsafe_plan(list(df.columns))
    with pytest.raises(ApplyError, match="plan is invalid"):
        apply_plan(df, bad, input_path="x", plan_path="y")


def test_unsafe_plan_is_rejected_by_validator() -> None:
    df = read_table(helpers.SAMPLE_PATH)
    result = helpers.check_unsafe_plan(list(df.columns), df.dtypes.astype(str).to_dict())
    assert not result.is_valid
    assert result.errors


def test_profile_rows_and_preview() -> None:
    rows = helpers.profile_rows(profile_file(helpers.SAMPLE_PATH))
    assert "price" in set(rows["column"])
    assert len(helpers.read_preview(helpers.SAMPLE_PATH)) <= helpers.PREVIEW_ROWS


def test_save_upload_sanitizes_and_validates(tmp_path) -> None:
    saved = helpers.save_upload("../evil/data.csv", b"a,b\n1,2\n")
    assert saved.name == "data.csv"
    assert saved.read_bytes() == b"a,b\n1,2\n"
    with pytest.raises(ValueError):
        helpers.save_upload("notes.txt", b"x")


def test_client_for_uses_explicit_key_without_env(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    client = client_for("nvidia", api_key="  session-key  ")
    assert "session-key" not in repr(client)


def test_ui_subcommand_launches_streamlit(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["ui", "--port", "9000"]) == 0
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert str(helpers.APP_PATH) in cmd
    assert cmd[cmd.index("--server.port") + 1] == "9000"
    assert cmd[cmd.index("--server.headless") + 1] == "false"


def test_ui_subcommand_reports_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 3)
    )
    assert cli.main(["ui"]) == 1
    assert "exited with code 3" in capsys.readouterr().err

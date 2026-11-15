"""Local Streamlit app: suggest (LLM) -> validate -> apply (deterministic). Run: datacleaner ui"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from datacleaner.io import read_table
from datacleaner.llm import LLMError
from datacleaner.pipeline import ApplyError, apply_plan
from datacleaner.profiler import profile_file
from datacleaner.providers import DEFAULT_PROVIDER, PROVIDERS, client_for
from datacleaner.suggest import SuggestError, suggest_plan
from datacleaner.ui import helpers

SAMPLE_LABEL = "Use the sample file (messy.csv)"
UPLOAD_LABEL = "Upload my own CSV / XLSX"

st.set_page_config(page_title="datacleaner", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
h2 {margin-top: 1.6rem; border-bottom: 2px solid #e2e8f0; padding-bottom: .3rem;}
.stButton > button {font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)
st.title("datacleaner")
st.caption("LLM-suggested, validated, deterministic data cleaning. Runs locally.")


def sidebar_settings() -> tuple[str, str]:
    """Provider and an optional session-only API key (never stored or shown)."""
    with st.sidebar:
        st.header("LLM settings")
        names = sorted(PROVIDERS)
        provider = st.selectbox("Provider", names, index=names.index(DEFAULT_PROVIDER))
        key = st.text_input(
            f"{provider} API key (optional)",
            type="password",
            help="Used only in this browser session. If empty, .env or the "
            f"{PROVIDERS[provider].env} environment variable is used.",
        )
        st.caption("No key? 'Use saved demo plan' works fully offline.")
        st.info(
            "Data privacy: the column profile and a few sample rows are sent to the chosen "
            "provider. The full file never leaves this machine. Applying a plan is local."
        )
    return provider, key


def choose_file() -> Path | None:
    st.header("1 · Choose a file")
    mode = st.radio("Source", [SAMPLE_LABEL, UPLOAD_LABEL], horizontal=True)
    if mode == SAMPLE_LABEL:
        return helpers.SAMPLE_PATH
    upload = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"])
    if upload is None:
        st.caption("Waiting for a file.")
        return None
    try:
        return helpers.save_upload(upload.name, upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return None


def show_profile(path: Path) -> None:
    st.header("2 · Profile")
    st.info("Only this profile + a few sample rows are sent to the LLM. Not the whole file.")
    st.dataframe(helpers.profile_rows(profile_file(path)), use_container_width=True,
                 hide_index=True)


def request_plan(path: Path, provider: str, api_key: str) -> None:
    started = time.time()
    try:
        with st.spinner(f"Asking {provider}..."):
            client = client_for(provider, api_key=api_key)
            result = suggest_plan(path, client)
    except (LLMError, SuggestError) as exc:
        st.error(f"Live LLM call failed: {exc}. Paste a key in the sidebar, or use the saved "
                 "demo plan.")
        return
    st.session_state["plan"] = dict(result.plan)
    st.session_state["plan_info"] = (
        f"Live answer from **{provider}** / `{client.model}` in "
        f"{time.time() - started:.0f}s, attempts: {result.attempts}"
    )
    st.session_state.pop("outcome", None)


def suggest_step(path: Path, provider: str, api_key: str) -> None:
    st.header("3 · Suggest a plan")
    live, saved = st.columns(2)
    if live.button("Suggest plan with LLM", type="primary"):
        request_plan(path, provider, api_key)
    if saved.button("Use saved demo plan"):
        st.session_state["plan"] = helpers.load_demo_plan()
        st.session_state["plan_info"] = "Loaded the saved demo plan (no LLM call, no key needed)."
        st.session_state.pop("outcome", None)
    plan = st.session_state.get("plan")
    if plan:
        st.success(st.session_state["plan_info"])
        st.caption(f"{len(plan['actions'])} actions proposed, validated before use.")
        st.dataframe(helpers.plan_actions_table(plan), use_container_width=True, hide_index=True)
        with st.expander("Plan JSON"):
            st.json(plan)


def apply_step(path: Path) -> None:
    st.header("4 · Apply (deterministic, no LLM)")
    plan = st.session_state.get("plan")
    if st.button("Apply plan", disabled=plan is None):
        try:
            st.session_state["outcome"] = apply_plan(
                read_table(path), plan, input_path=path.name, plan_path="cleaning_plan.json"
            )
        except ApplyError as exc:
            st.session_state.pop("outcome", None)
            st.error(str(exc))
    outcome = st.session_state.get("outcome")
    if outcome is None:
        return
    st.subheader("Cleaned data")
    st.dataframe(helpers.preview(outcome.cleaned), use_container_width=True, hide_index=True)
    st.subheader("Changed cells")
    cells = helpers.changed_cells_table(outcome)
    total = sum(outcome.diff.changed_cells_by_column.values())
    st.caption(f"{total} cells changed in total; showing the first {len(cells)}.")
    st.dataframe(cells, use_container_width=True, hide_index=True)
    st.subheader("Audit: steps run")
    st.dataframe(helpers.steps_table(outcome), use_container_width=True, hide_index=True)
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download cleaned CSV", helpers.cleaned_csv_bytes(outcome),
                        file_name="cleaned.csv", mime="text/csv")
    dl2.download_button("Download audit_report.md", outcome.audit_markdown,
                        file_name="audit_report.md", mime="text/markdown")


def safety_step(path: Path) -> None:
    st.header("5 · Safety")
    st.write("Plans are data, never code. Anything outside the fixed action list is rejected.")
    if st.button("Try an unsafe plan"):
        df = read_table(path)
        result = helpers.check_unsafe_plan(list(df.columns), df.dtypes.astype(str).to_dict())
        st.error("Plan REJECTED by the validator. Nothing was executed.")
        for err in result.errors:
            st.code(err, language="text")


def main() -> None:
    provider, api_key = sidebar_settings()
    chosen = choose_file()
    if chosen is None:
        return
    st.dataframe(helpers.read_preview(chosen), use_container_width=True, hide_index=True)
    show_profile(chosen)
    suggest_step(chosen, provider, api_key)
    apply_step(chosen)
    safety_step(chosen)


main()

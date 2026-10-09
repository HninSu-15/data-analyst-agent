from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import logging
from pathlib import Path
from queue import Empty, Queue
import re
import time
from uuid import uuid4

import pandas as pd
import streamlit as st
from google.adk.events import Event

from app.agent import workflow_service
from app.config import ANALYSIS_USER_ID, UPLOADS_DIR
from app.tools.csv_tools import validate_schema

logger = logging.getLogger(__name__)

_USER_ID_KEY = "analysis_user_id"
_SESSION_ID_KEY = "analysis_session_id"
_CSV_PATH_KEY = "analysis_csv_path"
_CSV_NAME_KEY = "analysis_csv_name"
_CSV_DIGEST_KEY = "analysis_csv_digest"
_HISTORY_SELECT_KEY = "analysis_history_selection"
_PENDING_NEW_UPLOAD_KEY = "pending_new_analysis_upload"
_PENDING_VALIDATION_SESSION_KEY = "pending_validation_session_id"
_UPLOAD_VALIDATION_KEY = "analysis_upload_validation"

_EVENT_PROGRESS_LABELS = {
    "validate_dataset": "Validating the uploaded dataset",
    "manager_agent": "Planning the analysis",
    "profiler_agent": "Profiling the dataset",
    "sales_agent_guard": "Analyzing sales",
    "sales_agent": "Analyzing sales",
    "product_agent_guard": "Analyzing products",
    "product_agent": "Analyzing products",
    "customer_agent_guard": "Analyzing customers",
    "customer_agent": "Analyzing customers",
    "insight_agent": "Synthesizing findings",
    "recommendation_agent": "Preparing recommendations",
    "followup_agent": "Preparing your answer",
}
_EVENT_PROGRESS_DETAILS = {
    "Validating the uploaded dataset": (
        "Checking that the CSV can be read and includes the columns and values "
        "needed for the requested analysis."
    ),
    "Planning the analysis": (
        "Selecting which sales, product, and customer calculations are relevant "
        "to your question."
    ),
    "Profiling the dataset": (
        "Summarizing the dataset structure and checking for data-quality issues "
        "before interpreting results."
    ),
    "Analyzing sales": (
        "Calculating completed-order sales totals and sales patterns from the CSV."
    ),
    "Analyzing products": (
        "Comparing product and category performance using completed orders."
    ),
    "Analyzing customers": (
        "Calculating customer counts, repeat-purchase measures, and spending summaries."
    ),
    "Synthesizing findings": (
        "Combining the returned analysis results into a concise finding, with "
        "supporting evidence and caveats."
    ),
    "Preparing recommendations": (
        "Turning the findings into practical suggestions, while noting what "
        "should be checked before acting."
    ),
    "Preparing your answer": (
        "Answering your follow-up from the saved analysis and conversation."
    ),
}
_TOKEN_STREAM_AGENTS = {
    "insight_agent": ("Live findings draft", "json"),
    "recommendation_agent": ("Live recommendations draft", "json"),
    "followup_agent": ("Live answer", "text"),
}
_STREAM_WORD_INTERVAL_SECONDS = 0.04


def _event_progress_label(event: Event) -> str | None:
    for part in event.content.parts if event.content else []:
        function_call = part.function_call
        if function_call is not None:
            tool_labels = {
                "profile_uploaded_dataset": "Profiling the dataset",
                "analyze_sales": "Analyzing sales",
                "analyze_products": "Analyzing products",
                "analyze_customers": "Analyzing customers",
            }
            if function_call.name in tool_labels:
                return tool_labels[function_call.name]
    return _EVENT_PROGRESS_LABELS.get(event.author)


def _event_progress_detail(event: Event) -> str | None:
    label = _event_progress_label(event)
    if label is None:
        return None
    return _EVENT_PROGRESS_DETAILS[label]


def _partial_text(event: Event) -> str:
    if not event.partial or event.content is None:
        return ""

    return "".join(
        part.text
        for part in event.content.parts
        if part.text and not part.thought
    )


def _take_ready_words(text: str, flush: bool = False) -> tuple[list[str], str]:
    words: list[str] = []
    remaining = text
    while match := re.match(r"\s*\S+\s+", remaining):
        words.append(match.group())
        remaining = remaining[match.end():]
    if flush and remaining:
        words.extend(re.findall(r"\s*\S+\s*", remaining))
        remaining = ""
    return words, remaining


def _stream_ui_chunks(
    event_queue: Queue[Event],
    run_future: Future,
    progress,
    progress_detail,
):
    current_progress_label = "Starting analysis..."
    current_text_agent: str | None = None
    pending_text = ""

    while True:
        try:
            event = event_queue.get(timeout=0.1)
        except Empty:
            if run_future.done():
                break
            continue

        label = _event_progress_label(event)
        if label is not None and label != current_progress_label:
            current_progress_label = label
            progress.update(label=label, state="running")
            detail = _event_progress_detail(event)
            if detail:
                progress_detail.markdown(f"**Why this step:** {detail}")

        if event.author not in _TOKEN_STREAM_AGENTS:
            continue
        text = _partial_text(event)
        if not text:
            continue
        if event.author != current_text_agent:
            words, pending_text = _take_ready_words(pending_text, flush=True)
            for word in words:
                yield word
                time.sleep(_STREAM_WORD_INTERVAL_SECONDS)
            title, _ = _TOKEN_STREAM_AGENTS[event.author]
            yield f"\n\n### {title} — draft, not final\n\n"
            current_text_agent = event.author
        pending_text += text
        words, pending_text = _take_ready_words(pending_text)
        for word in words:
            yield word
            time.sleep(_STREAM_WORD_INTERVAL_SECONDS)

    words, pending_text = _take_ready_words(pending_text, flush=True)
    for word in words:
        yield word
        time.sleep(_STREAM_WORD_INTERVAL_SECONDS)
    run_future.result()


def save_uploaded_csv(
    file_name: str,
    content: bytes,
    uploads_dir: Path,
) -> tuple[Path, dict]:
    """Validate CSV content and save it under a generated, trusted name."""
    if Path(file_name).suffix.lower() != ".csv":
        raise ValueError("Please upload a file with the .csv extension.")
    if not content:
        raise ValueError("The uploaded CSV is empty.")

    try:
        dataframe = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"The uploaded file could not be read as a CSV: {exc}") from exc

    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    validation = validate_schema(dataframe)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / f"{uuid4().hex}.csv"
    saved_path.write_bytes(content)
    return saved_path, validation


def _history_label(analysis) -> str:
    question = analysis.turns[0]["question"] if analysis.turns else "New analysis"
    updated = datetime.fromtimestamp(analysis.last_update_time).strftime("%Y-%m-%d %H:%M")
    return f"{question[:55]} ({updated})"


def _render_conversation(analysis) -> None:
    for turn in analysis.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            _render_result(turn["result"])


def _render_result(result: dict) -> None:
    followup_answer = result.get("followup_answer")
    if isinstance(followup_answer, str):
        st.markdown(followup_answer)
        return

    workflow_error = result.get("workflow_error")
    if workflow_error:
        st.error(workflow_error.get("message", "The analysis workflow could not complete."))
        for error in workflow_error.get("errors", []):
            st.write(f"- {error}")
        for warning in workflow_error.get("warnings", []):
            st.warning(warning)
        return

    insights = result.get("business_insights")
    if isinstance(insights, dict):
        st.subheader("Key finding")
        st.write(insights.get("key_finding", "No key finding was returned."))

        for title, key in (
            ("Supporting evidence", "supporting_evidence"),
            ("Possible drivers", "possible_drivers"),
            ("Important caveats", "important_caveats"),
        ):
            items = insights.get(key, [])
            if items:
                st.markdown(f"**{title}**")
                for item in items:
                    st.markdown(f"- {item}")
    elif insights:
        st.subheader("Business insights")
        st.write(insights)

    recommendations = result.get("recommendations")
    if isinstance(recommendations, dict):
        st.subheader("Recommendations")
        for index, recommendation in enumerate(
            recommendations.get("recommendations", []),
            start=1,
        ):
            if not isinstance(recommendation, dict):
                st.write(recommendation)
                continue
            st.markdown(f"**{index}. {recommendation.get('action', 'Suggested action')}**")
            st.write(recommendation.get("rationale", ""))
            for evidence in recommendation.get("evidence", []):
                st.markdown(f"- {evidence}")
            if recommendation.get("validation_needed"):
                st.caption(f"Validate before acting: {recommendation['validation_needed']}")
        if recommendations.get("overall_caveat"):
            st.warning(recommendations["overall_caveat"])
    elif recommendations:
        st.subheader("Recommendations")
        st.write(recommendations)

    validation = result.get("dataset_validation", {})
    for warning in validation.get("warnings", []):
        st.warning(warning)

    with st.expander("Analysis details"):
        if result.get("analysis_goal"):
            st.markdown("**Analysis goal**")
            _render_json_detail(result["analysis_goal"])
        for key in (
            "data_profile",
            "sales_analysis",
            "product_analysis",
            "customer_analysis",
        ):
            if key in result:
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                _render_json_detail(result[key])
        for issue in result.get("structured_output_errors", []):
            st.warning(issue.get("error", "Some structured output could not be validated."))


def _render_json_detail(value: object) -> None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            st.code(value, language="text")
            return
    st.json(value)


def main() -> None:
    """Render the upload-and-chat application."""
    st.set_page_config(
        page_title="Clothing Retail Analyst",
        page_icon="📊",
        layout="wide",
    )
    st.title("Clothing Retail Analyst")
    st.write("Upload a sales CSV, ask a business question, and continue with follow-ups.")

    if _USER_ID_KEY not in st.session_state:
        st.session_state[_USER_ID_KEY] = ANALYSIS_USER_ID
    if st.session_state.pop(_PENDING_NEW_UPLOAD_KEY, False):
        st.session_state.pop(_SESSION_ID_KEY, None)
        st.session_state.pop(_HISTORY_SELECT_KEY, None)
    pending_validation_session_id = st.session_state.pop(
        _PENDING_VALIDATION_SESSION_KEY,
        None,
    )
    if pending_validation_session_id:
        st.session_state[_SESSION_ID_KEY] = pending_validation_session_id
        st.session_state.pop(_HISTORY_SELECT_KEY, None)
    user_id = st.session_state[_USER_ID_KEY]
    analyses = asyncio.run(workflow_service.list_analyses(user_id))
    analyses_by_id = {analysis.session_id: analysis for analysis in analyses}

    with st.sidebar:
        st.header("Analysis history")
        active_session_id = st.session_state.get(_SESSION_ID_KEY)
        if active_session_id not in analyses_by_id:
            active_session_id = None
            st.session_state.pop(_SESSION_ID_KEY, None)
        options = [""] + list(analyses_by_id)
        if st.session_state.get(_HISTORY_SELECT_KEY) not in options:
            st.session_state.pop(_HISTORY_SELECT_KEY, None)
        selected_session_id = st.selectbox(
            "Open a saved analysis",
            options=options,
            index=options.index(active_session_id) if active_session_id else 0,
            format_func=lambda session_id: (
                "New analysis" if not session_id else _history_label(analyses_by_id[session_id])
            ),
            key=_HISTORY_SELECT_KEY,
        )
        if selected_session_id:
            selected_analysis = analyses_by_id[selected_session_id]
            st.session_state[_SESSION_ID_KEY] = selected_session_id
            st.session_state[_CSV_PATH_KEY] = selected_analysis.csv_path
            st.session_state[_CSV_NAME_KEY] = selected_analysis.dataset_name
            st.session_state.pop(_UPLOAD_VALIDATION_KEY, None)
            st.caption(f"Dataset: {selected_analysis.dataset_name}")
            if st.button("Start a new analysis", use_container_width=True):
                st.session_state.pop(_SESSION_ID_KEY, None)
                st.session_state.pop(_HISTORY_SELECT_KEY, None)
                st.rerun()
        else:
            st.session_state.pop(_SESSION_ID_KEY, None)

        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload a sales CSV", type=["csv"])
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            content_digest = sha256(content).hexdigest()
            if content_digest != st.session_state.get(_CSV_DIGEST_KEY):
                st.session_state.pop(_CSV_PATH_KEY, None)
                st.session_state.pop(_CSV_NAME_KEY, None)
                st.session_state.pop(_CSV_DIGEST_KEY, None)
                st.session_state.pop(_UPLOAD_VALIDATION_KEY, None)
                try:
                    saved_path, validation = save_uploaded_csv(
                        uploaded_file.name,
                        content,
                        UPLOADS_DIR,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[_CSV_PATH_KEY] = str(saved_path)
                    st.session_state[_CSV_NAME_KEY] = Path(uploaded_file.name).name
                    st.session_state[_CSV_DIGEST_KEY] = content_digest
                    st.session_state[_UPLOAD_VALIDATION_KEY] = validation
                    if validation["valid"]:
                        st.session_state[_PENDING_NEW_UPLOAD_KEY] = True
                    else:
                        validation_run = asyncio.run(
                            workflow_service.create_validation_report(
                                csv_path=saved_path,
                                validation=validation,
                                user_id=user_id,
                                dataset_name=Path(uploaded_file.name).name,
                            )
                        )
                        st.session_state[_PENDING_VALIDATION_SESSION_KEY] = (
                            validation_run.session_id
                        )
                    st.rerun()

        upload_validation = st.session_state.get(_UPLOAD_VALIDATION_KEY)
        if isinstance(upload_validation, dict):
            if not upload_validation.get("valid"):
                st.error(
                    "The CSV does not match the required schema. A validation report "
                    "has been added to the conversation."
                )
                for error in upload_validation.get("errors", []):
                    st.write(f"- {error}")
            else:
                for warning in upload_validation.get("warnings", []):
                    st.warning(warning)
                if not upload_validation.get("warnings"):
                    st.success("CSV validated and ready.")

        if st.session_state.get(_CSV_PATH_KEY):
            st.caption(f"Using: {st.session_state[_CSV_NAME_KEY]}")

    current_session_id = st.session_state.get(_SESSION_ID_KEY)
    current_analysis = analyses_by_id.get(current_session_id)
    csv_path = st.session_state.get(_CSV_PATH_KEY)
    if not csv_path:
        if analyses:
            st.info("Select a saved analysis or start a new analysis.")
        else:
            st.info("Upload a CSV with the required sales columns to get started.")
        return

    if current_analysis is not None:
        _render_conversation(current_analysis)

    question = st.chat_input("Ask a question about this dataset")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.markdown("### Analysis process")
        progress = st.status("Starting analysis...", expanded=True)
        try:
            event_queue: Queue[Event] = Queue()
            active_session_id = st.session_state.get(_SESSION_ID_KEY)
            dataset_name = st.session_state.get(_CSV_NAME_KEY)

            def run_operation():
                if active_session_id:
                    return asyncio.run(
                        workflow_service.continue_analysis(
                            user_id=user_id,
                            session_id=active_session_id,
                            question=question,
                            on_event=event_queue.put,
                        )
                    )
                return asyncio.run(
                    workflow_service.start_analysis(
                        question=question,
                        csv_path=csv_path,
                        user_id=user_id,
                        dataset_name=dataset_name,
                        on_event=event_queue.put,
                    )
                )

            with progress:
                st.caption(
                    "Agent progress appears here. The streamed response appears below."
                )
                progress_detail = st.empty()
            with ThreadPoolExecutor(max_workers=1) as executor:
                run_future = executor.submit(run_operation)
                st.write_stream(
                    _stream_ui_chunks(
                        event_queue,
                        run_future,
                        progress,
                        progress_detail,
                    ),
                    cursor="▌",
                )
                run = run_future.result()
            if not active_session_id:
                st.session_state[_SESSION_ID_KEY] = run.session_id
                st.session_state.pop(_HISTORY_SELECT_KEY, None)
        except Exception:
            progress.update(label="Analysis failed", state="error")
            logger.exception("The analysis workflow failed while handling a UI question.")
            st.error("Analysis failed. Check the application logs for details and try again.")
        else:
            progress.update(label="Analysis complete", state="complete", expanded=False)
            st.markdown("### Final response")
            _render_result(run.result)
            st.rerun()


if __name__ == "__main__":
    main()

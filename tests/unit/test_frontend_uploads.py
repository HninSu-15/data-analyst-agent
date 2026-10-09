import asyncio
from concurrent.futures import Future
from pathlib import Path
from queue import Queue

import pandas as pd
import pytest
from google.adk.events import Event
from google.genai import types
from streamlit.testing.v1 import AppTest

from app.agent import workflow_service
from app.tools.csv_tools import REQUIRED_COLUMNS
from frontend.streamlit_app import (
    _CSV_PATH_KEY,
    _HISTORY_SELECT_KEY,
    _SESSION_ID_KEY,
    _USER_ID_KEY,
    _event_progress_detail,
    _event_progress_label,
    _partial_text,
    _stream_ui_chunks,
    save_uploaded_csv,
)


def test_streamlit_app_renders_without_errors():
    app_path = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Clothing Retail Analyst"
    assert app.selectbox[0].label == "Open a saved analysis"
    assert len(app.get("file_uploader")) == 1


def test_streaming_helpers_select_progress_and_only_partial_text():
    text_event = Event(
        author="insight_agent",
        content=types.Content(
            role="model",
            parts=[types.Part(text='{"key_finding":'), types.Part(text='"Top product"}')],
        ),
        partial=True,
    )
    final_event = Event(
        author="insight_agent",
        content=types.Content(role="model", parts=[types.Part(text="complete result")]),
        partial=False,
    )
    tool_event = Event(
        author="product_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="analyze_products", args={})
                )
            ],
        ),
        partial=True,
    )
    mixed_event = Event(
        author="product_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Checking the product totals."),
                types.Part(
                    function_call=types.FunctionCall(
                        name="analyze_products",
                        args={"include_returns": False},
                    )
                ),
            ],
        ),
        partial=True,
    )
    thought_event = Event(
        author="product_agent",
        content=types.Content(
            role="model",
            parts=[types.Part(text="hidden reasoning", thought=True)],
        ),
        partial=True,
    )

    assert _event_progress_label(tool_event) == "Analyzing products"
    assert "completed orders" in _event_progress_detail(tool_event)
    assert _partial_text(text_event) == '{"key_finding":"Top product"}'
    assert _partial_text(final_event) == ""
    assert _partial_text(tool_event) == ""
    assert _partial_text(mixed_event) == "Checking the product totals."
    assert _partial_text(thought_event) == ""


def test_stream_ui_chunks_yield_each_partial_chunk_for_typewriter_rendering():
    class Progress:
        def update(self, **kwargs):
            pass

    class Detail:
        def markdown(self, text):
            pass

    event_queue = Queue()
    event_queue.put(
        Event(
            author="followup_agent",
            content=types.Content(role="model", parts=[types.Part(text="The ")]),
            partial=True,
        )
    )
    event_queue.put(
        Event(
            author="followup_agent",
            content=types.Content(role="model", parts=[types.Part(text="answer.")]),
            partial=True,
        )
    )
    run_future = Future()
    run_future.set_result(None)

    chunks = list(
        _stream_ui_chunks(
            event_queue,
            run_future,
            progress=Progress(),
            progress_detail=Detail(),
        )
    )

    assert chunks == [
        "\n\n### Live answer — draft, not final\n\n",
        "The ",
        "answer.",
    ]


def test_saved_analysis_with_plain_text_details_renders_without_json_errors():
    user_id = "plain-text-detail-render-test"

    async def create_session():
        await workflow_service.session_service.create_session(
            app_name=workflow_service.app_name,
            user_id=user_id,
            session_id="plain-text-detail-render-session",
            state={
                "csv_path": str(Path("data/sample/sales.csv").resolve()),
                "dataset_name": "sales.csv",
                "analysis_turns": [
                    {
                        "question": "Summarize this data.",
                        "result": {
                            "analysis_goal": '{"analysis_goal": "Summarize sales"}',
                            "data_profile": "## Dataset profile\\n100 rows, 16 columns",
                            "sales_analysis": "Malformed { JSON result",
                        },
                    },
                    {
                        "question": "What was the strongest month?",
                        "result": {"followup_answer": "June had the strongest sales."},
                    },
                ],
            },
        )

    asyncio.run(create_session())
    app_path = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.session_state[_USER_ID_KEY] = user_id
    app.run()
    app.selectbox[0].select_index(1).run()

    assert not app.exception
    assert any("Dataset profile" in element.value for element in app.code)
    assert any("Malformed { JSON result" in element.value for element in app.code)
    assert any("June had the strongest sales." in element.value for element in app.markdown)


def test_new_invalid_upload_clears_selected_history_and_shows_validation(tmp_path):
    user_id = "invalid-upload-history-switch-test"
    session_id = "invalid-upload-history-switch-session"

    async def create_session():
        await workflow_service.session_service.create_session(
            app_name=workflow_service.app_name,
            user_id=user_id,
            session_id=session_id,
            state={
                "csv_path": str(Path("data/sample/sales.csv").resolve()),
                "dataset_name": "sales.csv",
                "analysis_turns": [
                    {
                        "question": "Show the old analysis.",
                        "result": {"business_insights": {"key_finding": "Old analysis"}},
                    }
                ],
            },
        )

    asyncio.run(create_session())
    app_path = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.session_state[_USER_ID_KEY] = user_id
    app.session_state[_SESSION_ID_KEY] = session_id
    app.session_state[_HISTORY_SELECT_KEY] = session_id
    app.run()
    app.get("file_uploader")[0].set_value(
        ("invalid.csv", b"order_id,quantity\nORD-1,1\n", "text/csv")
    ).run()

    uploaded_path = Path(app.session_state[_CSV_PATH_KEY])
    try:
        assert not app.exception
        assert "product_id" in " ".join(
            str(getattr(element, "value", "")) for element in app.markdown
        )
        assert "Dataset validation failed" in " ".join(
            str(getattr(element, "value", "")) for element in app.error
        )
        assert "Show the old analysis." not in " ".join(
            str(getattr(element, "value", "")) for element in app.markdown
        )
        assert _SESSION_ID_KEY in app.session_state
    finally:
        uploaded_path.unlink(missing_ok=True)


def test_save_uploaded_csv_uses_generated_name_and_validates_schema(tmp_path):
    source = pd.DataFrame(
        [{column: "value" for column in REQUIRED_COLUMNS}],
    )
    source["order_date"] = "2025-01-01"
    for column in ("quantity", "unit_price", "discount_pct", "cost_per_unit"):
        source[column] = 1
    source["order_status"] = "Completed"
    content = source.to_csv(index=False).encode("utf-8")

    saved_path, validation = save_uploaded_csv(
        r"..\untrusted\customer-data.csv",
        content,
        tmp_path,
    )

    assert saved_path.parent == tmp_path
    assert saved_path.name != "customer-data.csv"
    assert saved_path.suffix == ".csv"
    assert saved_path.read_bytes() == content
    assert validation["valid"] is True


@pytest.mark.parametrize(
    ("file_name", "content", "expected_error"),
    [
        ("data.txt", b"contents", "extension"),
        ("data.csv", b"", "empty"),
    ],
)
def test_save_uploaded_csv_rejects_invalid_uploads(
    tmp_path,
    file_name,
    content,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        save_uploaded_csv(file_name, content, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_save_uploaded_csv_preserves_schema_validation_failure_for_workflow(tmp_path):
    content = b"order_id,quantity\nORD-1,1\n"

    saved_path, validation = save_uploaded_csv("data.csv", content, tmp_path)

    assert saved_path.exists()
    assert saved_path.read_bytes() == content
    assert validation["valid"] is False
    assert "product_id" in validation["missing_columns"]

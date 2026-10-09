import asyncio
from concurrent.futures import ThreadPoolExecutor

from google.adk import Context, Runner
from google.adk.agents._streaming_mode import StreamingMode
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import FunctionNode, JoinNode, START, Workflow
from google.genai import types

from app.agent import WorkflowService, root_agent, run_analysis
from app.agent import _guarded_specialist_node
from google.adk.sessions import DatabaseSessionService


def _fresh_session_service(tmp_path):
    db_path = tmp_path / "adk_sessions.db"
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_path.as_posix()}")


def test_workflow_validates_dataset_before_invoking_llm_agents(tmp_path):
    invalid_csv = tmp_path / "missing_columns.csv"
    invalid_csv.write_text("order_id,quantity\nORD-1,1\n", encoding="utf-8")

    result = asyncio.run(
        run_analysis(
            "What were total sales?",
            invalid_csv,
        )
    )

    assert result["workflow_error"]["status"] == "validation_failed"
    assert "product_id" in result["workflow_error"]["missing_columns"]
    assert "analysis_goal" not in result


def test_workflow_service_reuses_session_dataset_and_prior_results(tmp_path):
    invalid_csv = tmp_path / "missing_columns.csv"
    invalid_csv.write_text("order_id,quantity\nORD-1,1\n", encoding="utf-8")
    service = WorkflowService(session_service=_fresh_session_service(tmp_path))

    async def run_turns():
        first = await service.start_analysis(
            "What were total sales?",
            invalid_csv,
            user_id="test-user",
        )
        second = await service.continue_analysis(
            user_id=first.user_id,
            session_id=first.session_id,
            question="Which category generated the most revenue?",
        )
        session = await service.session_service.get_session(
            app_name=service.app_name,
            user_id=first.user_id,
            session_id=first.session_id,
        )
        history = await service.list_analyses(first.user_id)
        reopened = await service.get_analysis(first.user_id, first.session_id)
        return first, second, session, history, reopened

    first, second, session, history, reopened = asyncio.run(run_turns())

    assert first.session_id == second.session_id
    assert first.user_id == second.user_id == "test-user"
    assert first.result["workflow_error"]["status"] == "validation_failed"
    assert "CSV analysis did not complete successfully" in second.result["followup_answer"]
    assert session is not None
    assert session.state["csv_path"] == str(invalid_csv.resolve())
    assert session.state["user_question"] == "What were total sales?"
    assert len(history) == 1
    assert history[0].session_id == first.session_id
    assert history[0].dataset_name == invalid_csv.name
    assert reopened is not None
    assert [turn["question"] for turn in reopened.turns] == [
        "What were total sales?",
        "Which category generated the most revenue?",
    ]
    assert reopened.turns[0]["result"]["workflow_error"]["status"] == "validation_failed"
    assert reopened.turns[1]["result"] == second.result


def test_followup_uses_saved_analysis_without_rerunning_full_workflow(tmp_path):
    session_service = _fresh_session_service(tmp_path)

    class StubRunner:
        def __init__(self, state_delta):
            self.state_delta = state_delta
            self.calls = []
            self.run_configs = []

        async def run_async(
            self,
            *,
            user_id,
            session_id,
            new_message,
            state_delta=None,
            run_config=None,
        ):
            self.calls.append(new_message.parts[0].text)
            self.run_configs.append(run_config)
            session = await session_service.get_session(
                app_name="clothing_retail_analyst",
                user_id=user_id,
                session_id=session_id,
            )
            changes = dict(state_delta or {})
            changes.update(self.state_delta)
            await session_service.append_event(
                session,
                Event(
                    author="stub_agent",
                    invocation_id=f"stub-{len(self.calls)}",
                    actions=EventActions(state_delta=changes),
                ),
            )
            yield Event(
                author="stub_agent",
                invocation_id=f"partial-{len(self.calls)}",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="streamed chunk")],
                ),
                partial=True,
            )

    workflow_runner = StubRunner(
        {
            "analysis_goal": {"analysis_goal": "Summarize sales"},
            "business_insights": {"key_finding": "June had the highest sales"},
        }
    )
    followup_runner = StubRunner({"followup_answer": "June had the highest sales."})
    streamed_events = []
    service = WorkflowService(
        session_service=session_service,
        runner=workflow_runner,
        followup_runner=followup_runner,
    )

    async def run_turns():
        first = await service.start_analysis(
            "Summarize sales",
            tmp_path / "saved.csv",
            user_id="followup-user",
            on_event=streamed_events.append,
        )
        followup = await service.continue_analysis(
            user_id=first.user_id,
            session_id=first.session_id,
            question="Which month had the highest sales?",
            on_event=streamed_events.append,
        )
        reopened = await service.get_analysis(first.user_id, first.session_id)
        return first, followup, reopened

    first, followup, reopened = asyncio.run(run_turns())

    assert workflow_runner.calls == ["Summarize sales"]
    assert followup_runner.calls == ["Which month had the highest sales?"]
    assert all(
        config is not None and config.streaming_mode is StreamingMode.SSE
        for config in workflow_runner.run_configs + followup_runner.run_configs
    )
    assert [event.partial for event in streamed_events] == [True, True]
    assert all(event.content.parts[0].text == "streamed chunk" for event in streamed_events)
    assert followup.result == {"followup_answer": "June had the highest sales."}
    assert reopened is not None
    assert reopened.turns[0]["result"] == first.result
    assert reopened.turns[1] == {
        "question": "Which month had the highest sales?",
        "result": followup.result,
    }


def test_create_validation_report_persists_failure_without_running_agents(tmp_path):
    class RunnerMustNotRun:
        async def run_async(self, **kwargs):
            raise AssertionError("Validation reports must not invoke the analysis runner.")
            yield None

    csv_path = tmp_path / "wrong_schema.csv"
    csv_path.write_text("order_id,quantity\nORD-1,1\n", encoding="utf-8")
    validation = {
        "valid": False,
        "missing_columns": ["product_id"],
        "errors": ["Missing required column: product_id"],
        "warnings": [],
    }
    service = WorkflowService(
        session_service=_fresh_session_service(tmp_path),
        runner=RunnerMustNotRun(),
    )

    async def run():
        report = await service.create_validation_report(
            csv_path=csv_path,
            validation=validation,
            user_id="validation-report-user",
            dataset_name="wrong_schema.csv",
        )
        reopened = await service.get_analysis(report.user_id, report.session_id)
        return report, reopened

    report, reopened = asyncio.run(run())

    assert report.result["workflow_error"]["status"] == "validation_failed"
    assert report.result["workflow_error"]["missing_columns"] == ["product_id"]
    assert reopened is not None
    assert reopened.dataset_name == "wrong_schema.csv"
    assert reopened.turns == [
        {
            "question": "Validate uploaded dataset",
            "result": report.result,
        }
    ]


def test_workflow_service_does_not_allow_another_user_to_resume_session(tmp_path):
    invalid_csv = tmp_path / "missing_columns.csv"
    invalid_csv.write_text("order_id,quantity\nORD-1,1\n", encoding="utf-8")
    service = WorkflowService(session_service=_fresh_session_service(tmp_path))

    async def run():
        first = await service.start_analysis(
            "What were total sales?",
            invalid_csv,
            user_id="owner",
        )
        try:
            await service.continue_analysis(
                user_id="other-user",
                session_id=first.session_id,
                question="Follow-up question",
            )
        except ValueError as exc:
            assert "not found for this user" in str(exc)
        else:
            raise AssertionError("A different user must not resume this session.")

    asyncio.run(run())


def test_streaming_callback_can_run_workflow_on_ui_worker_thread(tmp_path):
    invalid_csv = tmp_path / "missing_columns.csv"
    invalid_csv.write_text("order_id,quantity\nORD-1,1\n", encoding="utf-8")
    service = WorkflowService(session_service=_fresh_session_service(tmp_path))
    streamed_events = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        run = executor.submit(
            asyncio.run,
            service.start_analysis(
                "What were total sales?",
                invalid_csv,
                user_id="streaming-worker-user",
                on_event=streamed_events.append,
            ),
        ).result()

    reopened = asyncio.run(service.get_analysis(run.user_id, run.session_id))

    assert streamed_events
    assert run.result["workflow_error"]["status"] == "validation_failed"
    assert reopened is not None
    assert reopened.turns[0]["result"] == run.result


def test_root_agent_has_expected_graph_topology():
    graph = root_agent.graph
    assert graph is not None

    edges = {
        (edge.from_node.name, edge.to_node.name, edge.route)
        for edge in graph.edges
    }
    assert ("__START__", "validate_dataset", None) in edges
    assert ("validate_dataset", "manager_agent", "valid") in edges
    assert ("validate_dataset", "invalid_dataset", "invalid") in edges
    assert ("profiler_agent", "sales_agent_guard", None) in edges
    assert ("profiler_agent", "product_agent_guard", None) in edges
    assert ("profiler_agent", "customer_agent_guard", None) in edges
    assert ("sales_agent_guard", "specialist_join", None) in edges
    assert ("product_agent_guard", "specialist_join", None) in edges
    assert ("customer_agent_guard", "specialist_join", None) in edges
    assert ("specialist_join", "insight_agent", None) in edges
    assert ("insight_agent", "recommendation_agent", None) in edges


def test_failed_specialist_does_not_discard_successful_parallel_result():
    async def successful_specialist(ctx: Context):
        ctx.state["successful_analysis"] = {"revenue": 125.0}
        return {"revenue": 125.0}

    async def failing_specialist():
        raise RuntimeError("simulated specialist failure")

    success_node = FunctionNode(func=successful_specialist, name="successful_specialist")
    failure_node = FunctionNode(func=failing_specialist, name="failing_specialist")
    success_guard = _guarded_specialist_node(success_node, "successful_analysis")
    failure_guard = _guarded_specialist_node(failure_node, "failed_analysis")
    join = JoinNode(name="test_join")
    workflow = Workflow(
        name="specialist_failure_test",
        edges=[
            (START, (success_guard, failure_guard)),
            (success_guard, join),
            (failure_guard, join),
        ],
    )

    async def run() -> dict:
        app_name = "specialist_failure_test"
        user_id = "test_user"
        session_id = "test_session"
        session_service = InMemorySessionService()
        runner = Runner(node=workflow, app_name=app_name, session_service=session_service)
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        async for _ in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Run specialist analyses.")],
            ),
        ):
            pass
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        assert session is not None
        return session.state

    result = asyncio.run(run())
    assert result["successful_analysis"] == {"revenue": 125.0}
    assert result["failed_analysis"]["available"] is False
    assert "failed with" in result["failed_analysis"]["error"]

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from google.adk import Context, Runner
from google.adk.agents import RunConfig
from google.adk.agents._streaming_mode import StreamingMode
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.workflow import BaseNode, FunctionNode, JoinNode, START, Workflow
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.agents.customer_agent import customer_agent
from app.agents.followup_agent import followup_agent
from app.agents.insight_agent import insight_agent
from app.agents.manager_agent import manager_agent
from app.agents.product_agent import product_agent
from app.agents.profiler_agent import profiler_agent
from app.agents.recommendation_agent import recommendation_agent
from app.agents.sales_agent import sales_agent
from app.config import (
    LIVE_QUALITY_EVALUATION_ENABLED,
    SAMPLE_DATA_PATH,
    SESSION_DB_PATH,
)
from app.live_quality import (
    QualityEvaluationManager,
    create_live_invocation,
    live_quality_evaluations,
)
from app.schemas.analysis_models import AnalysisPlan, BusinessInsights, Recommendations
from app.tools import load_csv, validate_schema

logger = logging.getLogger(__name__)
EventCallback = Callable[[Event], None]
_LIVE_QUALITY_STATE_PREFIX = "live_response_quality_"

STRUCTURED_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "analysis_goal": AnalysisPlan,
    "business_insights": BusinessInsights,
    "recommendations": Recommendations,
}


def validate_dataset_node(ctx: Context) -> Event:
    """Validate the session CSV before any model-backed agent is run."""
    csv_path = ctx.state.get("csv_path")
    if not isinstance(csv_path, str) or not csv_path.strip():
        validation: dict[str, Any] = {
            "valid": False,
            "errors": ["No CSV path was provided in session state under 'csv_path'."],
            "warnings": [],
        }
    else:
        try:
            validation = validate_schema(load_csv(csv_path))
        except (FileNotFoundError, ValueError, OSError) as exc:
            validation = {"valid": False, "errors": [str(exc)], "warnings": []}

    ctx.state["dataset_validation"] = validation
    route = "valid" if validation["valid"] else "invalid"
    return Event(
        output={"validation": validation},
        actions=EventActions(route=route),
    )


def invalid_dataset_node(ctx: Context) -> dict[str, Any]:
    """Create a clear final result when the CSV cannot pass validation."""
    validation = ctx.state.get("dataset_validation", {})
    result = {
        "status": "validation_failed",
        "message": "Dataset validation failed. Analysis was not run.",
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "missing_columns": validation.get("missing_columns", []),
    }
    ctx.state["workflow_error"] = result
    return result


def _guarded_specialist_node(agent: BaseNode, output_key: str) -> FunctionNode:
    async def run_specialist(ctx: Context, node_input: Any) -> dict[str, Any]:
        try:
            await ctx.run_node(agent, node_input)
        except Exception as exc:
            logger.exception("Specialist agent %s failed.", agent.name)
            failure = {
                "available": False,
                "error": f"{agent.name} failed with {type(exc).__name__}.",
            }
            ctx.state[output_key] = failure
            return failure

        if output_key not in ctx.state:
            logger.error(
                "Specialist agent %s completed without writing %s.",
                agent.name,
                output_key,
            )
            failure = {
                "available": False,
                "error": f"{agent.name} did not produce an analysis result.",
            }
            ctx.state[output_key] = failure
            return failure

        return {"available": True}

    return FunctionNode(
        func=run_specialist,
        name=f"{agent.name}_guard",
        rerun_on_resume=True,
    )


def _parse_structured_output(
    state: dict[str, Any],
    key: str,
    model: type[BaseModel],
) -> None:
    value = state.get(key)
    if isinstance(value, model):
        state[key] = value.model_dump(mode="json")
        return
    if isinstance(value, dict):
        try:
            state[key] = model.model_validate(value).model_dump(mode="json")
        except ValidationError as exc:
            issues = _format_validation_errors(exc)
            raise ValueError(
                f"Agent output '{key}' did not match {model.__name__}: {issues}"
            ) from exc
        return
    if not isinstance(value, str):
        raise ValueError(f"Agent did not produce structured output for '{key}'.")

    json_text = value.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        json_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        json_text = fenced_match.group(1)

    try:
        parsed_json = _decode_json_object(json_text)
    except ValueError as exc:
        raise ValueError(
            f"Agent output '{key}' did not contain a valid JSON object: {exc}"
        ) from exc

    try:
        parsed = model.model_validate(parsed_json)
    except ValidationError as exc:
        issues = _format_validation_errors(exc)
        raise ValueError(
            f"Agent output '{key}' JSON did not match {model.__name__}: {issues}"
        ) from exc
    state[key] = parsed.model_dump(mode="json")


def _decode_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no complete JSON object was found in the agent response.")


def _format_validation_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors(include_url=False)
    )


def _normalize_analysis_outputs(result: dict[str, Any]) -> None:
    structured_output_errors: list[dict[str, str | bool]] = []
    for key, model in STRUCTURED_OUTPUT_MODELS.items():
        if key not in result:
            continue
        try:
            _parse_structured_output(result, key, model)
        except ValueError as exc:
            logger.warning(
                "Could not validate structured agent output for %s; preserving the original output.",
                key,
            )
            structured_output_errors.append(
                {
                    "output_key": key,
                    "error": str(exc),
                    "raw_output_preserved": True,
                }
            )
    if structured_output_errors:
        result["structured_output_errors"] = structured_output_errors


validation_node = FunctionNode(func=validate_dataset_node, name="validate_dataset")
invalid_dataset_node = FunctionNode(
    func=invalid_dataset_node,
    name="invalid_dataset",
)
sales_node = _guarded_specialist_node(sales_agent, "sales_analysis")
product_node = _guarded_specialist_node(product_agent, "product_analysis")
customer_node = _guarded_specialist_node(customer_agent, "customer_analysis")
specialist_join = JoinNode(
    name="specialist_join",
    description="Waits for sales, product, and customer analyses to complete.",
)

root_agent = Workflow(
    name="clothing_retail_analyst",
    description=(
        "Validates a clothing-retail sales CSV, plans analysis, runs sales/product/customer "
        "agents concurrently, and synthesizes evidence-grounded insights and recommendations."
    ),
    max_concurrency=3,
    edges=[
        (START, validation_node),
        (validation_node, {"valid": manager_agent, "invalid": invalid_dataset_node}),
        (manager_agent, profiler_agent),
        (profiler_agent, (sales_node, product_node, customer_node)),
        (sales_node, specialist_join),
        (product_node, specialist_join),
        (customer_node, specialist_join),
        (specialist_join, insight_agent),
        (insight_agent, recommendation_agent),
    ],
)


@dataclass(frozen=True)
class AnalysisRun:
    user_id: str
    session_id: str
    result: dict[str, Any]


@dataclass(frozen=True)
class AnalysisHistory:
    session_id: str
    csv_path: str
    dataset_name: str
    turns: list[dict[str, Any]]
    last_update_time: float


class WorkflowService:
    """Own an ADK runner and session service for multi-turn analyses."""

    def __init__(
        self,
        session_service: BaseSessionService | None = None,
        runner: Runner | None = None,
        followup_runner: Runner | None = None,
        quality_evaluation_manager: QualityEvaluationManager | None = None,
    ) -> None:
        self.app_name = "clothing_retail_analyst"
        if session_service is None:
            SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            session_service = DatabaseSessionService(
                db_url=f"sqlite+aiosqlite:///{SESSION_DB_PATH.as_posix()}"
            )
        self.session_service = session_service
        self.runner = runner or Runner(
            node=root_agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )
        self.followup_runner = followup_runner or Runner(
            node=followup_agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )
        self.quality_evaluation_manager = (
            quality_evaluation_manager or live_quality_evaluations
        )

    async def start_analysis(
        self,
        question: str,
        csv_path: str | Path,
        user_id: str | None = None,
        dataset_name: str | None = None,
        on_event: EventCallback | None = None,
    ) -> AnalysisRun:
        """Start an analysis and return identifiers for follow-up turns."""
        normalized_question = self._validate_question(question)
        session_user_id = user_id or f"user-{uuid4().hex}"
        session_id = uuid4().hex
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=session_user_id,
            session_id=session_id,
            state={
                "csv_path": str(Path(csv_path).resolve()),
                "dataset_name": dataset_name or Path(csv_path).name,
                "user_question": normalized_question,
            },
        )
        result = await self._run_turn(
            user_id=session_user_id,
            session_id=session_id,
            question=normalized_question,
            on_event=on_event,
        )
        return AnalysisRun(session_user_id, session_id, result)

    async def create_validation_report(
        self,
        csv_path: str | Path,
        validation: dict[str, Any],
        user_id: str,
        dataset_name: str | None = None,
    ) -> AnalysisRun:
        """Persist an upload-time schema failure without invoking an LLM runner."""
        session_id = uuid4().hex
        csv_file = Path(csv_path).resolve()
        workflow_error = {
            "status": "validation_failed",
            "message": "Dataset validation failed. Analysis was not run.",
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
            "missing_columns": validation.get("missing_columns", []),
        }
        result = {
            "dataset_validation": validation,
            "workflow_error": workflow_error,
        }
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            state={
                "csv_path": str(csv_file),
                "dataset_name": dataset_name or csv_file.name,
                "dataset_validation": validation,
                "workflow_error": workflow_error,
            },
        )
        await self._append_analysis_turn(session, "Validate uploaded dataset", result)
        return AnalysisRun(user_id, session_id, result)

    async def continue_analysis(
        self,
        user_id: str,
        session_id: str,
        question: str,
        on_event: EventCallback | None = None,
    ) -> AnalysisRun:
        """Continue an analysis in its existing ADK session and dataset context."""
        normalized_question = self._validate_question(question)
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            raise ValueError("Analysis session was not found for this user.")
        result = await self._run_followup_turn(
            user_id=user_id,
            session_id=session_id,
            question=normalized_question,
            session=session,
            on_event=on_event,
        )
        return AnalysisRun(user_id, session_id, result)

    async def list_analyses(self, user_id: str) -> list[AnalysisHistory]:
        """Return this user's sessions that contain an analysis dataset."""
        response = await self.session_service.list_sessions(
            app_name=self.app_name,
            user_id=user_id,
        )
        analyses: list[AnalysisHistory] = []
        for listed_session in reversed(response.sessions):
            analysis = await self.get_analysis(user_id, listed_session.id)
            if analysis is not None:
                analyses.append(analysis)
        return analyses

    async def get_analysis(
        self,
        user_id: str,
        session_id: str,
    ) -> AnalysisHistory | None:
        """Load the saved transcript and dataset for a user's session."""
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            return None
        csv_path = session.state.get("csv_path")
        if not isinstance(csv_path, str) or not csv_path:
            return None

        turns = session.state.get("analysis_turns", [])
        if not turns:
            question = session.state.get("user_question")
            result = self._result_from_state(session.state)
            if isinstance(question, str) and question and result:
                turns = [{"question": question, "result": result}]

        resolved_turns = []
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("result"), dict):
                resolved_turns.append(turn)
                continue
            result = turn["result"]
            quality = result.get("live_response_quality")
            if not isinstance(quality, dict):
                resolved_turns.append(turn)
                continue
            evaluation_id = quality.get("evaluation_id")
            outcome = (
                session.state.get(f"{_LIVE_QUALITY_STATE_PREFIX}{evaluation_id}")
                if isinstance(evaluation_id, str)
                else None
            )
            if not isinstance(outcome, dict):
                resolved_turns.append(turn)
                continue
            resolved_turn = dict(turn)
            resolved_result = dict(result)
            resolved_result["live_response_quality"] = {
                "evaluation_id": evaluation_id,
                **outcome,
            }
            resolved_turn["result"] = resolved_result
            resolved_turns.append(resolved_turn)

        return AnalysisHistory(
            session_id=session.id,
            csv_path=csv_path,
            dataset_name=session.state.get("dataset_name", Path(csv_path).name),
            turns=resolved_turns,
            last_update_time=session.last_update_time,
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("The business question cannot be empty.")
        return normalized_question

    async def _run_turn(
        self,
        user_id: str,
        session_id: str,
        question: str,
        state_delta: dict[str, Any] | None = None,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        events: list[Event] = []
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            state_delta=state_delta,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=question)],
            ),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            events.append(event)
            if on_event is not None:
                on_event(event)

        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            raise RuntimeError("ADK completed the workflow but its session could not be retrieved.")

        result = self._result_from_state(session.state)
        _normalize_analysis_outputs(result)
        if LIVE_QUALITY_EVALUATION_ENABLED and not result.get("workflow_error"):
            result["live_response_quality"] = self._schedule_quality_evaluation(
                question,
                result,
                events,
            )

        await self._append_analysis_turn(session, question, result)
        return result

    async def _run_followup_turn(
        self,
        user_id: str,
        session_id: str,
        question: str,
        session: Any,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        turns = list(session.state.get("analysis_turns", []))
        base_turn = next(
            (
                turn
                for turn in turns
                if isinstance(turn, dict)
                and isinstance(turn.get("result"), dict)
                and "followup_answer" not in turn["result"]
            ),
            None,
        )
        analysis_result = (
            base_turn["result"]
            if base_turn is not None
            else self._result_from_state(session.state)
        )
        has_saved_analysis = any(
            key in analysis_result
            for key in (
                "analysis_goal",
                "data_profile",
                "sales_analysis",
                "product_analysis",
                "customer_analysis",
                "business_insights",
                "recommendations",
            )
        )

        if analysis_result.get("workflow_error") or not has_saved_analysis:
            answer = (
                "I can’t answer from a saved analysis because the CSV analysis did not "
                "complete successfully. Please upload a valid CSV and start a new analysis."
            )
        else:
            events: list[Event] = []
            async for event in self.followup_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                state_delta={"followup_answer": None},
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=question)],
                ),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                events.append(event)
                if on_event is not None:
                    on_event(event)

            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session is None:
                raise RuntimeError(
                    "ADK completed the follow-up but its session could not be retrieved."
                )
            answer = session.state.get("followup_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The follow-up agent did not produce a text answer.")
            answer = answer.strip()

        result = {"followup_answer": answer}
        if (
            LIVE_QUALITY_EVALUATION_ENABLED
            and has_saved_analysis
            and not analysis_result.get("workflow_error")
        ):
            result["live_response_quality"] = self._schedule_quality_evaluation(
                question,
                result,
                events,
                evidence_context=analysis_result,
            )
        await self._append_analysis_turn(session, question, result)
        return result

    def _schedule_quality_evaluation(
        self,
        question: str,
        result: dict[str, Any],
        events: list[Event],
        evidence_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            invocation = create_live_invocation(
                question,
                result,
                events,
                evidence_context=evidence_context,
            )
            evaluation_id = self.quality_evaluation_manager.submit(invocation)
        except Exception:
            logger.exception("Could not schedule live response-quality evaluation.")
            return {
                "status": "error",
                "message": (
                    "The response was delivered, but its quality evaluation "
                    "could not be scheduled. Check the application logs."
                ),
            }
        return {"evaluation_id": evaluation_id, "status": "pending"}

    async def save_live_quality_evaluation(
        self,
        user_id: str,
        session_id: str,
        evaluation_id: str,
        outcome: dict[str, Any],
    ) -> bool:
        """Persist a background judge result into its matching conversation turn."""
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            raise ValueError("Analysis session was not found for this user.")

        turns = session.state.get("analysis_turns", [])
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("result"), dict):
                continue
            quality = turn["result"].get("live_response_quality")
            if (
                not isinstance(quality, dict)
                or quality.get("evaluation_id") != evaluation_id
            ):
                continue
            state_key = f"{_LIVE_QUALITY_STATE_PREFIX}{evaluation_id}"
            if state_key in session.state:
                return True
            await self.session_service.append_event(
                session,
                Event(
                    author="live_quality_evaluation",
                    invocation_id=f"live-quality-{evaluation_id}",
                    actions=EventActions(state_delta={state_key: outcome}),
                ),
            )
            return True
        return False

    async def _append_analysis_turn(
        self,
        session: Any,
        question: str,
        result: dict[str, Any],
    ) -> None:
        turns = list(session.state.get("analysis_turns", []))
        turns.append({"question": question, "result": result})
        await self.session_service.append_event(
            session,
            Event(
                author="analysis_history",
                invocation_id=f"analysis-history-{uuid4().hex}",
                actions=EventActions(state_delta={"analysis_turns": turns}),
            ),
        )

    @staticmethod
    def _result_from_state(state: dict[str, Any]) -> dict[str, Any]:
        result_keys = (
            "analysis_goal",
            "dataset_validation",
            "data_profile",
            "sales_analysis",
            "product_analysis",
            "customer_analysis",
            "business_insights",
            "recommendations",
            "workflow_error",
        )
        return {key: state[key] for key in result_keys if key in state}


workflow_service = WorkflowService()


async def run_analysis(question: str, csv_path: str | Path) -> dict[str, Any]:
    """Compatibility helper for one-shot CLI runs."""
    run = await workflow_service.start_analysis(question, csv_path)
    return run.result


async def _cli(args: argparse.Namespace) -> None:
    result = await run_analysis(args.question, args.csv_path)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    """Run the clothing-retail ADK analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Run the clothing-retail ADK analysis workflow."
    )
    parser.add_argument(
        "--csv-path",
        default=str(SAMPLE_DATA_PATH),
        help="Path to a sales CSV file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Business question to analyze.",
    )
    args = parser.parse_args()
    asyncio.run(_cli(args))


if __name__ == "__main__":
    main()

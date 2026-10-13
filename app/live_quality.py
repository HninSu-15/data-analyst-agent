from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import re
import threading
import time
from typing import Any, Protocol
from uuid import uuid4

from google.adk.evaluation.eval_case import Invocation, InvocationEvent, InvocationEvents
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    JudgeModelOptions,
    PrebuiltMetrics,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.rubric_based_evaluator import (
    DefaultAutoRaterResponseParser,
    RubricResponse,
)
from google.adk.events import Event
from google.genai import types

from app.config import LIVE_QUALITY_JUDGE_MODEL, LIVE_QUALITY_THRESHOLD

logger = logging.getLogger(__name__)

_QUALITY_RUBRICS = [
    Rubric(
        rubric_id="evidence_grounding",
        rubric_content=RubricContent(
            text_property=(
                "Every factual claim and numerical value in the response is "
                "supported by the available analysis results; the agent does "
                "not invent data."
            )
        ),
    ),
    Rubric(
        rubric_id="question_relevance",
        rubric_content=RubricContent(
            text_property=(
                "The response directly addresses the user's question and "
                "emphasizes the requested business topic."
            )
        ),
    ),
    Rubric(
        rubric_id="caveats_and_limitations",
        rubric_content=RubricContent(
            text_property=(
                "The response clearly communicates material caveats or "
                "limitations present in the analysis results."
            )
        ),
    ),
    Rubric(
        rubric_id="recommendation_support",
        rubric_content=RubricContent(
            text_property=(
                "Any recommendations are consistent with the analysis findings "
                "and are presented as suggestions rather than proven outcomes."
            )
        ),
    ),
]
_QUALITY_RUBRIC_IDS = frozenset(rubric.rubric_id for rubric in _QUALITY_RUBRICS)


class _LiveAutoRaterResponseParser(DefaultAutoRaterResponseParser):
    """Recover the final structured rubric block from judge reasoning output."""

    def parse(self, auto_rater_response: str) -> list[RubricResponse]:
        first_rubric_id = re.escape(_QUALITY_RUBRICS[0].rubric_id)
        first_rubric_pattern = re.compile(
            rf"(?im)^\s*(?:ID:\s*|\[id:\s*){first_rubric_id}\s*\]?\s*$"
        )
        starts = list(first_rubric_pattern.finditer(auto_rater_response))
        for start in reversed(starts):
            parsed = super().parse(auto_rater_response[start.start():])
            if self._is_complete(parsed):
                logger.warning(
                    "Parsed the final structured rubric block after ignoring "
                    "unstructured text from the judge model."
                )
                return parsed

        if starts:
            logger.warning(
                "Judge response included a rubric block, but it did not contain "
                "a complete set of parseable live rubric scores."
            )
            return []

        parsed = super().parse(auto_rater_response)
        return parsed if self._is_complete(parsed) else []

    @staticmethod
    def _is_complete(parsed: list[RubricResponse]) -> bool:
        rubric_ids = [score.rubric_id for score in parsed]
        return len(rubric_ids) == len(_QUALITY_RUBRIC_IDS) and set(rubric_ids) == (
            _QUALITY_RUBRIC_IDS
        )


class _LiveRubricBasedFinalResponseQualityV1Evaluator(
    RubricBasedFinalResponseQualityV1Evaluator
):
    def __init__(self, eval_metric: EvalMetric) -> None:
        super().__init__(eval_metric)
        self._auto_rater_response_parser = _LiveAutoRaterResponseParser()


def create_live_invocation(
    question: str,
    result: dict[str, Any],
    events: list[Event],
    evidence_context: dict[str, Any] | None = None,
) -> Invocation:
    """Create the ADK evaluation input from a completed live workflow turn."""
    invocation_events = [
        InvocationEvent(
            author=event.author or "unknown",
            content=event.content,
            grounding_metadata=event.grounding_metadata,
            usage_metadata=event.usage_metadata,
            model_version=event.model_version,
        )
        for event in events
        if event.content is not None
        and (
            not event.partial
            or event.usage_metadata is not None
            or any(
                part.function_call is not None or part.function_response is not None
                for part in event.content.parts
            )
        )
    ]
    user_input = question
    if evidence_context:
        context_text = json.dumps(
            {
                key: value
                for key, value in evidence_context.items()
                if key != "live_response_quality"
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        user_input = (
            f"User's question:\n{question}\n\n"
            "Trusted saved analysis results for evaluating the answer "
            "(context only; not part of the user's question):\n"
            f"{context_text}"
        )

    return Invocation(
        invocation_id=next(
            (event.invocation_id for event in events if event.invocation_id),
            uuid4().hex,
        ),
        user_content=types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        ),
        final_response=types.Content(
            role="model",
            parts=[types.Part(text=_format_final_response(result))],
        ),
        intermediate_data=InvocationEvents(invocation_events=invocation_events),
        creation_timestamp=time.time(),
    )


def _format_final_response(result: dict[str, Any]) -> str:
    followup = result.get("followup_answer")
    if isinstance(followup, str):
        return followup

    sections: list[str] = []
    insights = result.get("business_insights")
    if isinstance(insights, dict):
        key_finding = insights.get("key_finding")
        if key_finding:
            sections.append(f"Key finding:\n{key_finding}")
        for title, key in (
            ("Supporting evidence", "supporting_evidence"),
            ("Possible drivers", "possible_drivers"),
            ("Important caveats", "important_caveats"),
        ):
            values = insights.get(key, [])
            if values:
                sections.append(
                    f"{title}:\n" + "\n".join(f"- {value}" for value in values)
                )
    elif result.get("business_insights"):
        sections.append(str(result["business_insights"]))

    recommendations = result.get("recommendations")
    if isinstance(recommendations, dict):
        items = recommendations.get("recommendations", [])
        if items:
            lines = []
            for item in items:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('action', 'Suggested action')}: "
                        f"{item.get('rationale', '')}"
                    )
                else:
                    lines.append(f"- {item}")
            sections.append("Recommendations:\n" + "\n".join(lines))
        caveat = recommendations.get("overall_caveat")
        if caveat:
            sections.append(f"Recommendation caveat:\n{caveat}")
    elif recommendations:
        sections.append(str(recommendations))

    validation = result.get("dataset_validation", {})
    if isinstance(validation, dict) and validation.get("warnings"):
        sections.append(
            "Dataset warnings:\n"
            + "\n".join(f"- {warning}" for warning in validation["warnings"])
        )
    return "\n\n".join(sections) or json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )


async def _evaluate(invocation: Invocation) -> dict[str, Any]:
    criterion = RubricsBasedCriterion(
        threshold=LIVE_QUALITY_THRESHOLD,
        judge_model_options=JudgeModelOptions(
            judge_model=LIVE_QUALITY_JUDGE_MODEL,
            num_samples=1,
        ),
        rubrics=_QUALITY_RUBRICS,
    )
    metric = EvalMetric(
        metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
        threshold=LIVE_QUALITY_THRESHOLD,
        criterion=criterion,
    )
    evaluation = await _LiveRubricBasedFinalResponseQualityV1Evaluator(
        metric
    ).evaluate_invocations([invocation])
    return {
        "status": evaluation.overall_eval_status.name.lower(),
        "score": evaluation.overall_score,
        "threshold": LIVE_QUALITY_THRESHOLD,
        "judge_model": LIVE_QUALITY_JUDGE_MODEL,
        "rubric_scores": [
            score.model_dump(mode="json")
            for score in (evaluation.overall_rubric_scores or [])
        ],
    }


def _evaluate_sync(invocation: Invocation) -> dict[str, Any]:
    return asyncio.run(_evaluate(invocation))


@dataclass
class _EvaluationJob:
    future: Future[dict[str, Any]]
    outcome: dict[str, Any] | None = None


class QualityEvaluationManager(Protocol):
    def submit(self, invocation: Invocation) -> str:
        ...

    def get_status(self, evaluation_id: str) -> dict[str, Any]:
        ...


class LiveQualityEvaluationManager:
    """Runs final-response quality checks without blocking agent responses."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="live-quality-eval",
        )
        self._jobs: dict[str, _EvaluationJob] = {}
        self._lock = threading.Lock()

    def submit(self, invocation: Invocation) -> str:
        evaluation_id = uuid4().hex
        future = self._executor.submit(_evaluate_sync, invocation)
        with self._lock:
            self._jobs[evaluation_id] = _EvaluationJob(future=future)
        return evaluation_id

    def get_status(self, evaluation_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(evaluation_id)
            if job is None:
                return {"status": "unavailable"}
            if job.outcome is not None:
                return dict(job.outcome)
            if not job.future.done():
                return {"status": "pending"}
            try:
                job.outcome = job.future.result()
            except Exception:
                logger.exception(
                    "Live final-response quality evaluation %s failed.",
                    evaluation_id,
                )
                job.outcome = {
                    "status": "error",
                    "message": (
                        "The response was delivered, but its quality evaluation "
                        "failed. Check the application logs."
                    ),
                }
            return dict(job.outcome)


live_quality_evaluations = LiveQualityEvaluationManager()

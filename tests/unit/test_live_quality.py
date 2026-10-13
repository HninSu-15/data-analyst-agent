import asyncio
from types import SimpleNamespace
import time

from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.events import Event
from google.genai import types

import app.live_quality as live_quality


def test_create_live_invocation_keeps_final_tool_events_and_filters_partial_text():
    partial_text = Event(
        author="insight_agent",
        content=types.Content(
            role="model",
            parts=[types.Part(text="draft text")],
        ),
        partial=True,
    )
    tool_event = Event(
        author="sales_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="analyze_sales",
                        args={},
                    )
                )
            ],
        ),
    )

    invocation = live_quality.create_live_invocation(
        "Summarize sales.",
        {"followup_answer": "Revenue was $100."},
        [partial_text, tool_event],
    )

    assert invocation.user_content.parts[0].text == "Summarize sales."
    assert invocation.final_response.parts[0].text == "Revenue was $100."
    assert len(invocation.intermediate_data.invocation_events) == 1
    assert (
        invocation.intermediate_data.invocation_events[0].content.parts[0]
        .function_call.name
        == "analyze_sales"
    )


def test_create_live_invocation_includes_saved_analysis_as_separate_evidence():
    invocation = live_quality.create_live_invocation(
        "Does Yangon exceed the other regions combined?",
        {"followup_answer": "Yes, by $481.60."},
        [],
        evidence_context={
            "sales_analysis": {
                "by_region": {
                    "Yangon": 2370.0,
                    "Mandalay": 792.05,
                    "Mawlamyine": 660.9,
                    "Naypyidaw": 435.45,
                }
            },
            "live_response_quality": {"evaluation_id": "irrelevant"},
        },
    )

    user_input = invocation.user_content.parts[0].text
    assert "User's question:\nDoes Yangon exceed the other regions combined?" in user_input
    assert "Trusted saved analysis results" in user_input
    assert '"Yangon": 2370.0' in user_input
    assert '"Mandalay": 792.05' in user_input
    assert "irrelevant" not in user_input
    assert invocation.final_response.parts[0].text == "Yes, by $481.60."


def test_evaluate_uses_single_sample_and_the_configured_rubrics(monkeypatch):
    captured = {}

    class Evaluator:
        def __init__(self, metric):
            captured["metric"] = metric

        async def evaluate_invocations(self, invocations):
            captured["invocations"] = invocations
            return SimpleNamespace(
                overall_eval_status=SimpleNamespace(name="PASSED"),
                overall_score=0.875,
                overall_rubric_scores=[
                    RubricScore(
                        rubric_id="evidence_grounding",
                        score=1.0,
                        rationale="The values are supported by the tool output.",
                    )
                ],
            )

    monkeypatch.setattr(
        live_quality,
        "_LiveRubricBasedFinalResponseQualityV1Evaluator",
        Evaluator,
    )
    monkeypatch.setattr(
        live_quality,
        "LIVE_QUALITY_JUDGE_MODEL",
        "test-provider/judge",
    )

    invocation = live_quality.create_live_invocation(
        "Summarize sales.",
        {"followup_answer": "Revenue was $100."},
        [],
    )
    result = asyncio.run(live_quality._evaluate(invocation))

    criterion = captured["metric"].criterion
    assert captured["invocations"] == [invocation]
    assert criterion.judge_model_options.judge_model == "test-provider/judge"
    assert criterion.judge_model_options.num_samples == 1
    assert len(criterion.rubrics) == 4
    assert result["score"] == 0.875
    assert result["status"] == "passed"
    assert result["rubric_scores"][0]["rubric_id"] == "evidence_grounding"


def test_live_response_parser_recovers_complete_rubric_block_after_judge_reasoning():
    rubric_blocks = []
    for rubric in live_quality._QUALITY_RUBRICS:
        rubric_blocks.append(
            "\n".join(
                (
                    f"ID: {rubric.rubric_id}",
                    f"Property: {rubric.rubric_content.text_property}",
                    "Evidence: The response addresses this evaluation criterion.",
                    "Rationale: The answer satisfies the criterion.",
                    "Verdict: yes",
                )
            )
        )
    judge_response = (
        "Property 2: question_relevance\n"
        "Rationale: The response is relevant.\n"
        "Verdict: yes\n\n"
        "Need ensure all requested rubric criteria are addressed.\n\n"
        + "\n\n".join(rubric_blocks)
    )

    parsed = live_quality._LiveAutoRaterResponseParser().parse(judge_response)

    assert [score.rubric_id for score in parsed] == [
        rubric.rubric_id for rubric in live_quality._QUALITY_RUBRICS
    ]
    assert all(score.score == 1.0 for score in parsed)


def test_live_evaluation_manager_reports_background_failure(monkeypatch):
    def fail_evaluation(_invocation):
        raise RuntimeError("mocked judge provider failure")

    monkeypatch.setattr(live_quality, "_evaluate_sync", fail_evaluation)
    manager = live_quality.LiveQualityEvaluationManager()
    invocation = live_quality.create_live_invocation("Question", {}, [])

    evaluation_id = manager.submit(invocation)
    for _ in range(100):
        status = manager.get_status(evaluation_id)
        if status["status"] != "pending":
            break
        time.sleep(0.01)

    assert status["status"] == "error"
    assert "Check the application logs" in status["message"]

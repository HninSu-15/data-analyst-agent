from app.agents.customer_agent import customer_agent
from app.agents.data_tools import (
    analyze_customers,
    analyze_products,
    analyze_sales,
    profile_uploaded_dataset,
)
from app.agents.followup_agent import followup_agent
from app.agents.insight_agent import insight_agent
from app.agents.manager_agent import manager_agent
from app.agents.product_agent import product_agent
from app.agents.profiler_agent import profiler_agent
from app.agents.recommendation_agent import recommendation_agent
from app.agents.sales_agent import sales_agent
from app.agent import _normalize_analysis_outputs, _parse_structured_output
from app.config import DEEPSEEK_MODEL, LLM_MODEL, SAMPLE_DATA_PATH
from app.schemas.analysis_models import AnalysisPlan, BusinessInsights, Recommendations


class DummyToolContext:
    def __init__(self, state: dict):
        self.state = state


def test_all_phase_two_agents_are_adk_llm_agents_with_expected_state_keys():
    agents = [
        manager_agent,
        profiler_agent,
        sales_agent,
        product_agent,
        customer_agent,
        insight_agent,
        recommendation_agent,
        followup_agent,
    ]

    assert [agent.name for agent in agents] == [
        "manager_agent",
        "profiler_agent",
        "sales_agent",
        "product_agent",
        "customer_agent",
        "insight_agent",
        "recommendation_agent",
        "followup_agent",
    ]
    assert [agent.output_key for agent in agents] == [
        "analysis_goal",
        "data_profile",
        "sales_analysis",
        "product_analysis",
        "customer_analysis",
        "business_insights",
        "recommendations",
        "followup_answer",
    ]
    assert all(agent.model is LLM_MODEL for agent in agents)
    assert DEEPSEEK_MODEL.startswith("deepseek/")
    assert manager_agent.output_schema is None
    assert insight_agent.output_schema is None
    assert recommendation_agent.output_schema is None
    assert followup_agent.mode == "chat"


def test_data_tools_use_csv_path_from_session_state():
    context = DummyToolContext({"csv_path": str(SAMPLE_DATA_PATH)})

    assert profile_uploaded_dataset(context)["available"] is True
    assert analyze_sales(context)["available"] is True
    assert analyze_products(context)["available"] is True
    customer_result = analyze_customers(context)
    assert customer_result["available"] is True
    assert customer_result["metrics"]["unique_customers"] > 0
    assert "top_customers_by_spend" not in customer_result["metrics"]


def test_data_tools_report_missing_session_csv_path():
    result = analyze_sales(DummyToolContext({}))

    assert result["available"] is False
    assert "csv_path" in result["error"]


def test_structured_output_parser_validates_and_normalizes_agent_json():
    state = {
        "analysis_goal": (
            '```json\n{"analysis_goal":"Compare category revenue",'
            '"priority_dimensions":["category"],"relevant_analyses":["product"],'
            '"requested_period":null}\n```'
        )
    }

    _parse_structured_output(state, "analysis_goal", AnalysisPlan)

    assert state["analysis_goal"] == {
        "analysis_goal": "Compare category revenue",
        "priority_dimensions": ["category"],
        "relevant_analyses": ["product"],
        "requested_period": None,
    }


def test_structured_output_parser_extracts_json_surrounded_by_explanation():
    state = {
        "business_insights": (
            "Here are the findings:\n"
            '{"key_finding":"Sales were strongest in June",'
            '"supporting_evidence":["June revenue was highest"],'
            '"possible_drivers":[],"important_caveats":[]}\n'
            "I hope this helps."
        )
    }

    _parse_structured_output(state, "business_insights", BusinessInsights)

    assert state["business_insights"]["key_finding"] == "Sales were strongest in June"
    assert state["business_insights"]["supporting_evidence"] == ["June revenue was highest"]


def test_structured_output_parser_rejects_invalid_agent_json():
    state = {"business_insights": "not JSON"}

    try:
        _parse_structured_output(state, "business_insights", BusinessInsights)
    except ValueError as exc:
        assert "did not contain a valid JSON object" in str(exc)
    else:
        raise AssertionError("Invalid structured output must not be accepted.")


def test_malformed_structured_output_is_preserved_with_explicit_warning():
    raw_output = '{"key_finding":"Top category was Jeans", malformed}'
    result = {"business_insights": raw_output}

    _normalize_analysis_outputs(result)

    assert result["business_insights"] == raw_output
    issue = result["structured_output_errors"][0]
    assert issue["output_key"] == "business_insights"
    assert "did not contain a valid JSON object" in issue["error"]
    assert issue["raw_output_preserved"] is True


def test_structured_output_parser_explains_schema_mismatches():
    state = {
        "business_insights": {
            "key_finding": "Revenue increased.",
            "supporting_evidence": [],
            "possible_drivers": [],
        }
    }

    try:
        _parse_structured_output(state, "business_insights", BusinessInsights)
    except ValueError as exc:
        assert "important_caveats" in str(exc)
        assert "Field required" in str(exc)
    else:
        raise AssertionError("Missing required fields must not be accepted.")


def test_structured_recommendation_model_accepts_expected_shape():
    recommendation = Recommendations.model_validate(
        {
            "recommendations": [
                {
                    "action": "Review category stock levels",
                    "evidence": ["Category revenue fell in the latest period."],
                    "rationale": "Stock constraints could be one factor worth checking.",
                    "validation_needed": "Compare with inventory availability.",
                }
            ],
            "overall_caveat": None,
        }
    )

    assert recommendation.recommendations[0].action == "Review category stock levels"
    assert recommendation.recommendations[0].evidence == [
        "Category revenue fell in the latest period."
    ]


def test_recommendation_model_normalizes_legacy_string_evidence():
    recommendation = Recommendations.model_validate(
        {
            "recommendations": [
                {
                    "action": "Review category stock levels",
                    "evidence": "Category revenue fell in the latest period.",
                    "rationale": "Stock constraints could be one factor worth checking.",
                }
            ]
        }
    )

    assert recommendation.recommendations[0].evidence == [
        "Category revenue fell in the latest period."
    ]


def test_structured_output_parser_accepts_list_evidence_from_model():
    state = {
        "recommendations": {
            "recommendations": [
                {
                    "action": "Review outerwear availability",
                    "evidence": [
                        "Jeans revenue was 1,399.50.",
                        "Outerwear revenue was 1,002.75.",
                    ],
                    "rationale": "Check whether stock constrained category demand.",
                }
            ],
            "overall_caveat": None,
        }
    }

    _parse_structured_output(state, "recommendations", Recommendations)

    assert state["recommendations"]["recommendations"][0]["evidence"] == [
        "Jeans revenue was 1,399.50.",
        "Outerwear revenue was 1,002.75.",
    ]

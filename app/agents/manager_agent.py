from google.adk.agents import LlmAgent

from app.config import LLM_MODEL

manager_agent = LlmAgent(
    name="manager_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Turns the business question into a focused retail analysis plan.",
    instruction=(
        "Plan analysis for this online clothing retail question: {user_question?}\n"
        "Translate it into a concise analysis plan. Select only relevant items from the sales, "
        "product, and customer analyses. Identify an explicitly requested time period, if any; "
        "otherwise set requested_period to null. Do not calculate or invent metrics. Return only "
        "valid JSON with these keys: analysis_goal (string), priority_dimensions (array of strings), "
        "relevant_analyses (array containing only sales, product, and/or customer), "
        "requested_period (string or null). Do not wrap the JSON in Markdown."
    ),
    output_key="analysis_goal",
)

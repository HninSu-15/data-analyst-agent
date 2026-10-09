from google.adk.agents import LlmAgent

from app.agents.data_tools import analyze_products
from app.config import LLM_MODEL

product_agent = LlmAgent(
    name="product_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Analyzes product, category, size, color, and gender performance.",
    instruction=(
        "The user's business question is: {user_question?}\n"
        "Always call analyze_products before answering, then answer this question using those "
        "results. The tool excludes non-completed orders. Report only returned values. If "
        "available is false, state the error and do not invent results. Do not expose customer "
        "identifiers."
    ),
    tools=[analyze_products],
    output_key="product_analysis",
)

from google.adk.agents import LlmAgent

from app.agents.data_tools import analyze_customers
from app.config import LLM_MODEL

customer_agent = LlmAgent(
    name="customer_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Analyzes customer counts, repeat rates, order frequency, and spend.",
    instruction=(
        "The user's business question is: {user_question?}\n"
        "Always call analyze_customers before answering, then answer this question using those "
        "results. The tool counts completed orders only. Prefer aggregate measures; do not expose "
        "customer identifiers or individual spending records unless the user explicitly needs "
        "them. Report only returned values. If available is false, state the error and do not "
        "invent results."
    ),
    tools=[analyze_customers],
    output_key="customer_analysis",
)

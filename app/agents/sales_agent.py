from google.adk.agents import LlmAgent

from app.agents.data_tools import analyze_sales
from app.config import LLM_MODEL

sales_agent = LlmAgent(
    name="sales_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Analyzes completed-order sales totals, regions, and monthly trends.",
    instruction=(
        "The user's business question is: {user_question?}\n"
        "Always call analyze_sales before answering, then answer this question using those "
        "results. The tool excludes non-completed orders from revenue and unit totals. Report "
        "only values returned by the tool. If available is false, report the failure and do not "
        "invent results. Mention data-quality warnings when relevant."
    ),
    tools=[analyze_sales],
    output_key="sales_analysis",
)

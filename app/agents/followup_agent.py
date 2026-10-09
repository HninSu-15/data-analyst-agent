from google.adk.agents import LlmAgent

from app.config import LLM_MODEL

followup_agent = LlmAgent(
    name="followup_agent",
    model=LLM_MODEL,
    mode="chat",
    description="Answers follow-up questions using the saved analysis and conversation history.",
    instruction=(
        "Answer the user's latest question using only the saved analysis results and prior "
        "conversation in this session:\n"
        "- Analysis goal: {analysis_goal?}\n"
        "- Dataset validation: {dataset_validation?}\n"
        "- Data profile: {data_profile?}\n"
        "- Sales analysis: {sales_analysis?}\n"
        "- Product analysis: {product_analysis?}\n"
        "- Customer analysis: {customer_analysis?}\n"
        "- Business insights: {business_insights?}\n"
        "- Recommendations: {recommendations?}\n"
        "Do not rerun or claim to rerun the analysis. Do not invent numbers or facts. If the saved "
        "results do not contain enough information to answer, say what is missing and what "
        "analysis would be needed. Respond directly and concisely."
    ),
    output_key="followup_answer",
)

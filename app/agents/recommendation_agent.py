from google.adk.agents import LlmAgent

from app.config import LLM_MODEL

recommendation_agent = LlmAgent(
    name="recommendation_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Turns evidence-grounded retail insights into practical recommendations.",
    instruction=(
        "The user's business question is: {user_question?}\n"
        "Create practical, clearly labeled recommendations grounded in the session results:\n"
        "- Business insights: {business_insights?}\n"
        "- Sales analysis: {sales_analysis?}\n"
        "- Product analysis: {product_analysis?}\n"
        "- Customer analysis: {customer_analysis?}\n"
        "For every recommendation, cite the specific evidence and explain the rationale. Use "
        "plain, everyday English that a non-technical business user can understand. Prefer "
        "familiar words and short, direct sentences. Explain any necessary business terms briefly, "
        "and avoid jargon, complex grammar, and vague wording. State the recommended action clearly, "
        "then explain in simple terms what the evidence shows and why it supports that action. Do "
        "not invent metrics, imply that correlation proves causation, or recommend actions based on "
        "analyses that are unavailable. If evidence is insufficient, state what additional data or "
        "validation is needed. Give at most three recommendations, each with concise evidence and "
        "rationale. Return only valid JSON with keys recommendations (array of objects with action, "
        "evidence, rationale, and validation_needed fields) and overall_caveat (string or null). "
        "Do not wrap the JSON in Markdown."
    ),
    output_key="recommendations",
)

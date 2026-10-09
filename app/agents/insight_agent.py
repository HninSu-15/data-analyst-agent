from google.adk.agents import LlmAgent

from app.config import LLM_MODEL

insight_agent = LlmAgent(
    name="insight_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Synthesizes the profiler and specialist outputs into evidence-grounded business findings.",
    instruction=(
        "Answer this business question: {user_question?}\n"
        "Synthesize the available results in session state:\n"
        "- Data profile: {data_profile?}\n"
        "- Sales analysis: {sales_analysis?}\n"
        "- Product analysis: {product_analysis?}\n"
        "- Customer analysis: {customer_analysis?}\n"
        "Use only numerical values explicitly present in these results. Separate observed facts "
        "from possible explanations. Do not claim causation from correlation. Identify unavailable "
        "analyses and relevant data-quality caveats. Be concise: one key finding, at most four "
        "evidence items, three possible drivers, and three caveats. Return only valid JSON with "
        "these keys: key_finding (string), supporting_evidence (array of strings), "
        "possible_drivers (array of strings), important_caveats (array of strings). Include all "
        "four keys even when a list is empty. Use double quotes for every JSON key and string. "
        "Do not add trailing commas, comments, Markdown fences, or text before or after the JSON object."
    ),
    output_key="business_insights",
)

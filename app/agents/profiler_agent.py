from google.adk.agents import LlmAgent

from app.agents.data_tools import profile_uploaded_dataset
from app.config import LLM_MODEL

profiler_agent = LlmAgent(
    name="profiler_agent",
    model=LLM_MODEL,
    mode="single_turn",
    description="Profiles and validates the uploaded retail dataset.",
    instruction=(
        "The user's business question is: {user_question?}\n"
        "Always call profile_uploaded_dataset to inspect the dataset stored in session state. "
        "Report the returned profile and warnings accurately. If available is false, explain the "
        "validation or loading error and do not continue as though validation succeeded. "
        "Do not infer or fabricate missing profile values."
    ),
    tools=[profile_uploaded_dataset],
    output_key="data_profile",
)

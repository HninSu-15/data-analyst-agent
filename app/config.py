import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "sales.csv"
UPLOADS_DIR = DATA_DIR / "uploads"
load_dotenv(BASE_DIR / ".env")

SESSION_DB_PATH = Path(
    os.getenv("ADK_SESSION_DB_PATH", str(DATA_DIR / "adk_sessions.db"))
).expanduser().resolve()
ANALYSIS_USER_ID = os.getenv("ANALYSIS_USER_ID", "local-user").strip() or "local-user"

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-flash")
LLM_MODEL = LiteLlm(model=DEEPSEEK_MODEL)

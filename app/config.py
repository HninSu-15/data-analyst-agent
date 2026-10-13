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

LIVE_QUALITY_JUDGE_MODEL = os.getenv(
    "LIVE_QUALITY_JUDGE_MODEL",
    DEEPSEEK_MODEL,
).strip() or DEEPSEEK_MODEL
LIVE_QUALITY_THRESHOLD = float(os.getenv("LIVE_QUALITY_THRESHOLD", "0.7"))
if not 0.0 <= LIVE_QUALITY_THRESHOLD <= 1.0:
    raise ValueError("LIVE_QUALITY_THRESHOLD must be between 0 and 1.")
_live_quality_enabled = os.getenv(
    "LIVE_QUALITY_EVALUATION_ENABLED",
    "true",
).strip().lower()
if _live_quality_enabled not in {"true", "false", "1", "0", "yes", "no"}:
    raise ValueError(
        "LIVE_QUALITY_EVALUATION_ENABLED must be true, false, 1, 0, yes, or no."
    )
LIVE_QUALITY_EVALUATION_ENABLED = _live_quality_enabled in {"true", "1", "yes"}

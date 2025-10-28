import os
from dotenv import load_dotenv

load_dotenv()

# пути
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_pipeline_A_base__HGB.joblib")
LE_PATH    = os.getenv("LE_PATH", "models/label_encoder_risk.joblib")
DB_PATH    = os.getenv("DB_PATH", "db/agent_memory.sqlite")

MODEL_PATH = os.path.abspath(MODEL_PATH)
LE_PATH    = os.path.abspath(LE_PATH)
DB_PATH    = os.path.abspath(DB_PATH)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# веса финальной смеси
W_ML   = float(os.getenv("W_ML", 0.6))
W_PRIOR= float(os.getenv("W_PRIOR", 0.25))
W_LLM  = float(os.getenv("W_LLM", 0.15))
THRESH = float(os.getenv("THRESH", 0.5))

# llm
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

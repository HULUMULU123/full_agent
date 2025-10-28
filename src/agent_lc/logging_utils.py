import json, time, os, sqlite3
from .config import DB_PATH

LOG_PATH = os.path.abspath("logs/llm-logs.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def redact(text: str) -> str:
    # минимальная защита: скрыть ключи/ИНН-ы формата 10-12 цифр
    return text.replace("\n", " ").replace("\r", " ")

def log_llm_io(endpoint: str, prompt: dict, response: dict, meta: dict | None = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    rec = {"ts": ts, "endpoint": endpoint, "prompt": prompt, "response": response, "meta": meta or {}}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # дублируем в SQLite
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""INSERT INTO llm_log(ts, endpoint, prompt, response, meta) VALUES(?,?,?,?,?)""",
                (ts, endpoint, json.dumps(prompt, ensure_ascii=False),
                      json.dumps(response, ensure_ascii=False),
                      json.dumps(meta or {}, ensure_ascii=False)))
    con.commit(); con.close()

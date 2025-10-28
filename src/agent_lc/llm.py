# src/agent_lc/llm.py
import os
import json
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_gigachat import GigaChat

from .prompt_v3 import PROMPT_V3
from .logging_utils import log_llm_io

# -----------------------------
# ИНИЦИАЛИЗАЦИЯ GigaChat (OAuth)
# -----------------------------
# В .env ожидаются:
#   GIGACHAT_API_KEY   — base64(client_id:client_secret)
#   GIGACHAT_SCOPE     — по умолчанию 'GIGACHAT_API_PERS'
#   GIGACHAT_MODEL     — например 'GigaChat-2'
GIGACHAT_API_KEY = os.environ.get("GIGACHAT_API_KEY")
GIGACHAT_SCOPE   = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL   = os.environ.get("GIGACHAT_MODEL", "GigaChat-2")

def _get_access_token() -> str:
    """Запрашиваем access_token у NGW. verify=False — как в твоём примере (внутренняя среда)."""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": "2aba969c-a22a-4816-a652-393a756a96c1",
        "Authorization": f"Basic {GIGACHAT_API_KEY}",
    }
    payload = {"scope": GIGACHAT_SCOPE}
    resp = requests.post(url, headers=headers, data=payload, verify=False, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]

# ленивый синглтон LLM
_LLM = None
def _get_llm():
    global _LLM
    if _LLM is None:
        # Можно передать credentials=GIGACHAT_API_KEY (SDK сам получит токен),
        # но раз у тебя уже настроен отдельный OAuth — возьмём явный токен.
        access_token = _get_access_token()
        _LLM = GigaChat(
            credentials=GIGACHAT_API_KEY,          # access_token
            model=GIGACHAT_MODEL,
            top_p=0,
            timeout=120,
            verify_ssl_certs=False,
            temperature=0.0,
        )
    return _LLM

def _extract_json(text: str) -> dict:
    """Робастный парсинг JSON: пробуем целиком, затем вырезку от первого '{' до последней '}'."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise ValueError("LLM вернул не-JSON и подходящих скобок не найдено")
        return json.loads(text[s : e + 1])

def call_llm(rows):
    """
    Вызов GigaChat + логирование prompt/response.
    rows: список словарей (INPUT_DATA из промпта).
    return: dict вида {"overall_observation": "...", "transactions": [...]}
    """
    # 1) собираем сообщения
    messages = [
        SystemMessage(content=PROMPT_V3),
        HumanMessage(content=json.dumps({"INPUT_DATA": rows}, ensure_ascii=False)),
    ]

    # 2) вызов GigaChat
    llm = _get_llm()
    resp = llm.invoke(messages)
    text = getattr(resp, "content", "").strip()

    # 3) робастный JSON
    try:
        data = _extract_json(text)
    except Exception as e:
        # логируем даже ошибочные ответы
        log_llm_io(
            endpoint="gigachat.chat",
            prompt={"system": PROMPT_V3[:2000] + "...", "input_rows_sample": rows[:3], "input_len": len(rows)},
            response={"raw_text": text, "error": str(e)},
            meta={"model": GIGACHAT_MODEL, "ok": False},
        )
        # отдаём пустую структуру — пайплайн сам подставит фолбэк
        return {"overall_observation": "", "transactions": []}

    # 4) логирование нормального ответа
    log_llm_io(
        endpoint="gigachat.chat",
        prompt={"system": PROMPT_V3[:2000] + "...", "input_rows_sample": rows[:3], "input_len": len(rows)},
        response=data,
        meta={"model": GIGACHAT_MODEL, "ok": True},
    )
    return data

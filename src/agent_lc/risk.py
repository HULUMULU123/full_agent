# src/agent_lc/risk.py
import math
from typing import Dict, Any, Optional
from .config import W_ML, W_PRIOR, W_LLM, THRESH


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_prior(hist: Dict[str, Any], row: Dict[str, Any]):
    """
    PRIOR из памяти: сигмоида поверх истории.

    Используем:
      - susp_rate / cnt_susp   — как и раньше
      - recency                — давность активности
      - amount_outlier         — крупность относительно p95
      - (опц.) llm_soft_rate   — мягкий вклад от LLM-красных, если такие счётчики есть в памяти
                                 (если колонок ещё нет — вклад = 0, код НЕ ломается)
    """
    def sigmoid(x): return 1/(1+math.exp(-x))

    susp_rate = max(hist.get("debit_susp_rate", 0.0), hist.get("credit_susp_rate", 0.0))
    cnt_susp  = max(hist.get("debit_cnt_suspicious", 0.0), hist.get("credit_cnt_suspicious", 0.0))
    last_days = min(hist.get("debit_last_seen_days", 1e6), hist.get("credit_last_seen_days", 1e6))
    recency   = max(0.0, 30.0 - float(last_days)) / 30.0

    # мягкий вклад от LLM-флагов (если ты начал их писать в память; иначе будет 0)
    llm_soft_d = float(hist.get("debit_llm_flags_total", 0.0) or 0.0)
    llm_soft_c = float(hist.get("credit_llm_flags_total", 0.0) or 0.0)
    llm_soft   = max(llm_soft_d, llm_soft_c)
    llm_soft_rate = min(1.0, llm_soft / 5.0)  # каждые ~5 красных LLM → до +1.0 (очень мягко)

    # крупность относительно p95
    amt  = float(row.get("amount") or 0.0)
    p95d = hist.get("debit_p95"); p95c = hist.get("credit_p95")
    amount_outlier = 0.0
    for p95 in (p95d, p95c):
        try:
            if p95 and float(p95) > 0 and amt > float(p95):
                amount_outlier = max(amount_outlier, (amt / float(p95)) - 1.0)
        except Exception:
            continue

    # логистическая регрессия «на глаз» (как была), с мягким добавлением llm_soft_rate
    z = 3.0 * susp_rate + 0.8 * math.log1p(cnt_susp) + 1.2 * recency + 0.4 * llm_soft_rate + 0.7 * amount_outlier - 1.5
    p_prior = sigmoid(z)

    return p_prior, dict(
        susp_rate=susp_rate,
        cnt_susp=cnt_susp,
        recency=recency,
        amount_outlier=amount_outlier,
        llm_soft_rate=llm_soft_rate,
        z=z
    )


def apply_hard_rules(row: Dict[str, Any], hist: Dict[str, Any]):
    """Жёсткие правила, пробивающие решение вверх при срабатывании."""
    hits = []
    amt = float(row.get("amount") or 0.0)
    sr  = max(hist.get("debit_susp_rate", 0.0), hist.get("credit_susp_rate", 0.0))
    cs  = max(hist.get("debit_cnt_suspicious", 0.0), hist.get("credit_cnt_suspicious", 0.0))
    last_days = min(hist.get("debit_last_seen_days", 1e6), hist.get("credit_last_seen_days", 1e6))
    p95 = max(hist.get("debit_p95") or 0.0, hist.get("credit_p95") or 0.0)
    watchlisted = bool(hist.get("debit_watchlisted") or hist.get("credit_watchlisted"))
    big_amount  = (p95 > 0 and amt > p95)

    if watchlisted and big_amount and last_days <= 14:
        hits.append("R1_watchlist_big_recent")
    if (sr >= 0.40 and cs >= 10 and last_days <= 30 and big_amount):
        hits.append("R2_heavy_history_big_recent")

    return (len(hits) > 0), hits


def label_to_prob(label: str, confidence: float | None = None) -> float:
    """
    Переводим метку LLM в вероятность. Если LLM вернул явный score — используем его (в [0,1]).
    Иначе по эвристике: красный→1.0, жёлтый→0.5, зелёный→0.2
    """
    if confidence is not None:
        try:
            v = float(confidence)
            return max(0.0, min(1.0, v))
        except Exception:
            pass
    lab = (label or "").strip().lower()
    if "красн" in lab or "risk" in lab or "подоз" in lab:
        return 1.0
    if "желт" in lab or "medium" in lab:
        return 0.5
    return 0.2


def llm_hint_floor(row: Dict[str, Any], p_llm: float) -> Optional[float]:
    """
    «Мягкий порог» (floor) для случаев, когда LLM видит красное, а остальные критерии — слабые.
    Возвращает минимальный уровень p_final, ниже которого опускаться не даём.
    None — если floor не нужен.
    """
    if p_llm < 0.99:
        return None  # floor только для «уверенного» красного от LLM

    purpose = (row.get("purpose") or "").lower()
    amount  = float(row.get("amount") or 0.0)
    cl, cd  = row.get("chain_length"), row.get("chain_duration_hours")

    has_high_kw = any(w in purpose for w in [
        "займ", "возврат займа", "перевод на карту", "крипто", "биткоин", "usdt",
        "swift", "иностранный перевод", "выдача наличных", "обналичивание",
        "агентское вознаграждение", "комиссионное"
    ])
    round_amount = (amount % 10000 == 0) or (amount % 100000 == 0)
    transit_short = False
    try:
        if cd is not None:
            cdv = float(cd)
            transit_short = (cdv < 24) and (cl is None or int(cl) >= 3 or True)
    except Exception:
        pass

    anomaly_purpose = float(row.get("anomaly_purpose") or 0.0) >= 0.6

    # если есть хоть слабое подтверждение — поднимем выше
    if has_high_kw or round_amount or transit_short or anomaly_purpose:
        return 0.45   # почти-жёлтая зона
    return 0.35       # базовый floor: пометим как требующий внимания


def mix_final(p_ml: float, p_prior: float, p_llm: float, hard_hit: bool,
              llm_floor: Optional[float] = None):
    """
    Финальная смесь рисков.
      - линейная композиция p_ml/p_prior/p_llm с весами из .config
      - мягкий floor от LLM (если есть)
      - жёсткие правила пробивают вверх до «красного»
    """
    # базовая линейная смесь
    p_lin = W_ML * p_ml + W_PRIOR * p_prior + W_LLM * p_llm

    # применяем «мягкий порог» от LLM, если он выше линейной смеси
    if llm_floor is not None and llm_floor > p_lin:
        p_lin = llm_floor

    # обрезка в [0, 1]
    p_final = max(0.0, min(1.0, p_lin))

    # жёсткие правила — повышаем минимум до красного
    if hard_hit:
        p_final = max(p_final, 0.70)

    is_suspicious = bool(hard_hit or (p_final >= THRESH))
    label = "красный" if p_final >= 0.70 else ("желтый" if p_final >= 0.40 else "зеленый")
    return p_final, is_suspicious, label

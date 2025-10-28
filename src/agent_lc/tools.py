# src/agent_lc/tools.py
import json
from io import StringIO
import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import Dict, Any, List

from .memory import combine_hist_for_row, mem_upsert_after_decision
from .risk import compute_prior, apply_hard_rules, label_to_prob, mix_final, llm_hint_floor
from .llm import call_llm


# ─────────────────────────────
# Доменные хелперы (флаги/объяснения)
# ─────────────────────────────
HIGH_WORDS = {
    "займ","договор займа","возврат займа","взаиморасчёт","перевод средств","без договора",
    "перевод на карту","личные нужды","крипто","биткоин","usdt","биржа","coin","crypto",
    "swift","иностранный перевод","валютный счёт","экспорт","передача активов",
    "пополнение","наличные","выдача наличных","обналичивание",
    "благотворительность","пожертвование","агентское вознаграждение","комиссионное"
}

def _is_round(amount):
    try:
        a = float(amount)
        return (a % 10000 == 0) or (a % 100000 == 0)
    except Exception:
        return False

def _flags_from_row(r: dict) -> list[str]:
    flags = []
    cl, cd = r.get("chain_length"), r.get("chain_duration_hours")
    if cd is not None and float(cd) < 0.01:
        flags.append("transit_very_short")
    if (cl is not None and cd is not None and int(cl) >= 3 and float(cd) < 24) or (cd is not None and float(cd) < 24):
        flags.append("transit_short")
    if float(r.get("anomaly_amount", 0) or 0) >= 0.6:
        flags.append("amount_anomaly_strong")
    if float(r.get("anomaly_frequency", 0) or 0) >= 0.6:
        flags.append("freq_anomaly_strong")
    if float(r.get("anomaly_purpose", 0) or 0) >= 0.6:
        flags.append("purpose_anomaly")
    if _is_round(r.get("amount")):
        flags.append("round_large_amount")
    p = str(r.get("purpose","")).lower()
    if any(w in p for w in HIGH_WORDS):
        flags.append("purpose_stopword_high")
    # memory-* подсказки (из памяти контрагентов)
    if r.get("debit_watchlisted") or r.get("credit_watchlisted"):
        flags.append("memory_watchlist")
    if float(r.get("debit_susp_rate",0))>=0.4 or float(r.get("credit_susp_rate",0))>=0.4:
        flags.append("memory_high_susp_rate")
    if float(r.get("debit_cnt_suspicious",0))>=10 or float(r.get("credit_cnt_suspicious",0))>=10:
        flags.append("memory_many_past_flags")
    if float(r.get("debit_last_seen_days",1e6))<30 or float(r.get("credit_last_seen_days",1e6))<30:
        flags.append("memory_recent_activity")
    if (r.get("debit_p95") and r.get("amount") and float(r["amount"])>float(r["debit_p95"])) or \
       (r.get("credit_p95") and r.get("amount") and float(r["amount"])>float(r["credit_p95"])):
        flags.append("memory_above_p95")
    return flags

def _reasons_from_row(r: dict) -> list[str]:
    out=[]
    mm = r.get("ml_metric", None)
    if mm is not None:
        try: out.append(f"ml_metric={round(float(mm or 0), 2)}")
        except: pass
    aa = float(r.get("anomaly_amount",0) or 0)
    if aa>=0.6: out.append(f"аномалия суммы={aa:.2f}")
    af = float(r.get("anomaly_frequency",0) or 0)
    if af>=0.6: out.append(f"аномалия частоты={af:.2f}")
    ap = float(r.get("anomaly_purpose",0) or 0)
    if ap>=0.6: out.append(f"аномалия назначения={ap:.2f}")
    if _is_round(r.get("amount")): out.append("круглая сумма")
    cl, cd = r.get("chain_length"), r.get("chain_duration_hours")
    try:
        if cl is not None and cd is not None and int(cl)>=3 and float(cd)<24:
            out.append("короткая транзитная цепочка")
    except: pass
    return out[:5]

def _merge_flags_and_reasons(t: dict, base: dict) -> dict:
    gen_flags = _flags_from_row(base)
    gen_reas  = _reasons_from_row(base)
    t.setdefault("flags", [])
    t.setdefault("primary_reasons", [])
    if not t["flags"]:
        t["flags"] = gen_flags
    else:
        seen = set(t["flags"])
        t["flags"].extend([f for f in gen_flags if f not in seen])
    if not t["primary_reasons"]:
        t["primary_reasons"] = gen_reas
    else:
        seen = set(t["primary_reasons"])
        t["primary_reasons"].extend([r for r in gen_reas if r not in seen])
    return t

_BAD_OK_PHRASES = (
    "не вызывает подозрений", "не является подозрительной", "типичная операция",
    "признаков риска не обнаружено", "риск низкий", "безопасна", "стандартная операция"
)

def _enforce_text_consistency(tr: dict) -> dict:
    """Если текст противоречит финальной метке, заменяем на шаблоны по метке."""
    lbl = (tr.get("risk_label") or "").lower()
    rec = (tr.get("recommendation") or "").lower()
    expl = (tr.get("risk_explanation") or "").lower()

    def has_ok_phrases(s: str) -> bool:
        return any(p in s for p in _BAD_OK_PHRASES)

    if lbl == "красный":
        # текст обязан быть «жёстким», убираем любые «всё ок»
        if has_ok_phrases(rec) or not tr.get("recommendation"):
            tr["recommendation"] = (
                "Свяжитесь с клиентом и запросите подтверждающие документы. "
                "При необходимости приостановите проведение операции."
            )
        if has_ok_phrases(expl) or not tr.get("risk_explanation"):
            tr["risk_explanation"] = (
                "Выявлены признаки повышенного риска по назначению/контрагентам/сумме. "
                "Операция выглядит подозрительной и требует проверки."
            )

    elif lbl == "желтый":
        # умеренно настороженный тон
        if has_ok_phrases(rec) or not tr.get("recommendation"):
            tr["recommendation"] = (
                "Проверьте назначение платежа и документы. "
                "Дополнительно оцените историю контрагента."
            )
        if has_ok_phrases(expl) or not tr.get("risk_explanation"):
            tr["risk_explanation"] = (
                "Есть отдельные настораживающие факторы, но без явных нарушений. "
                "Требуется уточняющая проверка."
            )

    else:  # зелёный
        # явно сообщаем, что подозрений нет
        if not tr.get("recommendation"):
            tr["recommendation"] = "Храните документы по операции. Мониторинг без дополнительных действий."
        if not tr.get("risk_explanation") or not has_ok_phrases(expl):
            tr["risk_explanation"] = (
                "Операция выглядит типичной для клиента и не вызывает подозрений. "
                "Назначение и сумма в обычных пределах."
            )

    return tr

def _fill_missing(tr: dict) -> dict:
    def _short_expl(t):
        fl  = t.get("flags", [])[:3]
        prs = t.get("primary_reasons", [])[:2]
        parts=[]
        if fl:  parts.append(", ".join(fl))
        if prs: parts.append("; ".join(prs))
        return (" ".join(parts)).strip() or "Операция выглядит типичной для клиента. Признаков риска не обнаружено."
    if not tr.get("risk_explanation"):
        tr["risk_explanation"] = _short_expl(tr)
    if not tr.get("recommendation"):
        lbl = tr.get("risk_label","зеленый")
        rec = {
            "красный": "Свяжитесь с клиентом. Запросите документы и при необходимости приостановите операцию.",
            "желтый":  "Проверьте документы и назначение платежа. Проведите дополнительную проверку контрагента.",
            "зеленый": "Храните документы по операции. Мониторинг без дополнительных действий."
        }.get(lbl, "Проведите проверку документов и назначение платежа.")
        tr["recommendation"] = rec
    ev = tr.setdefault("evidence", {})
    for k in ["ml_metric","anomaly_amount","amount","chain_match","chain_length","chain_duration_hours"]:
        ev.setdefault(k, None)
    return tr


# ─────────────────────────────
# Технические хелперы (приведение типов/округление)
# ─────────────────────────────
def _to_float(x):
    try:
        if pd.isna(x):  # NaN/NA -> None
            return None
        return float(x)
    except Exception:
        return None

def _to_int_or_none(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None

def _to_iso(x):
    """Timestamp/datetime -> ISO string; иначе None/str как есть."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        if isinstance(x, str):
            return x
        ts = pd.to_datetime(x, errors="coerce")
        return ts.isoformat() if pd.notna(ts) else (str(x) if x is not None else None)
    except Exception:
        return str(x) if x is not None else None

def _to_jsonable(v):
    """Приводит numpy/scalars/TS к json-совместимым типам."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return _to_iso(v)
    return v

def _round2(x):
    try:
        return None if x is None else round(float(x), 2)
    except Exception:
        return None


# ─────────────────────────────
# TOOL: сбор payload для LLM (с памятью)
# ─────────────────────────────
@tool("build_llm_payload", return_direct=True)
def build_llm_payload_tool(df_json: str) -> str:
    """Вход: JSON df (records). Выход: обогащённые строки для LLM (с памятью)."""
    df = pd.read_json(StringIO(df_json), orient="records")
    df = df.reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for i, r in df.iterrows():
        rid = _to_int_or_none(r.get("id")) or (i + 1)

        # компактный, токено-экономный Payload
        row = {
            "id": rid,
            "purpose": str(r.get("purpose", ""))[:300],   # ограничим длину
            "ml_metric": _round2(r.get("ml_metric")),
            "anomaly_amount": _round2(r.get("anomaly_amount")),
            "anomaly_frequency": _round2(r.get("anomaly_frequency")),
            "anomaly_purpose": _round2(r.get("anomaly_purpose")),
            "anomaly_overall": _round2(r.get("anomaly_overall")),
            "is_regular_payment": bool(r.get("is_regular_payment", False)),
            "debit_name_type": str(r.get("debit_name_type", "Прочее")),
            "credit_name_type": str(r.get("credit_name_type", "Прочее")),
            "debit_amount": _round2(r.get("debit_amount")),
            "credit_amount": _round2(r.get("credit_amount")),
            "amount": _round2(r.get("amount")),
            "debit_inn": "" if pd.isna(r.get("debit_inn")) else str(r.get("debit_inn")),
            "credit_inn": "" if pd.isna(r.get("credit_inn")) else str(r.get("credit_inn")),
            "chain_match": None if pd.isna(r.get("chain_id")) else str(r.get("chain_id")),
            "chain_length": _to_int_or_none(r.get("chain_length")),
            "chain_duration_hours": _round2(r.get("chain_duration_hours")),
            # TS → ISO (короткий вид до секунд)
            "ts": (_to_iso(r.get("ts") or r.get("date")) or "")[:19],
        }

        # 🔶 ПАМЯТЬ: подмешиваем историю контрагентов (SQLite)
        hist = combine_hist_for_row(row)
        # оставляем только нужные поля, округляем
        keep = {
            "debit_susp_rate","debit_cnt_suspicious","debit_last_seen_days","debit_watchlisted","debit_p95",
            "credit_susp_rate","credit_cnt_suspicious","credit_last_seen_days","credit_watchlisted","credit_p95"
        }
        hist = {k: _round2(v) if isinstance(v,(int,float)) else v for k,v in hist.items() if k in keep}

        # если есть причины от ML — шлём, но не засоряем пустым
        ml_reasons = r.get("ml_top_reasons", [])
        if isinstance(ml_reasons, list) and ml_reasons:
            row["ml_top_reasons"] = ml_reasons[:5]

        # финальный jsonable row
        row = {**{k: _to_jsonable(v) for k, v in row.items()}, **{k: _to_jsonable(v) for k, v in hist.items()}}
        rows.append(row)

    return json.dumps({"transactions": rows, "input_len": len(rows)}, ensure_ascii=False)


# ─────────────────────────────
# TOOL: вызов LLM + смешивание с ML/Prior/Rules + лог в память
# ─────────────────────────────
@tool("llm_assess_risk", return_direct=True)
def llm_assess_risk_tool(enriched_rows_json: str) -> str:
    """Вход: JSON enriched rows. Выход: финальные транзакции (ML+prior+LLM+rules) + лог в память."""
    payload = json.loads(enriched_rows_json) if isinstance(enriched_rows_json, str) else enriched_rows_json
    rows_enriched: List[Dict[str, Any]] = payload.get("transactions", [])

    # 1) Вспомогательная оценка LLM + объяснения (устойчиво)
    try:
        data = call_llm(rows_enriched)
        tx = data.get("transactions", [])
    except Exception:
        # 🔁 Fallback: если LLM оборвался/ошибка — считаем без LLM, чтобы отчёт не был пустым
        tx = []
        for base in rows_enriched:
            p_ml = float(base.get("ml_metric", 0.0) or 0.0)
            hist = {
                "debit_cnt_suspicious": base.get("debit_cnt_suspicious", 0.0),
                "debit_susp_rate": base.get("debit_susp_rate", 0.0),
                "debit_last_seen_days": base.get("debit_last_seen_days", 1e6),
                "debit_watchlisted": base.get("debit_watchlisted", 0),
                "debit_p95": base.get("debit_p95"),
                "credit_cnt_suspicious": base.get("credit_cnt_suspicious", 0.0),
                "credit_susp_rate": base.get("credit_susp_rate", 0.0),
                "credit_last_seen_days": base.get("credit_last_seen_days", 1e6),
                "credit_watchlisted": base.get("credit_watchlisted", 0),
                "credit_p95": base.get("credit_p95"),
            }
            p_prior, _ = compute_prior(hist, base)
            p_llm = 0.2  # консервативно зелёный
            hard_hit, rule_ids = apply_hard_rules(base, hist)
            p_final, is_suspicious, label = mix_final(p_ml, p_prior, p_llm, hard_hit)

            t = {
                "id": int(base.get("id")),
                "purpose": base.get("purpose",""),
                "risk_label": label,
                "risk_score": float(round(p_final, 2)),
                "flags": [],
                "primary_reasons": [],
                "evidence": {}
            }
            # флаги/причины + evidence + тексты
            t = _merge_flags_and_reasons(t, base)
            ev = t.get("evidence", {})
            ev.update({"ml_metric": p_ml, "prior": round(p_prior,3), "p_llm": round(p_llm,3), "p_final": round(p_final,3)})
            t["evidence"]=ev; t["rule_hits"]=rule_ids
            t = _fill_missing(t)
            # пишем в память
            mem_upsert_after_decision(
                dict(id=int(base.get("id")), ts=base.get("ts"),
                     debit_inn=base.get("debit_inn"), credit_inn=base.get("credit_inn"),
                     amount=base.get("amount"), purpose=base.get("purpose")),
                dict(p_ml=p_ml, p_prior=p_prior, p_llm=p_llm, p_final=p_final,
                     label_pred=label, is_suspicious=is_suspicious,
                     rule_hits=rule_ids, reasons_llm=t.get("primary_reasons", []))
            )
            tx.append({k: _to_jsonable(v) for k, v in t.items()})

        return json.dumps({"overall_observation": "", "transactions": tx}, ensure_ascii=False)

    # 2) Основной путь: есть ответ LLM → смешиваем и логируем
    final_tx: List[Dict[str, Any]] = []
    for t in tx:
        rid = _to_int_or_none(t.get("id"))
        base = next((r for r in rows_enriched if _to_int_or_none(r.get("id")) == rid), {})

        p_ml = float(base.get("ml_metric", 0.0) or 0.0)
        hist = {
            "debit_cnt_suspicious": base.get("debit_cnt_suspicious", 0.0),
            "debit_susp_rate": base.get("debit_susp_rate", 0.0),
            "debit_last_seen_days": base.get("debit_last_seen_days", 1e6),
            "debit_watchlisted": base.get("debit_watchlisted", 0),
            "debit_p95": base.get("debit_p95"),
            "credit_cnt_suspicious": base.get("credit_cnt_suspicious", 0.0),
            "credit_susp_rate": base.get("credit_susp_rate", 0.0),
            "credit_last_seen_days": base.get("credit_last_seen_days", 1e6),
            "credit_watchlisted": base.get("credit_watchlisted", 0),
            "credit_p95": base.get("credit_p95"),
        }

        # prior / llm / правила
        p_prior, _ = compute_prior(hist, base)
        p_llm = label_to_prob(t.get("risk_label"), t.get("risk_score"))
        hard_hit, rule_ids = apply_hard_rules(base, hist)

        # 🔸 LLM-floor: если LLM «красный», не опускаем итог ниже мягкого порога
        floor_hint = llm_hint_floor(base, p_llm)

        # финальная смесь
        p_final, is_suspicious, label = mix_final(p_ml, p_prior, p_llm, hard_hit, llm_floor=floor_hint)

        # итоговые поля → чтобы _fill_missing видел финальную метку
        t["risk_label"] = label
        t["risk_score"] = float(round(p_final, 2))
        t["rule_hits"] = rule_ids

        # автодобавим флаги/причины
        t = _merge_flags_and_reasons(t, base)

        # evidence + компоненты
        ev = t.get("evidence", {}) or {}
        ev.update({
            "ml_metric": p_ml,
            "prior": round(p_prior, 3),
            "p_llm": round(p_llm, 3),
            "p_final": round(p_final, 3),
        })
        t["evidence"] = ev

        # заполнить пустые тексты (учитывает финальную метку)
        t = _fill_missing(t)

        t = _enforce_text_consistency(t)

        # 🔶 ЛОГ В ПАМЯТЬ (и идемпотентные агрегаты будут пересчитаны в memory.py)
        mem_upsert_after_decision(
            dict(id=rid,
                 ts=base.get("ts"),
                 debit_inn=base.get("debit_inn"),
                 credit_inn=base.get("credit_inn"),
                 amount=base.get("amount"),
                 purpose=base.get("purpose")),
            dict(p_ml=p_ml, p_prior=p_prior, p_llm=p_llm, p_final=p_final,
                 label_pred=label, is_suspicious=is_suspicious,
                 rule_hits=rule_ids, reasons_llm=t.get("primary_reasons", []))
        )

        # json-совместимость на выходе
        final_tx.append({k: _to_jsonable(v) for k, v in t.items()})

    return json.dumps(
        {"overall_observation": data.get("overall_observation", ""), "transactions": final_tx},
        ensure_ascii=False
    )

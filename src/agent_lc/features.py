import pandas as pd
import numpy as np

HIGH_RISK_WORDS = [
    "займ","договор займа","возврат займа","взаиморасчёт","перевод средств","без договора","перевод на карту",
    "личные нужды","крипто","биткоин","usdt","биржа","coin","crypto","swift","иностранный перевод",
    "валютный счёт","экспорт","передача активов","пополнение","наличные","выдача наличных","обналичивание",
    "благотворительность","пожертвование","агентское вознаграждение","комиссионное"
]
MEDIUM_RISK_WORDS = [
    "оплата услуг","услуги по договору","консультационные","маркетинг","премия","бонус","вознаграждение",
    "аванс","предоплата","частичная оплата","аренда","лизинг","субаренда","логистика","транспорт","перевозка","доставка"
]

def extract_type(name: str) -> str:
    if pd.isna(name): return "Прочее"
    s = str(name)
    if s.startswith(("ООО","АО","ОАО","ЗАО")): return "ЮЛ"
    if s.startswith("ИП"): return "ИП"
    if s.startswith("ФЛ"): return "ФЛ"
    return "Прочее"

def has_any(text: str, words) -> int:
    t = "" if pd.isna(text) else str(text).lower()
    return int(any(w in t for w in words))

def _is_round(x) -> int:
    try:
        a = float(x)
        return int((a % 10000 == 0) or (a % 100000 == 0))
    except Exception:
        return 0

def _purpose_group(text: str) -> str:
    t = "" if pd.isna(text) else str(text).lower()
    if any(w in t for w in HIGH_RISK_WORDS): return "high_kw"
    if any(w in t for w in MEDIUM_RISK_WORDS): return "med_kw"
    return "low_kw"

def build_base_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # гарантируем нужные столбцы
    need = ["id","date","debit_account","debit_name","debit_inn",
            "credit_account","credit_name","credit_inn","debit_amount","credit_amount","purpose"]
    for c in need:
        if c not in df.columns:
            df[c] = None

    # amount
    if "credit_amount" in df.columns:
        df["amount"] = pd.to_numeric(df["credit_amount"], errors="coerce")
    else:
        df["amount"] = np.nan
    if "debit_amount" in df.columns:
        da = pd.to_numeric(df["debit_amount"], errors="coerce").abs()
        df["amount"] = df["amount"].fillna(0.0)
        df["amount"] = np.where((df["amount"] == 0.0) & da.notna(), da, df["amount"])
    df["amount"] = df["amount"].fillna(0.0).astype(float)

    # типы контрагентов
    df["debit_name_type"]  = df["debit_name"].map(extract_type)
    df["credit_name_type"] = df["credit_name"].map(extract_type)

    # ключевые слова
    df["purpose_kw_high"] = df["purpose"].apply(lambda x: has_any(x, HIGH_RISK_WORDS))
    df["purpose_kw_med"]  = df["purpose"].apply(lambda x: has_any(x, MEDIUM_RISK_WORDS))
    df["is_regular_payment"] = 0

    # заглушки по цепочкам (если нет вычисления цепочек)
    if "chain_id" not in df.columns: df["chain_id"] = None
    if "chain_length" not in df.columns: df["chain_length"] = None
    if "chain_duration_hours" not in df.columns: df["chain_duration_hours"] = None

    # базовые аномалии
    df["anomaly_amount"]    = 0.0
    df["anomaly_purpose"]   = df["purpose_kw_high"].astype(float)
    df["anomaly_frequency"] = 0.0
    df["anomaly_overall"]   = df[["anomaly_amount","anomaly_purpose","anomaly_frequency"]].max(axis=1)

    # ───────────────────────────────────────────────
    # ДОП. ПРИЗНАКИ, которых требует сохранённый Pipeline
    # ───────────────────────────────────────────────

    # date → datetime
    df["_dt"] = pd.to_datetime(df.get("date"), errors="coerce")

    # dow/hour
    df["dow"]  = df["_dt"].dt.dayofweek.fillna(-1).astype(int)
    df["hour"] = df["_dt"].dt.hour.fillna(-1).astype(int)

    # round_amount
    df["round_amount"] = df["amount"].apply(_is_round).astype(int)

    # purpose_group
    df["purpose_group"] = df["purpose"].apply(_purpose_group)

    # transit_like (короткая транзитная цепочка / быстрое прохождение)
    cl = df.get("chain_length")
    cd = df.get("chain_duration_hours")
    if cl is not None and cd is not None:
        df["transit_like"] = (
            ((pd.to_numeric(cl, errors="coerce").fillna(0) >= 3) &
             (pd.to_numeric(cd, errors="coerce").fillna(1e9) < 24)) |
            (pd.to_numeric(cd, errors="coerce").fillna(1e9) < 24)
        ).astype(int)
    elif cd is not None:
        df["transit_like"] = (pd.to_numeric(cd, errors="coerce").fillna(1e9) < 24).astype(int)
    else:
        df["transit_like"] = 0

    # уборка технич. колонки
    df.drop(columns=["_dt"], inplace=True)

    # id в int (если строка)
    try:
        df["id"] = df["id"].astype(int)
    except Exception:
        pass

    return df

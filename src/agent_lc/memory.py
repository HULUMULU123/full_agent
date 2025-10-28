# src/agent_lc/memory.py
import os, sqlite3, json, time
from typing import Dict, Any, Iterable
from .config import DB_PATH

# ─────────────────────────────────────────────────────────────────────────────
# INIT: создаём БД/таблицы и добавляем недостающие колонки (если схема обновилась)
# ─────────────────────────────────────────────────────────────────────────────
def mem_init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS tx (
      tx_id TEXT PRIMARY KEY,
      ts TEXT,
      debit_inn TEXT,
      credit_inn TEXT,
      amount REAL,
      purpose TEXT
    );

    CREATE TABLE IF NOT EXISTS decisions (
      tx_id TEXT PRIMARY KEY,
      p_ml REAL,
      p_prior REAL,
      p_llm REAL,
      p_final REAL,
      label_pred TEXT,
      is_suspicious INTEGER,
      rule_hits TEXT,
      reasons_llm TEXT,
      inserted_at TEXT
    );

    CREATE TABLE IF NOT EXISTS agg_counterparty (
      inn TEXT PRIMARY KEY,
      cnt_total REAL,
      cnt_suspicious REAL,
      susp_rate REAL,
      amt_total REAL,
      amt_suspicious REAL,
      last_seen_ts TEXT,
      watchlisted INTEGER DEFAULT 0,
      p50 REAL, p75 REAL, p90 REAL, p95 REAL
    );

    CREATE TABLE IF NOT EXISTS llm_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      endpoint TEXT,
      prompt TEXT,
      response TEXT,
      meta TEXT
    );
    """)
    # — добавляем мягкие поля для LLM-флагов, если их ещё нет
    try:
        cur.execute("ALTER TABLE agg_counterparty ADD COLUMN llm_flags_total REAL DEFAULT 0;")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE agg_counterparty ADD COLUMN llm_last_seen_ts TEXT;")
    except sqlite3.OperationalError:
        pass

    # индексы для скорости (idempotent)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_ts ON tx(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_debit ON tx(debit_inn);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_credit ON tx(credit_inn);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_label ON decisions(label_pred);")
    except sqlite3.OperationalError:
        pass

    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_select(query: str, params=()):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        r = cur.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        con.close()
        mem_init()
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        r = cur.execute(query, params).fetchone()
    finally:
        con.close()
    return r


def days_since(ts_str: str, now_ts: float = None) -> float:
    if not ts_str:
        return 1e6
    if now_ts is None:
        now_ts = time.time()
    # принимаем форматы "YYYY-mm-dd HH:MM:SS" или ISO
    try:
        # короткий формат
        import datetime as _dt
        try:
            dt = _dt.datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            # ISO / прочие — доверимся pandas при наличии
            from pandas import to_datetime
            dt = to_datetime(ts_str, errors="coerce").to_pydatetime()
            if dt is None:
                return 30.0
        delta = now_ts - dt.timestamp()
        return max(0.0, delta / 86400.0)
    except Exception:
        return 30.0


# ─────────────────────────────────────────────────────────────────────────────
# READ: агрегаты по контрагенту (в т.ч. мягкие LLM-счётчики)
# ─────────────────────────────────────────────────────────────────────────────
def mem_read_counterparty(inn: str) -> Dict[str, Any]:
    r = _safe_select("""
      SELECT cnt_total,cnt_suspicious,susp_rate,amt_total,amt_suspicious,
             last_seen_ts,watchlisted,p50,p75,p90,p95,
             llm_flags_total,llm_last_seen_ts
      FROM agg_counterparty WHERE inn=?""", (inn,))
    if not r:
        return dict(
            cnt_total=0, cnt_suspicious=0, susp_rate=0.0,
            amt_total=0.0, amt_suspicious=0.0,
            last_seen_ts=None, watchlisted=0,
            p50=None, p75=None, p90=None, p95=None,
            llm_flags_total=0.0, llm_last_seen_ts=None
        )
    keys = [
        "cnt_total","cnt_suspicious","susp_rate","amt_total","amt_suspicious",
        "last_seen_ts","watchlisted","p50","p75","p90","p95",
        "llm_flags_total","llm_last_seen_ts"
    ]
    return dict(zip(keys, r))


def combine_hist_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """🔶 ОБРАЩЕНИЕ К ПАМЯТИ: объединённые агрегаты по дебету/кредиту."""
    h_d = mem_read_counterparty(row.get("debit_inn") or "")
    h_c = mem_read_counterparty(row.get("credit_inn") or "")
    return {
        # дебет
        "debit_cnt_total": h_d["cnt_total"],
        "debit_cnt_suspicious": h_d["cnt_suspicious"],
        "debit_susp_rate": h_d["susp_rate"],
        "debit_last_seen_days": days_since(h_d["last_seen_ts"]),
        "debit_watchlisted": h_d["watchlisted"],
        "debit_p95": h_d["p95"],
        "debit_llm_flags_total": h_d.get("llm_flags_total", 0.0),
        "debit_llm_last_seen_days": days_since(h_d.get("llm_last_seen_ts")) if h_d.get("llm_last_seen_ts") else 1e6,

        # кредит
        "credit_cnt_total": h_c["cnt_total"],
        "credit_cnt_suspicious": h_c["cnt_suspicious"],
        "credit_susp_rate": h_c["susp_rate"],
        "credit_last_seen_days": days_since(h_c["last_seen_ts"]),
        "credit_watchlisted": h_c["watchlisted"],
        "credit_p95": h_c["p95"],
        "credit_llm_flags_total": h_c.get("llm_flags_total", 0.0),
        "credit_llm_last_seen_days": days_since(h_c.get("llm_last_seen_ts")) if h_c.get("llm_last_seen_ts") else 1e6,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WRITE: логирование решения и обновление агрегатов (вкл. мягкие LLM-флаги)
# ─────────────────────────────────────────────────────────────────────────────
def mem_upsert_after_decision(row: Dict[str, Any], decision: Dict[str, Any]):
    """
    ЛОГ + ИДЕМПОТЕНТНЫЕ АГРЕГАТЫ:
      - decisions: upsert по tx_id
      - agg_counterparty: пересчёт из факта (tx/decisions), без ручного инкремента
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 0) стабильный tx_id ----------
    tx_id = str(row.get("id"))
    ts    = str(row.get("ts") or now)
    debit = row.get("debit_inn")
    credit= row.get("credit_inn")
    amount= float(row.get("amount") or 0.0)
    purpose = row.get("purpose")

    # ---------- 1) сырые транзакции ----------
    cur.execute("""INSERT OR IGNORE INTO tx(tx_id,ts,debit_inn,credit_inn,amount,purpose)
                   VALUES(?,?,?,?,?,?)""", (tx_id, ts, debit, credit, amount, purpose))

    # ---------- 2) решения ----------
    cur.execute("""INSERT OR REPLACE INTO decisions
                   (tx_id,p_ml,p_prior,p_llm,p_final,label_pred,is_suspicious,rule_hits,reasons_llm,inserted_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (tx_id,
                 float(decision.get("p_ml", 0.0)),
                 float(decision.get("p_prior", 0.0)),
                 float(decision.get("p_llm", 0.0)),
                 float(decision.get("p_final", 0.0)),
                 str(decision.get("label_pred", "")),
                 int(bool(decision.get("is_suspicious", False))),
                 json.dumps(decision.get("rule_hits", []), ensure_ascii=False),
                 json.dumps(decision.get("reasons_llm", []), ensure_ascii=False),
                 now))

    # обновим агрегаты по затронутым ИНН
    for role in ("debit", "credit"):
        _recalc_for_inn(cur, row.get(f"{role}_inn"))

    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# NEW: предзагрузка всей выписки (до LLM)
# ─────────────────────────────────────────────────────────────────────────────
def mem_bulk_preload_statement(df_like) -> None:
    """
    Вставляет ВСЕ строки выписки в tx (id, ts/date, debit_inn, credit_inn, amount, purpose)
    и пересчитывает agg_counterparty по всем встреченным ИНН.
    Используется ДО LLM, чтобы PRIOR/квантили/last_seen учитывали всю таблицу.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Собираем строки для вставки
    inns = set()
    rows_to_insert = []
    for _, r in getattr(df_like, "iterrows", lambda: [])():
        tx_id = str(r.get("id"))
        ts = str(r.get("ts") or r.get("date") or "")
        debit = r.get("debit_inn") or ""
        credit = r.get("credit_inn") or ""
        amount = float(r.get("amount") or r.get("credit_amount") or r.get("debit_amount") or 0.0)
        purpose = r.get("purpose") or ""

        rows_to_insert.append((tx_id, ts, debit, credit, amount, purpose))
        if debit:
            inns.add(str(debit))
        if credit:
            inns.add(str(credit))

    # Вставим пачкой (idempotent)
    cur.executemany("""INSERT OR IGNORE INTO tx(tx_id,ts,debit_inn,credit_inn,amount,purpose)
                       VALUES(?,?,?,?,?,?)""", rows_to_insert)

    # Пересчёт агрегатов по всем встреченным ИНН
    for inn in inns:
        _recalc_for_inn(cur, inn)

    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Внутренний пересчёт агрегатов по ИНН (используется и в upsert, и в bulk)
# ─────────────────────────────────────────────────────────────────────────────
def _recalc_for_inn(cur: sqlite3.Cursor, inn: str):
    if not inn:
        return

    # все транзакции по ИНН (как дебит, так и кредит)
    cur.execute("""
        SELECT t.tx_id, t.amount, t.ts
          FROM tx t
         WHERE t.debit_inn = ? OR t.credit_inn = ?
    """, (inn, inn))
    rows = cur.fetchall()

    cnt_total = len(rows)
    amt_total = sum((r[1] or 0.0) for r in rows)
    last_seen_ts = None
    if rows:
        last_seen_ts = max((r[2] for r in rows if r[2]), default=None)

    # присоединим решения (могут быть не для всех tx_id — это нормально до LLM)
    tx_ids = [r[0] for r in rows]
    cnt_suspicious = 0
    amt_suspicious = 0.0
    llm_flags_total = 0

    if tx_ids:
        q_marks = ",".join(["?"] * len(tx_ids))
        cur.execute(f"""
            SELECT d.tx_id, d.is_suspicious, d.p_llm
              FROM decisions d
             WHERE d.tx_id IN ({q_marks})
        """, tx_ids)
        for d_tx_id, is_susp, p_llm in cur.fetchall():
            is_susp = int(bool(is_susp))
            if is_susp:
                amt = next((r[1] for r in rows if r[0] == d_tx_id), 0.0)
                amt_suspicious += float(amt or 0.0)
                cnt_suspicious += 1
            try:
                if float(p_llm or 0.0) >= 0.99:
                    llm_flags_total += 1
            except Exception:
                pass

    susp_rate = (cnt_suspicious / cnt_total) if cnt_total > 0 else 0.0

    # Квантили p50/p75/p90/p95 (по всем суммам этого ИНН)
    p50 = p75 = p90 = p95 = None
    amounts = [float(r[1] or 0.0) for r in rows if r[1] is not None]
    if amounts:
        try:
            import numpy as _np
            p50 = float(_np.percentile(amounts, 50))
            p75 = float(_np.percentile(amounts, 75))
            p90 = float(_np.percentile(amounts, 90))
            p95 = float(_np.percentile(amounts, 95))
        except Exception:
            pass

    # watchlisted: сохраняем текущее значение (если есть), не затираем
    cur.execute("SELECT watchlisted FROM agg_counterparty WHERE inn=?", (inn,))
    r_watch = cur.fetchone()
    watch = int(r_watch[0]) if r_watch and r_watch[0] is not None else 0

    # UPSERT агрегатов (идемпотентный)
    cur.execute("""
        INSERT INTO agg_counterparty
            (inn, cnt_total, cnt_suspicious, susp_rate, amt_total, amt_suspicious,
             last_seen_ts, watchlisted, p50, p75, p90, p95, llm_flags_total, llm_last_seen_ts)
        VALUES (?,  ?,         ?,              ?,         ?,          ?,
                ?,            ?,          ?,   ?,   ?,   ?,   ?,               ?)
        ON CONFLICT(inn) DO UPDATE SET
            cnt_total       = excluded.cnt_total,
            cnt_suspicious  = excluded.cnt_suspicious,
            susp_rate       = excluded.susp_rate,
            amt_total       = excluded.amt_total,
            amt_suspicious  = excluded.amt_suspicious,
            last_seen_ts    = excluded.last_seen_ts,
            p50             = excluded.p50,
            p75             = excluded.p75,
            p90             = excluded.p90,
            p95             = excluded.p95,
            llm_flags_total = excluded.llm_flags_total,
            llm_last_seen_ts= excluded.llm_last_seen_ts,
            watchlisted     = watchlisted  -- не трогаем вручную помеченный флаг
    """, (inn, cnt_total, cnt_suspicious, susp_rate, amt_total, amt_suspicious,
          last_seen_ts, watch, p50, p75, p90, p95, llm_flags_total, last_seen_ts))

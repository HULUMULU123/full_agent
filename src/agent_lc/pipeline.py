# src/agent_lc/pipeline.py
import json
import time
import pandas as pd
from langchain_core.runnables import RunnableSequence

from .memory import mem_init, mem_bulk_preload_statement
from .features import build_base_features
from .model import load_artifacts, predict_with_pipeline
from .tools import build_llm_payload_tool, llm_assess_risk_tool
from .export import export_excel_report

# ─────────────────────────────────────────────────────────────
# try/except для красивого прогресса
# ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm  # optional
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def _read_csv_robust(path: str) -> pd.DataFrame:
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin1"):
        for sep in (",", ";", "\t", "|"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                tried.append(f"{enc}/{repr(sep)} -> {e.__class__.__name__}")
                continue
    return pd.read_csv(path, engine="python", encoding_errors="replace", sep=None)


def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index + 1
    else:
        s = pd.to_numeric(df["id"], errors="coerce")
        # заполним NaN последовательностью 1..N
        fill_seq = pd.Series(range(1, len(df) + 1), index=df.index)
        s = s.fillna(fill_seq)
        s[s <= 0] = fill_seq[s <= 0]
        df["id"] = s.astype(int)
    return df


def run_pipeline(csv_path: str, out_xlsx: str, llm_batch_size: int = 10, verbose: bool = True) -> dict:
    # 1) Память/БД
    mem_init()

    # 2) Данные
    df_raw = _read_csv_robust(csv_path)

    # 3) Признаки
    df_prep = build_base_features(df_raw)

    # 4) Модель
    pipe, _ = load_artifacts()
    df_scored = predict_with_pipeline(pipe, df_prep)
    df_scored = _ensure_ids(df_scored)

    # 4.5) 🔶 ПРЕДЗАГРУЗКА ВСЕЙ ВЫПИСКИ В ПАМЯТЬ (tx + agg_counterparty)
    # Это нужно, чтобы PRIOR/квантили/last_seen уже учитывали всю таблицу до LLM.
    mem_bulk_preload_statement(df_scored)

    # 5) Оркестрация LLM ПО БАТЧАМ (как было)
    chain = RunnableSequence(first=build_llm_payload_tool, last=llm_assess_risk_tool)

    all_records = json.loads(df_scored.to_json(orient="records", force_ascii=False))
    merged_tx = []

    if not all_records:
        raise RuntimeError("Нет данных после подготовки признаков.")

    total = len(all_records)
    total_batches = (total + llm_batch_size - 1) // llm_batch_size

    # ── Индикатор прогресса ───────────────────────────────────
    start_ts = time.time()
    if verbose:
        if _HAS_TQDM:
            pbar = tqdm(total=total, desc=f"LLM batches (size={llm_batch_size})", unit="tx")
        else:
            print(f"[LLM] Запуск по пакетам: всего {total} транзакций, "
                  f"batch_size={llm_batch_size}, batches={total_batches}")

    processed = 0
    for bi in range(0, total, llm_batch_size):
        batch_records = all_records[bi:bi + llm_batch_size]
        batch_idx = bi // llm_batch_size + 1

        # Индикатор (plain)
        if verbose and not _HAS_TQDM:
            elapsed = time.time() - start_ts
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"  - пакет {batch_idx}/{total_batches} "
                  f"(rows {bi+1}..{bi+len(batch_records)}), "
                  f"готово {processed}/{total} | {rate:.1f} tx/s")

        # Передаём в первый tool именно подмножество
        enriched_json = chain.invoke(json.dumps(batch_records, ensure_ascii=False))
        # Надёжный парс (оба варианта принимаем)
        part = json.loads(enriched_json) if isinstance(enriched_json, str) else enriched_json
        got = len(part.get("transactions", []))
        merged_tx.extend(part.get("transactions", []))

        # обновим прогресс
        processed += len(batch_records)
        if verbose and _HAS_TQDM:
            pbar.update(len(batch_records))
            pbar.set_postfix_str(f"batch={batch_idx}/{total_batches}, got={got}")

    if verbose and _HAS_TQDM:
        pbar.close()
    if verbose and not _HAS_TQDM:
        elapsed = time.time() - start_ts
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(f"[LLM] Готово: {processed}/{total} за {elapsed:.1f}s ({rate:.1f} tx/s)")

    llm_resp = {"overall_observation": "", "transactions": merged_tx}

    # 6) Excel
    xlsx = export_excel_report(df_scored, llm_resp, out_xlsx)

    # 7) Сводка
    lbls = [t.get("risk_label") for t in llm_resp.get("transactions", [])]
    summary = {
        "red":    sum(1 for x in lbls if x == "красный"),
        "yellow": sum(1 for x in lbls if x in ("желтый", "жёлтый")),
        "green":  sum(1 for x in lbls if x in ("зеленый", "зелёный")),
        "total":  len(lbls),
    }
    return {"xlsx": xlsx, "summary": summary}

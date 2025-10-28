import os, glob, joblib
import pandas as pd
import numpy as np
from .config import MODEL_PATH, LE_PATH

def load_artifacts():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        cands = sorted(glob.glob("models/best_pipeline_*.joblib")) or sorted(glob.glob("best_pipeline_*.joblib"))
        assert cands, "Не найден сохранённый Pipeline (*.joblib)."
        model_path = cands[0]
    pipe = joblib.load(model_path)
    le = joblib.load(LE_PATH) if os.path.exists(LE_PATH) else None
    return pipe, le

def predict_with_pipeline(pipe, df_prep: pd.DataFrame) -> pd.DataFrame:
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(df_prep)
            ml_metric = proba[:, -1].astype(float) if proba is not None and proba.shape[1] >= 2 else np.clip(pipe.predict(df_prep).astype(float), 0, 1)
        except Exception:
            ml_metric = np.clip(pipe.predict(df_prep).astype(float), 0, 1)
    else:
        ml_metric = np.clip(pipe.predict(df_prep).astype(float), 0, 1)
    df_out = df_prep.copy()
    df_out["ml_metric"] = ml_metric
    return df_out

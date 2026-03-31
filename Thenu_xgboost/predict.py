"""
Prediction helper for the saved model in this folder.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)


def predict(input_dict: dict) -> dict:
    try:
        df = pd.DataFrame([input_dict])
        df["Mileage_Per_Year"] = df["Millage(KM)"] / (df["Car_Age"] + 1)
        df["Engine_Category"] = pd.cut(
            df["Engine (cc)"],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=["Small", "Medium", "Large", "XLarge", "Luxury"],
        )
        df["Comfort_Score"] = (
            pd.to_numeric(df["AIR CONDITION"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER STEERING"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER MIRROR"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER WINDOW"], errors="coerce").fillna(0)
        )
        prediction = model.predict(df)[0]
        if metadata.get("use_log_transform", False):
            prediction = np.expm1(prediction)
        return {"status": "success", "predicted_price": round(float(prediction), 2)}
    except Exception as exc:
        return {"status": "error", "errors": [str(exc)]}

from __future__ import annotations

import joblib
import pandas as pd

from research.src.explainability import top_shap_contributors
from research.src.utils import DISEASES, dataset_path, model_path


def load_model(disease_id: str):
    return joblib.load(model_path(disease_id))


def predict_sample(disease_id: str, features: dict[str, float | str]) -> dict:
    artifact = load_model(disease_id)
    pipeline = artifact["pipeline"]
    threshold = artifact.get("threshold", 0.5)
    spec = DISEASES[disease_id]
    frame = pd.DataFrame([features])
    probability = float(pipeline.predict_proba(frame)[0][1])
    prediction = int(probability >= threshold)
    background = pd.read_csv(dataset_path(disease_id)).drop(columns=[spec.target]).head(100)
    contributors = top_shap_contributors(pipeline, frame, background)
    return {
        "disease": spec.display_name,
        "prediction": prediction,
        "risk_probability": round(probability, 4),
        "threshold": threshold,
        "model": artifact.get("model_name", "unknown"),
        "top_contributors": contributors,
    }

from pathlib import Path
from threading import Lock
from typing import Dict, List

import joblib
import pandas as pd


FEATURE_NAMES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class DiabetesPredictionService:
    def __init__(self, model_dir: str = "ML-MODEL/XGBoost"):
        self.model_dir = Path(model_dir)
        self._model = None
        self._scaler = None
        self._threshold = None
        self._lock = Lock()

    def predict(self, features: Dict[str, float]) -> Dict[str, float | int | str]:
        self._load_artifacts()

        values = pd.DataFrame(
            [[features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
            dtype=float,
        )
        scaled_values = self._scaler.transform(values)

        probability = float(self._model.predict_proba(scaled_values)[0][1])
        threshold = float(self._threshold)
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "risk_probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "risk_label": "diabetes_risk_detected" if prediction else "low_diabetes_risk",
        }

    def _load_artifacts(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            self._model = joblib.load(self.model_dir / "xgboost_diabetes_model.pkl")
            self._scaler = joblib.load(self.model_dir / "scaler.pkl")
            self._threshold = joblib.load(self.model_dir / "threshold.pkl")


diabetes_prediction_service = DiabetesPredictionService()

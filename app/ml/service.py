from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd

from app.ml.explainability import shap_contributions
from app.ml.registry import DISEASE_REGISTRY, DiseaseConfig, feature_names, get_disease_config


DISCLAIMER = (
    "This is educational screening support from an ML model and RAG assistant, not a medical diagnosis. "
    "Please consult a qualified healthcare professional."
)


def confidence_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability <= 0.70:
        return "Moderate"
    return "High"


def reported_probability(probability: float) -> float:
    """Avoid publishing visually absolute probabilities while preserving threshold behavior."""
    return round(min(max(probability, 0.0), 0.999), 4)


def explanation_estimator(model):
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        return getattr(calibrated[0], "estimator", model)
    return model


class DiseasePredictionService:
    def __init__(self):
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def list_diseases(self) -> list[dict[str, Any]]:
        return [self._disease_metadata(config) for config in DISEASE_REGISTRY.values()]

    def metrics(self, disease_id: str) -> dict[str, Any]:
        config = get_disease_config(disease_id)
        metrics_path = config.artifact_dir / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        if disease_id == "diabetes" and self._legacy_available():
            return {
                "selected_model": "legacy_xgboost",
                "note": "Legacy diabetes artifacts are available. Run scripts/train_disease_models.py for publication metrics.",
            }
        raise FileNotFoundError(f"No metrics artifact found for {disease_id}.")

    def predict(self, disease_id: str, features: dict[str, float]) -> dict[str, Any]:
        config = get_disease_config(disease_id)
        names = feature_names(config)
        frame = pd.DataFrame([[features[name] for name in names]], columns=names, dtype=float)

        if (config.artifact_dir / "best_model.joblib").exists():
            artifacts = self._load_artifacts(config)
            model = artifacts["model"]
            probability = float(model.predict_proba(frame)[0][1])
            threshold = float(artifacts["metrics"].get("threshold", 0.5))
            prediction = int(probability >= threshold)
            explainer_model = explanation_estimator(model)
            shap_values = shap_contributions(
                explainer_model,
                frame,
                background=artifacts.get("shap_background"),
            )
            if not shap_values:
                shap_values = self._importance_fallback(artifacts["metrics"], frame)

            output_probability = reported_probability(probability)
            return {
                "disease_id": disease_id,
                "prediction": prediction,
                "risk_label": "risk_detected" if prediction else "low_risk",
                "risk_probability": output_probability,
                "confidence_level": confidence_level(output_probability),
                "threshold": round(threshold, 4),
                "selected_model": artifacts["metrics"].get("selected_model", "unknown"),
                "top_contributors": shap_values,
                "artifact_version": artifacts["metadata"].get("created_at", "unknown"),
                "disclaimer": DISCLAIMER,
            }

        if disease_id == "diabetes":
            return self._predict_legacy_diabetes(frame)

        raise FileNotFoundError(f"No trained artifact found for {disease_id}.")

    def _load_artifacts(self, config: DiseaseConfig) -> dict[str, Any]:
        disease_id = config.disease_id
        if disease_id in self._artifacts:
            return self._artifacts[disease_id]

        with self._lock:
            if disease_id in self._artifacts:
                return self._artifacts[disease_id]
            artifacts = {
                "model": joblib.load(config.artifact_dir / "best_model.joblib"),
                "metrics": json.loads((config.artifact_dir / "metrics.json").read_text(encoding="utf-8")),
                "metadata": json.loads((config.artifact_dir / "metadata.json").read_text(encoding="utf-8")),
            }
            background_path = config.artifact_dir / "shap_background.csv"
            if background_path.exists():
                artifacts["shap_background"] = pd.read_csv(background_path)
            self._artifacts[disease_id] = artifacts
            return artifacts

    def _disease_metadata(self, config: DiseaseConfig) -> dict[str, Any]:
        trained = (config.artifact_dir / "best_model.joblib").exists()
        legacy = config.disease_id == "diabetes" and self._legacy_available()
        return {
            "disease_id": config.disease_id,
            "display_name": config.display_name,
            "trained": trained or legacy,
            "artifact_dir": str(config.artifact_dir),
            "features": [feature.__dict__ for feature in config.feature_specs],
            "provenance": config.provenance,
            "artifact_type": "publication_pipeline" if trained else ("legacy" if legacy else "missing"),
        }

    def _importance_fallback(self, metrics: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
        importance = metrics.get("feature_importance", {})
        ranked = sorted(importance.items(), key=lambda item: abs(float(item[1])), reverse=True)
        values = frame.iloc[0].to_dict()
        return [
            {
                "feature": feature,
                "value": round(float(values.get(feature, 0.0)), 4),
                "contribution": round(float(score), 6),
            }
            for feature, score in ranked[:5]
        ]

    def _legacy_available(self) -> bool:
        model_dir = Path("ML-MODEL/XGBoost")
        return all(
            (model_dir / name).exists()
            for name in ("xgboost_diabetes_model.pkl", "scaler.pkl", "threshold.pkl")
        )

    def _predict_legacy_diabetes(self, frame: pd.DataFrame) -> dict[str, Any]:
        model_dir = Path("ML-MODEL/XGBoost")
        model = joblib.load(model_dir / "xgboost_diabetes_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        threshold = float(joblib.load(model_dir / "threshold.pkl"))
        scaled_values = scaler.transform(frame)
        probability = float(model.predict_proba(scaled_values)[0][1])
        prediction = int(probability >= threshold)

        output_probability = reported_probability(probability)
        return {
            "disease_id": "diabetes",
            "prediction": prediction,
            "risk_label": "diabetes_risk_detected" if prediction else "low_diabetes_risk",
            "risk_probability": output_probability,
            "confidence_level": confidence_level(output_probability),
            "threshold": round(threshold, 4),
            "selected_model": "legacy_xgboost",
            "top_contributors": [],
            "artifact_version": "legacy",
            "disclaimer": DISCLAIMER,
        }


disease_prediction_service = DiseasePredictionService()

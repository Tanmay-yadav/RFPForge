import json
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.diseases import router as diseases_router, validate_disease_context
from app.dependencies import get_llm_service, get_retrieval_service
from app.ml.preprocessing import DiabetesFeatureEngineer
from app.ml.registry import DISEASE_REGISTRY, DiseaseConfig, FeatureSpec
from app.ml.service import DiseasePredictionService
from app.ml.training import _make_pipeline


def make_artifact_dir(name: str) -> Path:
    path = Path(".test_artifacts") / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_diabetes_feature_engineering_adds_expected_features_and_nan_rules():
    transformer = DiabetesFeatureEngineer()
    data = pd.DataFrame(
        [
            {
                "Pregnancies": 1,
                "Glucose": 0,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 31.0,
                "DiabetesPedigreeFunction": 0.3,
                "Age": 40,
            }
        ]
    )

    transformed = transformer.fit_transform(data)

    assert pd.isna(transformed.loc[0, "Glucose"])
    assert "BMI_Category" in transformed.columns
    assert "Glucose_Age_Ratio" in transformed.columns
    assert "Insulin_Glucose_Ratio" in transformed.columns
    assert "BMI_Age_Product" in transformed.columns


def test_training_pipeline_places_smote_inside_pipeline():
    imblearn = pytest.importorskip("imblearn")
    pipeline = _make_pipeline(model=object())

    assert isinstance(pipeline, imblearn.pipeline.Pipeline)
    assert [name for name, _ in pipeline.steps] == ["features", "imputer", "scaler", "smote", "model"]


class DummyModel:
    def predict_proba(self, frame):
        return [[0.2, 0.8]]


def test_prediction_service_loads_publication_artifact(monkeypatch):
    artifact_dir = make_artifact_dir("dummy")
    joblib.dump(DummyModel(), artifact_dir / "best_model.joblib")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "selected_model": "dummy",
                "feature_importance": {"a": 0.9, "b": 0.1},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps({"created_at": "test"}), encoding="utf-8")

    config = DiseaseConfig(
        disease_id="dummy",
        display_name="Dummy Disease",
        dataset_path=artifact_dir / "dummy.csv",
        target_column="target",
        positive_labels=(1,),
        feature_specs=(FeatureSpec("a", "A", default=1.0), FeatureSpec("b", "B", default=2.0)),
        provenance={"source": "test"},
        artifact_dir=artifact_dir,
    )
    monkeypatch.setitem(DISEASE_REGISTRY, "dummy", config)

    result = DiseasePredictionService().predict("dummy", {"a": 1.0, "b": 2.0})

    assert result["prediction"] == 1
    assert result["risk_probability"] == 0.8
    assert result["selected_model"] == "dummy"
    assert result["top_contributors"][0]["feature"] == "a"


def test_diseases_router_predict_contract_with_mocked_artifact(monkeypatch):
    from fastapi import FastAPI

    artifact_dir = make_artifact_dir("dummy_api")
    joblib.dump(DummyModel(), artifact_dir / "best_model.joblib")
    (artifact_dir / "metrics.json").write_text(
        json.dumps({"threshold": 0.5, "selected_model": "dummy", "feature_importance": {"a": 1.0}}),
        encoding="utf-8",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps({"created_at": "test"}), encoding="utf-8")

    config = DiseaseConfig(
        disease_id="dummy_api",
        display_name="Dummy API Disease",
        dataset_path=artifact_dir / "dummy.csv",
        target_column="target",
        positive_labels=(1,),
        feature_specs=(FeatureSpec("a", "A", default=1.0),),
        provenance={"source": "test"},
        artifact_dir=artifact_dir,
    )
    monkeypatch.setitem(DISEASE_REGISTRY, "dummy_api", config)

    class Retrieval:
        def search(self, query, top_k=5):
            return [{"content": "medical context"}]

    class LLM:
        def stream(self, prompt):
            yield "mock advice"

    app = FastAPI()
    app.include_router(diseases_router)
    app.dependency_overrides[get_retrieval_service] = lambda: Retrieval()
    app.dependency_overrides[get_llm_service] = lambda: LLM()

    response = TestClient(app).post("/diseases/dummy_api/predict", json={"features": {"a": 1.0}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] == "dummy"
    assert payload["advice"] == "mock advice"


def test_breast_cancer_retrieval_validation_rejects_unrelated_ultrasound_context():
    results = [
        {
            "content": "Patient Guidance: Abdominal Ultrasound Procedure. Fasting may be required.",
            "score": 0.99,
            "metadata": {"source": "ultrasound"},
        },
        {
            "content": "Breast cancer evaluation may involve mammogram imaging and biopsy confirmation.",
            "score": 0.82,
            "metadata": {"source": "breast-cancer"},
        },
        {
            "content": "Breast cancer screening overview.",
            "score": 0.62,
            "metadata": {"source": "breast-cancer-low-score"},
        },
    ]

    accepted, validation = validate_disease_context("breast_cancer", results)

    assert [item["metadata"]["source"] for item in accepted] == ["breast-cancer"]
    assert validation["accepted"] == 1
    assert validation["rejected"] == 2

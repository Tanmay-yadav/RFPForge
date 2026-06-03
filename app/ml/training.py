from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.evaluation import choose_threshold, classification_metrics
from app.ml.explainability import model_feature_importance, save_shap_summary
from app.ml.preprocessing import DiabetesFeatureEngineer
from app.ml.registry import DiseaseConfig, feature_names, get_disease_config


RANDOM_STATE = 42


def _make_pipeline(model):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("features", DiabetesFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def _model_grids() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }


def _dataset_profile(data: pd.DataFrame, config: DiseaseConfig) -> dict[str, Any]:
    target = data[config.target_column]
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "target_column": config.target_column,
        "class_distribution": {str(key): int(value) for key, value in target.value_counts().to_dict().items()},
        "missing_values": {key: int(value) for key, value in data.isna().sum().to_dict().items()},
        "provenance": config.provenance,
    }


def _save_figures(
    artifact_dir: Path,
    data: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test,
    probabilities,
    threshold: float,
    pipeline,
    original_features: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures = artifact_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures / "correlation_heatmap.png", dpi=220)
    plt.close()

    predictions = (probabilities >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures / "confusion_matrix.png", dpi=220)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(figures / "roc_curve.png", dpi=220)
    plt.close()

    importance = model_feature_importance(pipeline, original_features)
    if importance:
        names = list(importance)[:15]
        values = [importance[name] for name in names]
        plt.figure(figsize=(9, 6))
        sns.barplot(x=values, y=names, color="#2563eb")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures / "feature_importance.png", dpi=220)
        plt.close()

    save_shap_summary(pipeline, X_test, figures / "shap_summary.png")


def train_disease_model(disease_id: str = "diabetes") -> dict[str, Any]:
    config = get_disease_config(disease_id)
    artifact_dir = config.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.dataset_path)
    names = feature_names(config)
    X = data[names]
    y = data[config.target_column].isin(config.positive_labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_results: list[dict[str, Any]] = []
    best_search = None
    best_name = ""
    best_score = -np.inf

    for model_name, (model, grid) in _model_grids().items():
        search = GridSearchCV(
            estimator=_make_pipeline(model),
            param_grid=grid,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        cv_frame = pd.DataFrame(search.cv_results_)
        cv_frame.insert(0, "model_family", model_name)
        model_results.append(cv_frame)

        if float(search.best_score_) > best_score:
            best_score = float(search.best_score_)
            best_search = search
            best_name = model_name

    if best_search is None:
        raise RuntimeError("No model was trained.")

    best_pipeline = best_search.best_estimator_
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=best_pipeline,
        method="sigmoid",
        cv=3,
    )
    calibrated_pipeline.fit(X_train, y_train)

    train_probabilities = calibrated_pipeline.predict_proba(X_train)[:, 1]
    threshold = choose_threshold(y_train, train_probabilities)
    test_probabilities = calibrated_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probabilities, threshold)

    cv_results = pd.concat(model_results, ignore_index=True)
    cv_results.to_csv(artifact_dir / "cv_results.csv", index=False)
    joblib.dump(calibrated_pipeline, artifact_dir / "best_model.joblib")
    X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE).to_csv(
        artifact_dir / "shap_background.csv",
        index=False,
    )

    feature_importance = model_feature_importance(best_pipeline, names)
    metrics = {
        "selected_model": best_name,
        "selection_metric": "roc_auc",
        "best_cv_roc_auc": round(best_score, 4),
        "test_metrics": test_metrics,
        "threshold": threshold,
        "best_params": best_search.best_params_,
        "feature_importance": feature_importance,
        "probability_calibration": {
            "method": "CalibratedClassifierCV",
            "calibration": "sigmoid",
            "cv": 3,
        },
    }
    metadata = {
        "disease_id": config.disease_id,
        "display_name": config.display_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "pipeline": ["features", "imputer", "scaler", "smote", "model", "CalibratedClassifierCV(sigmoid)"],
        "features": names,
        "target_column": config.target_column,
        "positive_labels": list(config.positive_labels),
        "provenance": config.provenance,
    }

    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifact_dir / "dataset_profile.json").write_text(
        json.dumps(_dataset_profile(data, config), indent=2),
        encoding="utf-8",
    )
    _save_figures(artifact_dir, data, X_test, y_test, test_probabilities, threshold, best_pipeline, names)

    return {"metadata": metadata, "metrics": metrics, "artifact_dir": str(artifact_dir)}

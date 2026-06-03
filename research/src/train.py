from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from research.src.evaluate import (
    compute_metrics,
    feature_importance_from_pipeline,
    save_confusion_matrix,
    save_feature_importance,
    save_heatmap,
    save_roc_curve,
)
from research.src.explainability import save_xgboost_shap
from research.src.preprocessing import FeatureEngineer, make_preprocessor, split_feature_types
from research.src.utils import (
    DISEASES,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    dataset_path,
    ensure_research_dirs,
    set_seed,
    write_json,
)


MODEL_SPECS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__min_samples_leaf": [1, 3],
        },
    ),
    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    ),
}


def build_pipeline(disease_id: str, frame: pd.DataFrame, target: str, model) -> ImbPipeline:
    engineered = FeatureEngineer(disease_id).fit_transform(frame.drop(columns=[target]))
    engineered[target] = frame[target].values
    numeric_columns, categorical_columns = split_feature_types(engineered, target)
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    return ImbPipeline(
        steps=[
            ("features", FeatureEngineer(disease_id)),
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def binary_target(frame: pd.DataFrame, disease_id: str) -> pd.Series:
    spec = DISEASES[disease_id]
    return frame[spec.target].astype(str).str.strip().isin([str(value) for value in spec.positive_values]).astype(int)


def train_disease(disease_id: str) -> tuple[list[dict], dict, pd.DataFrame]:
    spec = DISEASES[disease_id]
    frame = pd.read_csv(dataset_path(disease_id))
    frame = frame.replace(["?", "\t?", " ?"], np.nan)
    y = binary_target(frame, disease_id)
    X = frame.drop(columns=[spec.target])
    save_heatmap(pd.concat([X, y.rename(spec.target)], axis=1), spec.target, disease_id)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    result_rows: list[dict] = []
    best = {"score": -1.0, "name": "", "pipeline": None, "params": {}, "metrics": {}}
    xgboost_pipeline = None
    cv_frames = []

    for model_name, (model, grid) in MODEL_SPECS.items():
        pipeline = build_pipeline(disease_id, frame, spec.target, model)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        cv_accuracy = cross_val_score(search.best_estimator_, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1).mean()
        metrics = compute_metrics(y_test, y_pred, y_prob, cv_accuracy)
        result_rows.append({"Disease": spec.display_name, "Model": model_name, **metrics})
        save_confusion_matrix(y_test, y_pred, disease_id, model_name.replace(" ", "_").lower())
        save_roc_curve(y_test, y_prob, disease_id, model_name.replace(" ", "_").lower())

        feature_names = list(search.best_estimator_.named_steps["preprocessor"].get_feature_names_out())
        importances = feature_importance_from_pipeline(search.best_estimator_, feature_names)
        save_feature_importance(importances, disease_id, model_name.replace(" ", "_").lower())
        importance_table = pd.DataFrame(
            [{"Disease": spec.display_name, "Model": model_name, "Feature": key, "Importance Score": value} for key, value in importances.items()]
        )
        if not importance_table.empty:
            importance_table.to_csv(REPORTS_DIR / f"{disease_id}_{model_name.replace(' ', '_').lower()}_feature_importance.csv", index=False)

        cv_frame = pd.DataFrame(search.cv_results_)
        cv_frame.insert(0, "Disease", spec.display_name)
        cv_frame.insert(1, "Model", model_name)
        cv_frames.append(cv_frame)

        if metrics["ROC-AUC"] > best["score"]:
            best = {
                "score": metrics["ROC-AUC"],
                "name": model_name,
                "pipeline": search.best_estimator_,
                "params": search.best_params_,
                "metrics": metrics,
            }
        if model_name == "XGBoost":
            xgboost_pipeline = search.best_estimator_

    assert best["pipeline"] is not None
    artifact = {
        "disease_id": disease_id,
        "model_name": best["name"],
        "pipeline": best["pipeline"],
        "metrics": best["metrics"],
        "best_params": best["params"],
        "threshold": 0.5,
        "features": X.columns.tolist(),
    }
    joblib.dump(artifact, MODELS_DIR / f"{disease_id}_model.pkl")
    write_json(REPORTS_DIR / f"{disease_id}_best_hyperparameters.json", best["params"])

    if xgboost_pipeline is not None:
        save_xgboost_shap(xgboost_pipeline, X_test.head(100), disease_id)

    dataset_summary = {
        "Dataset": spec.display_name,
        "Samples": int(frame.shape[0]),
        "Features": int(X.shape[1]),
        "Target": spec.target,
    }
    return result_rows, dataset_summary, pd.concat(cv_frames, ignore_index=True)


def train_all(selected: list[str] | None = None) -> None:
    ensure_research_dirs()
    set_seed()
    selected = selected or list(DISEASES)
    all_results = []
    summaries = []
    cv_results = []
    best_rows = []

    for disease_id in selected:
        result_rows, dataset_summary, cv_frame = train_disease(disease_id)
        all_results.extend(result_rows)
        summaries.append(dataset_summary)
        cv_results.append(cv_frame)
        best_row = max(result_rows, key=lambda row: row["ROC-AUC"])
        best_rows.append(best_row)

    results = pd.DataFrame(all_results)
    results.to_csv(REPORTS_DIR / "results_table.csv", index=False)
    pd.DataFrame(best_rows).to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS_DIR / "dataset_summary.csv", index=False)
    pd.concat(cv_results, ignore_index=True).to_csv(REPORTS_DIR / "gridsearch_cv_results.csv", index=False)


def main() -> None:
    selected = sys.argv[1:] or None
    train_all(selected)


if __name__ == "__main__":
    main()

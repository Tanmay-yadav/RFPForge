from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def transformed_feature_names(pipeline, original_features: list[str]) -> list[str]:
    engineer = pipeline.named_steps.get("features")
    if engineer and hasattr(engineer, "get_feature_names_out"):
        names = engineer.get_feature_names_out(original_features)
        if len(names):
            return [str(name) for name in names]
    return original_features


def transform_before_model(pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    values = frame
    for name, step in pipeline.steps:
        if name in {"smote", "model"}:
            continue
        values = step.transform(values)
    if isinstance(values, pd.DataFrame):
        return values
    names = transformed_feature_names(pipeline, frame.columns.tolist())
    return pd.DataFrame(values, columns=names)


def model_feature_importance(pipeline, original_features: list[str]) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline, original_features)
    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])

    if values is None:
        return {}

    return {
        name: round(float(value), 6)
        for name, value in sorted(zip(names, values), key=lambda item: item[1], reverse=True)
    }


def shap_contributions(
    pipeline,
    frame: pd.DataFrame,
    top_n: int = 5,
    background: pd.DataFrame | None = None,
) -> list[dict[str, float | str]]:
    try:
        import shap
    except ImportError:
        return []

    try:
        transformed = transform_before_model(pipeline, frame)
        model = pipeline.named_steps["model"]
        explainer_background = transform_before_model(pipeline, background) if background is not None else transformed
        explainer = shap.Explainer(model, explainer_background)
        values = explainer(transformed)
        shap_values = values.values[0]
        if np.ndim(shap_values) > 1:
            shap_values = shap_values[:, -1]
        if not np.any(np.abs(shap_values) > 1e-12):
            return []
        ranked = sorted(
            zip(transformed.columns, shap_values, transformed.iloc[0].values),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        return [
            {
                "feature": str(feature),
                "value": round(float(value), 4) if np.isfinite(value) else None,
                "contribution": round(float(contribution), 6),
            }
            for feature, contribution, value in ranked[:top_n]
        ]
    except Exception:
        return []


def save_shap_summary(pipeline, frame: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return False

    try:
        transformed = transform_before_model(pipeline, frame)
        model = pipeline.named_steps["model"]
        explainer = shap.Explainer(model, transformed)
        values = explainer(transformed)
        plt.figure()
        shap.summary_plot(values, transformed, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False

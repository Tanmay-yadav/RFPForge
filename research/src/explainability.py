from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.src.utils import VISUALS_DIR


def transformed_data(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    Xt = X
    for name, step in pipeline.steps:
        if name in {"smote", "model"}:
            continue
        Xt = step.transform(Xt)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame(Xt, columns=names), list(names)


def save_xgboost_shap(pipeline, X_sample: pd.DataFrame, disease_id: str) -> dict[str, str]:
    try:
        import shap
    except ImportError:
        return {}

    model = pipeline.named_steps["model"]
    if model.__class__.__name__ != "XGBClassifier":
        return {}

    Xt, names = transformed_data(pipeline, X_sample)
    explainer = shap.Explainer(model, Xt)
    values = explainer(Xt)
    output: dict[str, str] = {}

    summary_path = VISUALS_DIR / "shap" / f"{disease_id}_xgboost_summary.png"
    shap.summary_plot(values, Xt, show=False)
    plt.title(f"{disease_id.title()} XGBoost SHAP Summary")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    output["summary"] = str(summary_path)

    bar_path = VISUALS_DIR / "shap" / f"{disease_id}_xgboost_bar.png"
    shap.plots.bar(values, show=False)
    plt.title(f"{disease_id.title()} XGBoost SHAP Bar Plot")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    output["bar"] = str(bar_path)

    dependence_feature = names[int(np.argmax(np.abs(values.values).mean(axis=0)))]
    dep_path = VISUALS_DIR / "shap" / f"{disease_id}_xgboost_dependence.png"
    shap.dependence_plot(dependence_feature, values.values, Xt, show=False)
    plt.title(f"{disease_id.title()} XGBoost SHAP Dependence: {dependence_feature}")
    plt.tight_layout()
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    output["dependence"] = str(dep_path)
    return output


def top_shap_contributors(pipeline, row: pd.DataFrame, background: pd.DataFrame, top_n: int = 5) -> list[dict[str, float | str]]:
    try:
        import shap
    except ImportError:
        return []

    Xt_background, names = transformed_data(pipeline, background)
    Xt_row, _ = transformed_data(pipeline, row)
    model = pipeline.named_steps["model"]
    explainer = shap.Explainer(model, Xt_background)
    values = explainer(Xt_row)
    shap_values = values.values[0]
    if np.ndim(shap_values) > 1:
        shap_values = shap_values[:, -1]
    ranked = sorted(zip(names, Xt_row.iloc[0].values, shap_values), key=lambda item: abs(float(item[2])), reverse=True)
    return [
        {"feature": str(name), "value": round(float(value), 4), "contribution": round(float(score), 6)}
        for name, value, score in ranked[:top_n]
    ]


from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from research.src.utils import VISUALS_DIR


def compute_metrics(y_true, y_pred, y_prob, cv_accuracy: float) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "CV Mean Accuracy": round(float(cv_accuracy), 4),
    }


def save_heatmap(frame: pd.DataFrame, target: str, disease_id: str) -> Path:
    path = VISUALS_DIR / "heatmaps" / f"{disease_id}_heatmap.png"
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="coolwarm", annot=False, square=False)
    plt.title(f"{disease_id.title()} Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def save_confusion_matrix(y_true, y_pred, disease_id: str, model_name: str) -> Path:
    path = VISUALS_DIR / "confusion_matrices" / f"{disease_id}_{model_name}.png"
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues")
    plt.title(f"{disease_id.title()} {model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def save_roc_curve(y_true, y_prob, disease_id: str, model_name: str) -> Path:
    path = VISUALS_DIR / "roc_curves" / f"{disease_id}_{model_name}.png"
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{disease_id.title()} {model_name} ROC Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def save_feature_importance(importances: dict[str, float], disease_id: str, model_name: str) -> Path | None:
    if not importances:
        return None
    path = VISUALS_DIR / "feature_importance" / f"{disease_id}_{model_name}.png"
    items = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)[:20]
    names = [item[0] for item in items]
    values = [item[1] for item in items]
    plt.figure(figsize=(9, 7))
    sns.barplot(x=values, y=names, color="#2563eb")
    plt.title(f"{disease_id.title()} {model_name} Feature Importance")
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def feature_importance_from_pipeline(pipeline, feature_names: list[str]) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return {}
    return {name: round(float(value), 6) for name, value in zip(feature_names, values)}


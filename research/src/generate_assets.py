from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research.src.utils import DISEASES, REPORTS_DIR, VISUALS_DIR, ensure_research_dirs


def markdown_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return "_Not generated yet._"
    frame = pd.read_csv(csv_path)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def generate_architecture_diagram() -> Path:
    ensure_research_dirs()
    path = VISUALS_DIR / "architecture" / "system_architecture.png"
    labels = [
        "User Input",
        "Disease Selection",
        "Preprocessing",
        "Feature Engineering",
        "Model Selection",
        "Prediction",
        "SHAP Explanation",
        "Output",
    ]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")
    y_positions = list(reversed(range(len(labels))))
    for label, y in zip(labels, y_positions):
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#2563eb", linewidth=1.8),
        )
        if y > 0:
            ax.annotate(
                "",
                xy=(0.5, y - 0.38),
                xytext=(0.5, y - 0.72),
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.8),
            )
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xlim(0, 1)
    plt.title("Multi-Disease Prediction System Architecture", fontsize=15, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_ieee_notes() -> Path:
    ensure_research_dirs()
    generate_architecture_diagram()
    dataset_table = markdown_table(REPORTS_DIR / "dataset_summary.csv")
    results_table = markdown_table(REPORTS_DIR / "results_table.csv")
    metrics_table = markdown_table(REPORTS_DIR / "metrics_summary.csv")
    references = "\n".join(f"- {spec.citation}" for spec in DISEASES.values())

    content = f"""# IEEE Notes: Research-Grade Multi-Disease Prediction System

## Abstract
This work presents a reproducible multi-disease prediction framework for diabetes, heart disease, liver disease, and chronic kidney disease. The framework compares Logistic Regression, Random Forest, and XGBoost using leakage-safe preprocessing pipelines, SMOTE class balancing, 5-fold cross-validation, GridSearchCV tuning, and SHAP explainability.

## Introduction
Machine learning methods can support early screening by learning patterns from tabular clinical datasets. The system is designed as an educational research implementation and does not provide clinical diagnosis.

## Literature Review
Prior medical prediction studies commonly evaluate linear models, tree ensembles, and gradient boosting on structured patient features. Logistic Regression provides interpretable probabilistic baselines, Random Forest improves non-linear robustness through bagging, and XGBoost often performs strongly on tabular medical data due to regularized boosting.

## Methodology
The workflow includes dataset preparation, missing value handling, median imputation, categorical encoding, standardization, feature engineering, SMOTE imbalance handling, model training, 5-fold cross-validation, GridSearchCV tuning, evaluation, visualization, and SHAP-based explainability.

## Dataset Summary
{dataset_table}

## Mathematical Foundations

### Logistic Regression
Logistic Regression estimates the probability of the positive class as:

```text
P(y=1|x) = 1 / (1 + e^(-z))
z = w^T x + b
```

The sigmoid maps a linear score to a probability between 0 and 1. The decision boundary is the hyperplane where the predicted probability crosses the selected classification threshold.

### Random Forest
Random Forest is an ensemble of decision trees trained on bootstrap samples. Classification is produced by averaging tree votes. Node splitting can use Gini impurity:

```text
Gini = 1 - sum(p_i^2)
```

where `p_i` is the proportion of class `i` in a node.

### XGBoost
XGBoost performs gradient boosting by adding trees sequentially to correct previous errors. Its regularized objective is:

```text
Obj = sum l(y_i, y_hat_i) + sum Omega(f_k)
Omega(f) = gamma T + 1/2 lambda ||w||^2
```

The regularization term penalizes overly complex trees, improving generalization on tabular medical datasets.

### StandardScaler
Standardization uses:

```text
z = (x - mu) / sigma
```

where `mu` is the training mean and `sigma` is the training standard deviation.

### Evaluation Metrics

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

ROC-AUC summarizes discrimination across thresholds using true positive rate and false positive rate.

## Experimental Setup
All experiments use `random_state=42`, `np.random.seed(42)`, 5-fold stratified cross-validation, SMOTE inside the training pipeline, and GridSearchCV for hyperparameter tuning.

## Results and Discussion

### Model Comparison Table
{results_table}

### Best Model Metrics
{metrics_table}

## Explainability Analysis
SHAP summary, bar, and dependence plots are generated for XGBoost models. Feature importance plots are also saved for Logistic Regression, Random Forest, and XGBoost where supported.

## Limitations
The datasets are public benchmark datasets and may not represent local clinical populations. The implementation is not clinically validated and should not be used as a diagnostic device.

## Future Work
Future work should include external validation, calibration analysis, confidence intervals, fairness analysis, prospective data collection, and clinician review of feature interpretations.

## Conclusion
The system provides a reproducible IEEE-style experimental framework for multi-disease prediction with full metrics, visualizations, explainability assets, and saved models.

## References Placeholder
{references}
"""
    path = REPORTS_DIR / "ieee_notes.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    print(generate_ieee_notes())


if __name__ == "__main__":
    main()

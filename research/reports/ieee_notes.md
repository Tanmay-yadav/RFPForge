# IEEE Notes: Research-Grade Multi-Disease Prediction System

## Abstract
This work presents a reproducible multi-disease prediction framework for diabetes, heart disease, liver disease, and chronic kidney disease. The framework compares Logistic Regression, Random Forest, and XGBoost using leakage-safe preprocessing pipelines, SMOTE class balancing, 5-fold cross-validation, GridSearchCV tuning, and SHAP explainability.

## Introduction
Machine learning methods can support early screening by learning patterns from tabular clinical datasets. The system is designed as an educational research implementation and does not provide clinical diagnosis.

## Literature Review
Prior medical prediction studies commonly evaluate linear models, tree ensembles, and gradient boosting on structured patient features. Logistic Regression provides interpretable probabilistic baselines, Random Forest improves non-linear robustness through bagging, and XGBoost often performs strongly on tabular medical data due to regularized boosting.

## Methodology
The workflow includes dataset preparation, missing value handling, median imputation, categorical encoding, standardization, feature engineering, SMOTE imbalance handling, model training, 5-fold cross-validation, GridSearchCV tuning, evaluation, visualization, and SHAP-based explainability.

## Dataset Summary
| Dataset | Samples | Features | Target |
| --- | --- | --- | --- |
| Diabetes | 768 | 8 | Outcome |
| Heart Disease | 297 | 13 | target |
| Liver Disease | 583 | 10 | Selector |
| Chronic Kidney Disease | 400 | 24 | class |

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
| Disease | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | Logistic Regression | 0.7078 | 0.5714 | 0.6667 | 0.6154 | 0.8085 | 0.7589 |
| Diabetes | Random Forest | 0.7208 | 0.5733 | 0.7963 | 0.6667 | 0.8248 | 0.7638 |
| Diabetes | XGBoost | 0.7403 | 0.5921 | 0.8333 | 0.6923 | 0.8167 | 0.7589 |
| Heart Disease | Logistic Regression | 0.85 | 0.8519 | 0.8214 | 0.8364 | 0.9498 | 0.8143 |
| Heart Disease | Random Forest | 0.8667 | 0.8846 | 0.8214 | 0.8519 | 0.9408 | 0.8099 |
| Heart Disease | XGBoost | 0.8333 | 0.875 | 0.75 | 0.8077 | 0.933 | 0.8099 |
| Liver Disease | Logistic Regression | 0.7094 | 0.9153 | 0.6506 | 0.7606 | 0.8398 | 0.6288 |
| Liver Disease | Random Forest | 0.7265 | 0.8493 | 0.747 | 0.7949 | 0.8047 | 0.6523 |
| Liver Disease | XGBoost | 0.735 | 0.7766 | 0.8795 | 0.8249 | 0.7562 | 0.6931 |
| Chronic Kidney Disease | Logistic Regression | 0.975 | 1.0 | 0.96 | 0.9796 | 0.9993 | 0.9938 |
| Chronic Kidney Disease | Random Forest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9906 |
| Chronic Kidney Disease | XGBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9812 |

### Best Model Metrics
| Disease | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | Random Forest | 0.7208 | 0.5733 | 0.7963 | 0.6667 | 0.8248 | 0.7638 |
| Heart Disease | Logistic Regression | 0.85 | 0.8519 | 0.8214 | 0.8364 | 0.9498 | 0.8143 |
| Liver Disease | Logistic Regression | 0.7094 | 0.9153 | 0.6506 | 0.7606 | 0.8398 | 0.6288 |
| Chronic Kidney Disease | Random Forest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9906 |

## Explainability Analysis
SHAP summary, bar, and dependence plots are generated for XGBoost models. Feature importance plots are also saved for Logistic Regression, Random Forest, and XGBoost where supported.

## Limitations
The datasets are public benchmark datasets and may not represent local clinical populations. The implementation is not clinically validated and should not be used as a diagnostic device.

## Future Work
Future work should include external validation, calibration analysis, confidence intervals, fairness analysis, prospective data collection, and clinician review of feature interpretations.

## Conclusion
The system provides a reproducible IEEE-style experimental framework for multi-disease prediction with full metrics, visualizations, explainability assets, and saved models.

## References Placeholder
- Confirm final Pima diabetes citation/license before IEEE submission.
- UCI Machine Learning Repository, Heart Disease dataset.
- Ramana, B. & Venkateswarlu, N. (2022). ILPD [Dataset]. UCI ML Repository. DOI: 10.24432/C5D02C.
- Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI ML Repository. DOI: 10.24432/C5G020.

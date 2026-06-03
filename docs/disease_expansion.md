# Disease Model Expansion Guide

This project supports multiple disease prediction models through `app/ml/registry.py`.

## Current Disease IDs
- `diabetes`: local diabetes CSV already included.
- `heart_disease`: UCI processed Cleveland Heart Disease dataset.
- `breast_cancer`: Wisconsin Diagnostic Breast Cancer dataset exported from scikit-learn.
- `parkinsons`: UCI Parkinsons voice dataset.

## Prepare Built-In Approved Datasets
Run this once to create the extra CSV files under `ML-MODEL/Data/`:

```powershell
venv\Scripts\python.exe scripts\prepare_disease_datasets.py --dataset all
```

This downloads Heart Disease and Parkinsons from UCI source URLs and exports Breast Cancer from scikit-learn.

## Train Models
Train one disease:

```powershell
venv\Scripts\python.exe scripts\train_disease_models.py --disease heart_disease
```

Train every registered disease:

```powershell
venv\Scripts\python.exe scripts\train_disease_models.py --disease all
```

Artifacts are saved to:

```text
ML-MODEL/artifacts/{disease_id}/
```

Each trained disease appears automatically in the Streamlit disease selector after the backend restarts.

## Add a New Disease Manually
1. Put your approved CSV in `ML-MODEL/Data/`.
2. Confirm the target column and which target value means “risk/disease present”.
3. Add a `FeatureSpec` tuple in `app/ml/registry.py`.
4. Add a `DiseaseConfig` entry in `DISEASE_REGISTRY` with:
   - `disease_id`
   - `display_name`
   - `dataset_path`
   - `target_column`
   - `positive_labels`
   - `feature_specs`
   - `artifact_dir`
   - `provenance`
5. If the dataset uses unusual raw column names, add cleaning/export logic to `scripts/prepare_disease_datasets.py`.
6. Train it:

```powershell
venv\Scripts\python.exe scripts\train_disease_models.py --disease your_disease_id
```

7. Restart the backend and frontend.

## Notes for IEEE-Style Reporting
- Record dataset source URL, citation, license, target mapping, row count, and feature list.
- Do not mix unapproved internet datasets into the paper without documenting provenance.
- Keep `metrics.json`, `cv_results.csv`, `dataset_profile.json`, and `figures/` for tables and figures.
- “Disease-specific retrieval validation was implemented to prevent unrelated clinical context generation in the RAG explanation module.”
- Treat this as educational screening software unless you perform clinical validation.

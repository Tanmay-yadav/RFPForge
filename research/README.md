# Research-Grade Multi-Disease Prediction

This workspace contains an IEEE-style research implementation for:

- Diabetes
- Heart Disease
- Liver Disease
- Chronic Kidney Disease

It is separate from the main FastAPI app so experiments do not break production routes.

## Prepare Datasets

```powershell
venv\Scripts\python.exe -m research.src.prepare_datasets
```

The liver and kidney datasets require `ucimlrepo`.

## Train Models

Train all diseases:

```powershell
venv\Scripts\python.exe -m research.src.train
```

Train one disease:

```powershell
venv\Scripts\python.exe -m research.src.train diabetes
```

## Generate IEEE Notes And Architecture Diagram

```powershell
venv\Scripts\python.exe -m research.src.generate_assets
```

## Run Research UI

```powershell
venv\Scripts\python.exe -m streamlit run research/app.py
```

## Outputs

- Models: `research/models/`
- Metrics: `research/reports/`
- Figures: `research/visuals/`
- IEEE notes: `research/reports/ieee_notes.md`


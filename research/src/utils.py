from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class DiseaseSpec:
    disease_id: str
    display_name: str
    dataset_file: str
    target: str
    positive_values: tuple[Any, ...]
    source: str
    citation: str
    feature_help: dict[str, str]


DISEASES: dict[str, DiseaseSpec] = {
    "diabetes": DiseaseSpec(
        disease_id="diabetes",
        display_name="Diabetes",
        dataset_file="diabetes.csv",
        target="Outcome",
        positive_values=(1,),
        source="Local Pima-style diabetes CSV approved in the project",
        citation="Confirm final Pima diabetes citation/license before IEEE submission.",
        feature_help={
            "Pregnancies": "Number of times pregnant.",
            "Glucose": "Plasma glucose concentration.",
            "BloodPressure": "Diastolic blood pressure.",
            "SkinThickness": "Triceps skinfold thickness.",
            "Insulin": "2-hour serum insulin.",
            "BMI": "Body mass index.",
            "DiabetesPedigreeFunction": "Family-history diabetes pedigree score.",
            "Age": "Age in years.",
        },
    ),
    "heart": DiseaseSpec(
        disease_id="heart",
        display_name="Heart Disease",
        dataset_file="heart.csv",
        target="target",
        positive_values=(1, 2, 3, 4),
        source="UCI Heart Disease processed Cleveland dataset",
        citation="UCI Machine Learning Repository, Heart Disease dataset.",
        feature_help={
            "age": "Age in years.",
            "sex": "Sex encoded by the source dataset.",
            "cp": "Chest pain type.",
            "trestbps": "Resting blood pressure.",
            "chol": "Serum cholesterol.",
            "fbs": "Fasting blood sugar indicator.",
            "restecg": "Resting electrocardiographic result.",
            "thalach": "Maximum heart rate achieved.",
            "exang": "Exercise-induced angina indicator.",
            "oldpeak": "ST depression induced by exercise.",
            "slope": "Slope of peak exercise ST segment.",
            "ca": "Number of major vessels.",
            "thal": "Thalassemia category.",
        },
    ),
    "liver": DiseaseSpec(
        disease_id="liver",
        display_name="Liver Disease",
        dataset_file="liver.csv",
        target="Selector",
        positive_values=(1,),
        source="UCI ILPD Indian Liver Patient Dataset",
        citation="Ramana, B. & Venkateswarlu, N. (2022). ILPD [Dataset]. UCI ML Repository. DOI: 10.24432/C5D02C.",
        feature_help={
            "Age": "Age of the patient.",
            "Gender": "Patient gender from the source dataset.",
            "TB": "Total bilirubin.",
            "DB": "Direct bilirubin.",
            "Alkphos": "Alkaline phosphotase.",
            "Sgpt": "Alamine aminotransferase.",
            "Sgot": "Aspartate aminotransferase.",
            "TP": "Total proteins.",
            "ALB": "Albumin.",
            "AG_Ratio": "Albumin and globulin ratio.",
        },
    ),
    "kidney": DiseaseSpec(
        disease_id="kidney",
        display_name="Chronic Kidney Disease",
        dataset_file="kidney.csv",
        target="class",
        positive_values=("ckd",),
        source="UCI Chronic Kidney Disease dataset",
        citation="Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI ML Repository. DOI: 10.24432/C5G020.",
        feature_help={
            "age": "Age in years.",
            "bp": "Blood pressure.",
            "sg": "Urine specific gravity.",
            "al": "Albumin level category.",
            "su": "Sugar level category.",
            "rbc": "Red blood cell status.",
            "pc": "Pus cell status.",
            "pcc": "Pus cell clumps.",
            "ba": "Bacteria presence.",
            "bgr": "Blood glucose random.",
            "bu": "Blood urea.",
            "sc": "Serum creatinine.",
            "sod": "Sodium.",
            "pot": "Potassium.",
            "hemo": "Hemoglobin.",
            "pcv": "Packed cell volume.",
            "wc": "White blood cell count.",
            "rc": "Red blood cell count.",
            "htn": "Hypertension.",
            "dm": "Diabetes mellitus.",
            "cad": "Coronary artery disease.",
            "appet": "Appetite.",
            "pe": "Pedal edema.",
            "ane": "Anemia.",
        },
    ),
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_research_dirs() -> None:
    for directory in [
        DATASETS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        VISUALS_DIR / "heatmaps",
        VISUALS_DIR / "confusion_matrices",
        VISUALS_DIR / "roc_curves",
        VISUALS_DIR / "feature_importance",
        VISUALS_DIR / "shap",
        VISUALS_DIR / "architecture",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def dataset_path(disease_id: str) -> Path:
    return DATASETS_DIR / DISEASES[disease_id].dataset_file


def model_path(disease_id: str) -> Path:
    return MODELS_DIR / f"{disease_id}_model.pkl"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


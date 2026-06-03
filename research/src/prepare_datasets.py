from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from research.src.utils import DATASETS_DIR, ensure_research_dirs


ROOT = Path(__file__).resolve().parents[2]
HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
KIDNEY_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "class",
]


def prepare_diabetes() -> None:
    source = ROOT / "ML-MODEL" / "Data" / "diabetes.csv"
    shutil.copyfile(source, DATASETS_DIR / "diabetes.csv")


def prepare_heart() -> None:
    prepared = ROOT / "ML-MODEL" / "Data" / "heart_disease_uci.csv"
    if prepared.exists():
        shutil.copyfile(prepared, DATASETS_DIR / "heart.csv")
        return
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    frame = pd.read_csv(url, names=HEART_COLUMNS, na_values="?").dropna()
    frame.to_csv(DATASETS_DIR / "heart.csv", index=False)


def prepare_liver() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install ucimlrepo to prepare the UCI liver dataset.") from exc

    dataset = fetch_ucirepo(id=225)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    frame.columns = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "AG_Ratio", "Selector"]
    frame.to_csv(DATASETS_DIR / "liver.csv", index=False)


def prepare_kidney() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install ucimlrepo to prepare the UCI chronic kidney disease dataset.") from exc

    dataset = fetch_ucirepo(id=336)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    frame.columns = KIDNEY_COLUMNS
    frame = frame.replace({"\t?": None, "?": None, "ckd\t": "ckd", " yes": "yes", "\tyes": "yes", "\tno": "no"})
    frame.to_csv(DATASETS_DIR / "kidney.csv", index=False)


def main() -> None:
    ensure_research_dirs()
    selected = sys.argv[1:] or ["diabetes", "heart", "liver", "kidney"]
    tasks = {
        "diabetes": prepare_diabetes,
        "heart": prepare_heart,
        "liver": prepare_liver,
        "kidney": prepare_kidney,
    }
    for disease_id in selected:
        tasks[disease_id]()
        print(f"prepared {disease_id}")


if __name__ == "__main__":
    main()


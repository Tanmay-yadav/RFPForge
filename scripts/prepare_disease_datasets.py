from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ML-MODEL" / "Data"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
PARKINSONS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


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


def sanitize_column(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def prepare_heart_disease() -> Path:
    output = DATA_DIR / "heart_disease_uci.csv"
    raw = pd.read_csv(HEART_URL, names=HEART_COLUMNS, na_values="?")
    raw = raw.dropna().reset_index(drop=True)
    for column in HEART_COLUMNS:
        raw[column] = pd.to_numeric(raw[column])
    raw.to_csv(output, index=False)
    return output


def prepare_breast_cancer() -> Path:
    output = DATA_DIR / "breast_cancer_wdbc.csv"
    dataset = load_breast_cancer()
    frame = pd.DataFrame(dataset.data, columns=[sanitize_column(name) for name in dataset.feature_names])
    frame["target"] = dataset.target
    frame.to_csv(output, index=False)
    return output


def prepare_parkinsons() -> Path:
    output = DATA_DIR / "parkinsons_uci.csv"
    raw = pd.read_csv(PARKINSONS_URL)
    raw = raw.drop(columns=["name"])
    raw = raw.rename(
        columns={
            "MDVP:Fo(Hz)": "MDVP_Fo_Hz",
            "MDVP:Fhi(Hz)": "MDVP_Fhi_Hz",
            "MDVP:Flo(Hz)": "MDVP_Flo_Hz",
            "MDVP:Jitter(%)": "MDVP_Jitter_percent",
            "MDVP:Jitter(Abs)": "MDVP_Jitter_Abs",
            "MDVP:RAP": "MDVP_RAP",
            "MDVP:PPQ": "MDVP_PPQ",
            "Jitter:DDP": "Jitter_DDP",
            "MDVP:Shimmer": "MDVP_Shimmer",
            "MDVP:Shimmer(dB)": "MDVP_Shimmer_dB",
            "Shimmer:APQ3": "Shimmer_APQ3",
            "Shimmer:APQ5": "Shimmer_APQ5",
            "MDVP:APQ": "MDVP_APQ",
            "Shimmer:DDA": "Shimmer_DDA",
        }
    )
    raw.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare approved disease datasets for training.")
    parser.add_argument(
        "--dataset",
        choices=["all", "heart_disease", "breast_cancer", "parkinsons"],
        default="all",
        help="Dataset to prepare. Defaults to all non-diabetes datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tasks = {
        "heart_disease": prepare_heart_disease,
        "breast_cancer": prepare_breast_cancer,
        "parkinsons": prepare_parkinsons,
    }
    selected = tasks if args.dataset == "all" else {args.dataset: tasks[args.dataset]}

    for dataset_id, task in selected.items():
        path = task()
        print(f"{dataset_id}: wrote {path}")


if __name__ == "__main__":
    main()

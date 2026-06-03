from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, disease_id: str):
        self.disease_id = disease_id
        self.columns_: list[str] | None = None

    def fit(self, X, y=None):
        transformed = self._transform_frame(X)
        self.columns_ = transformed.columns.tolist()
        return self

    def transform(self, X):
        return self._transform_frame(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_ or list(input_features or []))

    def _transform_frame(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame = frame.replace(["?", "\t?", " ?"], np.nan)

        if self.disease_id == "diabetes":
            for column in DIABETES_ZERO_AS_MISSING:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(0, np.nan)

        if {"BMI", "Age"}.issubset(frame.columns):
            frame["BMI_Age"] = pd.to_numeric(frame["BMI"], errors="coerce") * pd.to_numeric(frame["Age"], errors="coerce")

        if {"Glucose", "BMI"}.issubset(frame.columns):
            glucose = pd.to_numeric(frame["Glucose"], errors="coerce")
            bmi = pd.to_numeric(frame["BMI"], errors="coerce").replace(0, np.nan)
            frame["Glucose_BMI"] = glucose / bmi

        if {"chol", "age"}.issubset(frame.columns):
            frame["Chol_Age"] = pd.to_numeric(frame["chol"], errors="coerce") * pd.to_numeric(frame["age"], errors="coerce")

        if {"trestbps", "thalach"}.issubset(frame.columns):
            bp = pd.to_numeric(frame["trestbps"], errors="coerce")
            heart_rate = pd.to_numeric(frame["thalach"], errors="coerce").replace(0, np.nan)
            frame["BP_HeartRate"] = bp / heart_rate

        return frame


def split_feature_types(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    features = frame.drop(columns=[target])
    numeric_columns = []
    categorical_columns = []

    for column in features.columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        non_missing = features[column].notna().sum()
        numeric_ratio = converted.notna().sum() / max(non_missing, 1)
        if numeric_ratio > 0.8:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


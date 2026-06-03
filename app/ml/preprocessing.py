from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe diabetes feature engineering for sklearn/imblearn pipelines."""

    def __init__(self, invalid_zero_columns: tuple[str, ...] | None = None):
        self.invalid_zero_columns = invalid_zero_columns or (
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        )
        self.feature_names_out_: list[str] | None = None

    def fit(self, X, y=None):
        transformed = self._transform_frame(X)
        self.feature_names_out_ = transformed.columns.tolist()
        return self

    def transform(self, X):
        return self._transform_frame(X)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            if input_features is None:
                return np.array([])
            return np.array(input_features)
        return np.array(self.feature_names_out_)

    def _transform_frame(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()

        for column in self.invalid_zero_columns:
            if column in frame.columns:
                frame[column] = frame[column].replace(0, np.nan)

        if "BMI" in frame.columns:
            frame["BMI_Category"] = pd.cut(
                frame["BMI"],
                bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
                labels=[0, 1, 2, 3],
            ).astype(float)

        if {"Glucose", "Age"}.issubset(frame.columns):
            frame["Glucose_Age_Ratio"] = frame["Glucose"] / frame["Age"].replace(0, np.nan)

        if {"Insulin", "Glucose"}.issubset(frame.columns):
            frame["Insulin_Glucose_Ratio"] = frame["Insulin"] / frame["Glucose"].replace(0, np.nan)

        if {"BMI", "Age"}.issubset(frame.columns):
            frame["BMI_Age_Product"] = frame["BMI"] * frame["Age"]

        return frame


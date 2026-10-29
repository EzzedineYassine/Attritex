from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnsense.data.schema import categorical_feature_names, numeric_feature_names


class TotalChargesImputer(BaseEstimator, TransformerMixin):
    def fit(self, features: pd.DataFrame, target=None):
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(frame.columns):
            estimate = frame["tenure"] * frame["MonthlyCharges"]
            frame["TotalCharges"] = frame["TotalCharges"].fillna(estimate)
        return frame

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_column_transformer() -> ColumnTransformer:
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric_feature_names()),
            ("categorical", categorical_steps, categorical_feature_names()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("derive_total_charges", TotalChargesImputer()),
        ("columns", build_column_transformer()),
    ])


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    return [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]

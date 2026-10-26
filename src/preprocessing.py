"""Leakage-safe Scikit-learn preprocessing and estimator pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build imputers and encoders from training columns only."""
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [name for name in frame.columns if name not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(frame: pd.DataFrame, estimator) -> Pipeline:
    """Combine preprocessing and estimator so transformations fit with the model."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(frame)),
            ("classifier", estimator),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Get transformed feature names from a fitted pipeline."""
    names: Sequence[str] = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [str(name) for name in names]
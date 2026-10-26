"""Train, compare, evaluate, and persist the ChurnGuard model."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import DECISION_THRESHOLD, MODEL_ARTIFACT_PATH, PROJECT_ROOT
from src.data_validation import load_training_data
from src.evaluate import evaluate_probabilities, save_evaluation_plots, write_json
from src.features import build_feature_schema
from src.preprocessing import build_pipeline, get_feature_names


SEED = 42
logger = logging.getLogger(__name__)


def _make_estimators(y_train: pd.Series, seed: int = SEED) -> dict[str, Any]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=negatives / max(positives, 1),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
        ),
    }


def _feature_importance(pipeline, feature_names: list[str]) -> np.ndarray:
    estimator = pipeline.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:
        raise TypeError(f"Unsupported estimator for importance: {type(estimator).__name__}")
    if len(importance) != len(feature_names):
        raise ValueError("Feature importance and transformed feature names do not align.")
    return importance


def train(
    csv_path: Path | None = None,
    artifact_path: Path | None = None,
    threshold: float = DECISION_THRESHOLD,
    seed: int = SEED,
    estimators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run stratified train/validation/test comparison and save measured artifacts."""
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    features, target, data_metadata = load_training_data(csv_path)
    if target.value_counts().min() < 3:
        raise ValueError("Each target class needs at least three examples for stratified splits.")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=0.4, random_state=seed, stratify=target
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.5, random_state=seed, stratify=y_holdout
    )

    validation_probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    candidates = estimators or _make_estimators(y_train, seed)
    for name, estimator in candidates.items():
        pipeline = build_pipeline(x_train, estimator)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        validation_probabilities[name] = probabilities
        fitted[name] = pipeline
        comparison[name] = evaluate_probabilities(
            y_validation.to_numpy(), probabilities, threshold
        )
        logger.info("%s validation ROC-AUC: %s", name, comparison[name]["roc_auc"])

    eligible = {
        name: result["roc_auc"]
        for name, result in comparison.items()
        if result["roc_auc"] is not None
    }
    if not eligible:
        raise ValueError("Validation ROC-AUC could not be calculated for any model.")
    selected_name = max(eligible, key=eligible.get)
    selected_pipeline = fitted[selected_name]
    test_probabilities = selected_pipeline.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test.to_numpy(), test_probabilities, threshold)

    feature_names = get_feature_names(selected_pipeline)
    importance = _feature_importance(selected_pipeline, feature_names)
    output_model_path = artifact_path or MODEL_ARTIFACT_PATH
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_model_path.parent / "metrics.json"
    metadata_path = output_model_path.parent / "metadata.json"
    report_dir = PROJECT_ROOT / "artifacts" / "training"
    training_plots = save_evaluation_plots(
        y_validation.to_numpy(),
        validation_probabilities,
        y_test.to_numpy(),
        test_probabilities,
        threshold,
        feature_names,
        importance,
        report_dir,
    )

    feature_schema = build_feature_schema(x_train)

    trained_at = datetime.now(timezone.utc)
    metadata = {
        "model_version": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": trained_at.isoformat(),
        "model_type": selected_name,
        "selection_metric": "validation_roc_auc",
        "validation_roc_auc": comparison[selected_name]["roc_auc"],
        "decision_threshold": threshold,
        "risk_bands": {"low_upper_bound": 0.3, "medium_upper_bound": 0.6},
        "random_seed": seed,
        "split_rows": {
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
        "target_distribution": {
            str(label): int(count) for label, count in target.value_counts().items()
        },
        "data": data_metadata,
        "preprocessing": {
            "numeric_features": x_train.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_features": [
                name
                for name in x_train.columns
                if name not in x_train.select_dtypes(include=["number"]).columns
            ],
            "numeric_transform": "median imputation followed by standard scaling",
            "categorical_transform": (
                "most-frequent imputation followed by one-hot encoding; "
                "unknown categories ignored"
            ),
            "fit_split": "training only",
        },
        "library_versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "shap": version("shap"),
        },
        "training_medians": {
            "monthly_charges": (
                float(x_train["Monthly Charges"].median())
                if "Monthly Charges" in x_train
                else (
                    float(x_train["MonthlyCharges"].median())
                    if "MonthlyCharges" in x_train
                    else None
                )
            )
        },
        "feature_schema": feature_schema,
        "feature_names": feature_names,
        "global_feature_importance": [
            {"feature": feature, "importance": float(value)}
            for feature, value in sorted(
                zip(feature_names, importance), key=lambda item: item[1], reverse=True
            )
        ],
    }
    metrics = {
        "selected_model": selected_name,
        "selection_metric": "validation_roc_auc",
        "validation_comparison": comparison,
        "held_out_test": test_metrics,
        "plots": training_plots,
    }
    background_sample = x_train.sample(min(100, len(x_train)), random_state=seed)
    background = selected_pipeline.named_steps["preprocessor"].transform(background_sample)
    if hasattr(background, "toarray"):
        background = background.toarray()
    artifact = {
        "pipeline": selected_pipeline,
        "metadata": metadata,
        "metrics": metrics,
        "background": np.asarray(background, dtype=float),
    }
    joblib.dump(artifact, output_model_path)
    write_json(metrics_path, metrics)
    write_json(metadata_path, metadata)
    write_json(report_dir / "metrics.json", metrics)
    return {
        "artifact_path": str(output_model_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
        "selected_model": selected_name,
        "validation_roc_auc": comparison[selected_name]["roc_auc"],
        "held_out_test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, help="Optional explicit raw CSV path.")
    parser.add_argument("--artifact-path", type=Path, help="Model artifact output path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
        help="Decision threshold for this run (default 0.5; business review required).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = train(
        arguments.csv_path,
        arguments.artifact_path,
        arguments.threshold,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
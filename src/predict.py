"""Validated prediction service independent of the web API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_ARTIFACT_PATH
from src.data_validation import EXCLUDED_COLUMNS, TARGET_ALIASES
from src.explain import ExplainabilityError, explain_prediction


class PredictionValidationError(ValueError):
    """Raised when provided prediction data does not match the trained schema."""


def risk_category(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


def retention_suggestions(features: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Create transparent review prompts from attributes present in the input."""
    suggestions = []
    contract = features.get("Contract")
    if isinstance(contract, str) and contract.casefold() == "month-to-month":
        suggestions.append(
            "Review whether an annual contract option suits the customer's needs; "
            "do not assume a contract change is beneficial."
        )
    support = features.get("Tech Support")
    if isinstance(support, str) and support.casefold() in {"no", "no internet service"}:
        suggestions.append(
            "Check whether the customer needs technical support or help resolving "
            "a service issue."
        )

    monthly = next(
        (
            features.get(name)
            for name in ("Monthly Charges", "MonthlyCharges")
            if name in features
        ),
        None,
    )
    median = metadata.get("training_medians", {}).get("monthly_charges")
    if monthly is not None and median is not None and float(monthly) > float(median):
        suggestions.append(
            "Review the bill and plan fit with the customer; a higher charge "
            "is not proof of dissatisfaction."
        )
    if not suggestions:
        suggestions.append(
            "Use a human review to understand the customer's circumstances before "
            "considering any retention offer."
        )
    return suggestions


class ChurnPredictor:
    """Load an artifact once and validate every prediction against its schema."""

    def __init__(self, artifact_path: Path | None = None):
        path = artifact_path or MODEL_ARTIFACT_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or not {
            "pipeline",
            "metadata",
            "metrics",
            "background",
        }.issubset(artifact):
            raise ValueError(f"Model artifact has an unsupported format: {path}")
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact["metadata"]
        self.metrics = artifact["metrics"]
        self.background = artifact["background"]
        self.feature_schema = self.metadata.get("feature_schema", {})
        self.feature_names = list(self.feature_schema)
        self.threshold = float(self.metadata["decision_threshold"])
        if not self.feature_names or not 0 < self.threshold < 1:
            raise ValueError("Model artifact is missing feature schema or a valid threshold.")

    def validate_features(self, features: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(features, dict):
            raise PredictionValidationError("Customer features must be a JSON object.")
        prohibited = set(TARGET_ALIASES) | set(EXCLUDED_COLUMNS)
        supplied = set(features)
        forbidden = sorted(supplied & prohibited)
        if forbidden:
            raise PredictionValidationError(
                f"Target, identifier, or excluded fields are not prediction features: {forbidden}"
            )
        expected = set(self.feature_names)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing features: {missing}")
            if unknown:
                details.append(f"unknown features: {unknown}")
            raise PredictionValidationError("; ".join(details))

        normalized: dict[str, Any] = {}
        for name in self.feature_names:
            value = features[name]
            if value is None:
                normalized[name] = np.nan
                continue
            spec = self.feature_schema[name]
            if spec["type"] == "number":
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise PredictionValidationError(
                        f"Feature '{name}' must be numeric or null."
                    ) from exc
                if not np.isfinite(numeric_value):
                    raise PredictionValidationError(f"Feature '{name}' must be finite.")
                normalized[name] = numeric_value
            elif isinstance(value, (str, int, float, bool)):
                normalized[name] = str(value)
            else:
                raise PredictionValidationError(
                    f"Feature '{name}' must be a string, number, or null."
                )
        return normalized

    def predict(
        self, features: dict[str, Any], include_explanation: bool = True
    ) -> dict[str, Any]:
        normalized = self.validate_features(features)
        row = pd.DataFrame([normalized], columns=self.feature_names)
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        if not 0 <= probability <= 1:
            raise RuntimeError("The model returned a probability outside [0, 1].")
        prediction = {
            "probability": probability,
            "predicted_class": "Yes" if probability >= self.threshold else "No",
            "risk_category": risk_category(probability),
            "threshold": self.threshold,
            "model_version": self.metadata["model_version"],
            "explanation": None,
            "explanation_warning": None,
            "retention_suggestions": retention_suggestions(normalized, self.metadata),
        }
        if include_explanation:
            try:
                prediction["explanation"] = explain_prediction(
                    self.pipeline, self.background, row
                )
            except ExplainabilityError as exc:
                prediction["explanation_warning"] = str(exc)
        return prediction

    def predict_batch(
        self, records: list[dict[str, Any]], include_explanation: bool = False
    ) -> list[dict[str, Any]]:
        return [
            self.predict(record, include_explanation=include_explanation)
            for record in records
        ]
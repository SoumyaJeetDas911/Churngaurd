"""SHAP explanations for the selected fitted estimator."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy.sparse import issparse

from src.preprocessing import get_feature_names


class ExplainabilityError(RuntimeError):
    """Raised when a local explanation cannot be computed reliably."""


def explain_prediction(pipeline, background: np.ndarray, features) -> dict[str, Any]:
    """Return positive-class SHAP contributions using transformed feature names."""
    try:
        shap = importlib.import_module("shap")
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(features)
        if issparse(transformed):
            transformed = transformed.toarray()
        names = get_feature_names(pipeline)

        if estimator.__class__.__name__ == "LogisticRegression":
            explainer = shap.LinearExplainer(estimator, background)
        else:
            explainer = shap.TreeExplainer(estimator, data=background)
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]
        else:
            raise ExplainabilityError(
                f"Unexpected SHAP output shape: {values.shape}."
            )
        if len(values) != len(names):
            raise ExplainabilityError("SHAP values do not align with encoded feature names.")

        expected_value = np.asarray(explanation.base_values)
        if expected_value.ndim >= 2:
            base_value = float(expected_value.reshape(-1, expected_value.shape[-1])[0, -1])
        else:
            base_value = float(expected_value.reshape(-1)[0])
        order = np.argsort(np.abs(values))[-8:][::-1]
        return {
            "available": True,
            "method": "SHAP",
            "scale": "raw model output",
            "base_value": base_value,
            "features": [
                {
                    "feature": names[index],
                    "contribution": float(values[index]),
                    "direction": (
                        "increases model output"
                        if values[index] > 0
                        else "decreases model output"
                    ),
                }
                for index in order
                if values[index] != 0
            ],
            "caveat": "Attributions describe model contribution, not causation.",
        }
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ExplainabilityError(
            f"SHAP explanation unavailable: {type(exc).__name__}: {exc}"
        ) from exc

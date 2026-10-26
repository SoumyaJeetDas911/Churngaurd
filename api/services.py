"""API business logic separated from route definitions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.database import add_prediction
from src.predict import ChurnPredictor


logger = logging.getLogger("churnguard.api")


def create_prediction(
    predictor: ChurnPredictor,
    session: Session | None,
    features: dict[str, Any],
    customer_label: str | None,
) -> tuple[dict[str, Any], Any | None, str | None]:
    """Predict and optionally persist the minimal history record."""
    clean_features = predictor.validate_features(features)
    prediction = predictor.predict(clean_features)
    if session is None:
        return prediction, None, "Prediction history is unavailable."
    values = {
        "timestamp": datetime.now(timezone.utc),
        "customer_label": customer_label,
        "probability": prediction["probability"],
        "predicted_class": prediction["predicted_class"],
        "risk_category": prediction["risk_category"],
        "threshold": prediction["threshold"],
        "model_version": prediction["model_version"],
        "customer_features": {
            name: (
                None
                if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for name, value in clean_features.items()
        },
    }
    try:
        return prediction, add_prediction(session, values), None
    except SQLAlchemyError:
        logger.exception("Prediction succeeded, but history persistence failed.")
        return (
            prediction,
            None,
            "Prediction succeeded, but history could not be saved.",
        )

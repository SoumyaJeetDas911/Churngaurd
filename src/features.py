"""Model input schema helpers shared by training and inference artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_feature_schema(
    frame: pd.DataFrame, category_limit: int = 100
) -> dict[str, dict[str, Any]]:
    """Describe numeric/categorical inputs and observed categorical choices."""
    schema: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            schema[str(column)] = {"type": "number", "categories": []}
        else:
            categories = sorted(frame[column].dropna().astype(str).unique().tolist())
            schema[str(column)] = {
                "type": "string",
                "categories": categories if len(categories) <= category_limit else [],
            }
    return schema
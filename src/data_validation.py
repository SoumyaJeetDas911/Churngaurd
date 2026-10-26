"""Data loading and validation for the approved Telco dataset variant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.inspect_data import find_raw_csv


TARGET_ALIASES = ("Churn", "Churn Label")
TOTAL_CHARGES_ALIASES = ("TotalCharges", "Total Charges")
EXCLUDED_COLUMNS = {
    "CustomerID": "unique customer identifier",
    "customerID": "unique customer identifier",
    "customer_id": "unique customer identifier",
    "Customer ID": "unique customer identifier",
    "Churn Value": "duplicate encoding of the target",
    "Churn Score": "outcome-related score with leakage risk",
    "Churn Reason": "post-outcome explanation",
    "CLTV": "potential outcome-derived feature excluded pending provenance review",
    "Country": "geographic attribute excluded for privacy and generalization",
    "State": "geographic attribute excluded for privacy and generalization",
    "City": "geographic attribute excluded for privacy and generalization",
    "Zip Code": "geographic attribute excluded for privacy and generalization",
    "Lat Long": "precise location excluded for privacy",
    "Latitude": "precise location excluded for privacy",
    "Longitude": "precise location excluded for privacy",
    "Count": "dataset row-count field, not a customer attribute",
}


def validate_and_prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Validate target, convert charges, and remove documented non-predictors."""
    target = next((name for name in TARGET_ALIASES if name in frame.columns), None)
    if target is None:
        raise ValueError(
            "Missing churn target. Expected one of "
            f"{TARGET_ALIASES}; found columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("The dataset contains no rows.")

    labels = frame[target].astype("string").str.strip().str.casefold()
    target_map = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    unknown_labels = sorted(set(labels.dropna().unique()) - set(target_map))
    if unknown_labels:
        raise ValueError(f"Unrecognized values in target '{target}': {unknown_labels}")
    if labels.isna().any():
        raise ValueError(f"Target column '{target}' contains missing values.")
    y = labels.map(target_map).astype("int8").rename("churn")
    if y.nunique() != 2:
        raise ValueError("Training requires both churn classes in the target.")
    if "Churn Value" in frame.columns:
        parallel_target = pd.to_numeric(frame["Churn Value"], errors="coerce")
        matches_target = (
            not parallel_target.isna().any()
            and parallel_target.isin([0, 1]).all()
            and parallel_target.astype("int8").reset_index(drop=True).equals(
                y.reset_index(drop=True)
            )
        )
        if not matches_target:
            raise ValueError(
                "'Churn Value' is documented as a duplicate target, but does not "
                "match the validated churn label."
            )

    features = frame.drop(columns=[target]).copy()
    charges_column = next(
        (name for name in TOTAL_CHARGES_ALIASES if name in features.columns), None
    )
    invalid_charge_count = 0
    if charges_column:
        raw_charges = features[charges_column].astype("string").str.strip()
        blank_charges = raw_charges.isna() | raw_charges.eq("")
        converted_charges = pd.to_numeric(
            raw_charges.replace("", pd.NA), errors="coerce"
        )
        invalid_charge_count = int((converted_charges.isna() & ~blank_charges).sum())
        features[charges_column] = converted_charges

    removed_columns = {
        column: reason
        for column, reason in EXCLUDED_COLUMNS.items()
        if column in features.columns
    }
    features = features.drop(columns=list(removed_columns), errors="ignore")
    constant_columns = [
        column for column in features.columns if features[column].nunique(dropna=False) <= 1
    ]
    for column in constant_columns:
        removed_columns[column] = "constant feature"
    features = features.drop(columns=constant_columns)
    if features.empty:
        raise ValueError("No usable predictor features remain after validation.")

    metadata = {
        "target_column": target,
        "total_charges_column": charges_column,
        "total_charges_missing_after_conversion": (
            int(features[charges_column].isna().sum()) if charges_column else None
        ),
        "total_charges_invalid_values": invalid_charge_count,
        "excluded_columns": removed_columns,
        "source_rows": int(len(frame)),
        "duplicate_rows_reported": int(frame.duplicated().sum()),
    }
    return features, y, metadata


def load_training_data(csv_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load and validate the selected raw CSV without modifying it."""
    source = find_raw_csv(csv_path)
    frame = pd.read_csv(source)
    features, target, metadata = validate_and_prepare(frame)
    metadata["source_file"] = source.name
    return features, target, metadata

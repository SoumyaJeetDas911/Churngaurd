"""Inspect the raw churn dataset and generate reproducible EDA artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_CANDIDATES = (PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "Data" / "Raw")
TARGET_CANDIDATES = ("Churn", "Churn Label")
ALIASES = {
    "customer_id": ("customerID", "CustomerID"),
    "target": TARGET_CANDIDATES,
    "contract": ("Contract",),
    "tenure": ("tenure", "Tenure Months"),
    "monthly_charges": ("MonthlyCharges", "Monthly Charges"),
    "total_charges": ("TotalCharges", "Total Charges"),
    "payment_method": ("PaymentMethod", "Payment Method"),
}


def find_raw_csv(csv_path: Path | None = None) -> Path:
    """Find the sole raw CSV, or validate the explicit path supplied by the user."""
    if csv_path is not None:
        resolved = csv_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {resolved}")
        return resolved

    raw_dir = next((path for path in RAW_DIR_CANDIDATES if path.is_dir()), None)
    if raw_dir is None:
        expected = " or ".join(str(path) for path in RAW_DIR_CANDIDATES)
        raise FileNotFoundError(f"Raw data directory not found. Checked: {expected}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) != 1:
        names = ", ".join(path.name for path in csv_files) or "none"
        raise ValueError(
            f"Expected exactly one CSV in {raw_dir}; found {len(csv_files)} ({names}). "
            "Pass --csv-path to select a dataset explicitly."
        )
    return csv_files[0]


def resolve_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    """Resolve known dataset spelling variants without guessing column names."""
    return next((name for name in aliases if name in columns), None)


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the CSV and record schema differences from the expected Telco format."""
    frame = pd.read_csv(csv_path)
    target = resolve_column(frame.columns, TARGET_CANDIDATES)
    if target is None:
        raise ValueError(
            "The dataset must contain a target column named 'Churn' for model training. "
            f"Observed columns: {', '.join(map(str, frame.columns))}"
        )

    issues: list[str] = []
    if "Churn" not in frame.columns:
        issues.append(
            f"Expected target column 'Churn'; found '{target}' instead. "
            "The application uses the explicitly approved mapping for this dataset variant."
        )
    if "TotalCharges" not in frame.columns:
        observed_total = resolve_column(frame.columns, ("Total Charges",))
        if observed_total:
            issues.append(
                f"Expected 'TotalCharges'; found '{observed_total}'. "
                "Its whitespace/invalid-value conversion is included in this data audit."
            )
    if "Churn Value" in frame.columns and target != "Churn":
        issues.append(
            "'Churn Value' duplicates the churn label and must not be used as a predictor."
        )

    total_column = resolve_column(frame.columns, ALIASES["total_charges"])
    if total_column:
        original_missing = int(frame[total_column].isna().sum())
        cleaned = pd.to_numeric(
            frame[total_column].astype("string").str.strip().replace("", pd.NA),
            errors="coerce",
        )
        frame[total_column] = cleaned
        issues.append(
            f"Converted '{total_column}' to numeric; "
            f"{int(cleaned.isna().sum())} values are missing after conversion "
            f"({original_missing} were already missing)."
        )

    return frame, {
        "target_column_used_for_eda": target,
        "expected_target_column": "Churn",
        "training_target_mapping": (
            f"{target} -> churn" if target != "Churn" else "Churn -> churn"
        ),
        "schema_compatible_for_training": target in TARGET_CANDIDATES,
        "schema_issues": issues,
    }


def build_profile(frame: pd.DataFrame, schema: dict[str, Any], csv_path: Path) -> dict[str, Any]:
    """Return dataset quality and descriptive statistics as JSON-ready values."""
    target = schema["target_column_used_for_eda"]
    counts = frame[target].value_counts(dropna=False)
    percentages = frame[target].value_counts(normalize=True, dropna=False) * 100
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    category_cardinalities = {
        str(column): int(frame[column].nunique(dropna=True))
        for column in categorical_columns
    }
    category_values = {
        str(column): sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in categorical_columns
        if frame[column].nunique(dropna=True) <= 50
    }
    observations = [
        (
            f"{label}: {int(count)} of {len(frame)} rows "
            f"({float(percentages[label]):.2f}%)."
        )
        for label, count in counts.items()
    ]

    contract = resolve_column(frame.columns, ALIASES["contract"])
    segment_churn: dict[str, list[dict[str, Any]]] = {}
    for segment_column in (
        contract,
        resolve_column(frame.columns, ALIASES["payment_method"]),
    ):
        if not segment_column:
            continue
        segment_rows = []
        for segment_value, group in frame.groupby(
            segment_column, dropna=False, observed=True
        ):
            label_counts = group[target].value_counts(dropna=False)
            positive_label = next(
                (
                    label
                    for label in label_counts.index
                    if str(label).strip().casefold() in {"yes", "1", "true"}
                ),
                None,
            )
            positive_count = (
                int(label_counts[positive_label]) if positive_label is not None else None
            )
            segment_rows.append(
                {
                    "segment": str(segment_value),
                    "customers": int(len(group)),
                    "positive_label_count": positive_count,
                    "positive_label_percentage": (
                        round(positive_count * 100 / len(group), 2)
                        if positive_count is not None and len(group)
                        else None
                    ),
                    "label_counts": {
                        str(label): int(count) for label, count in label_counts.items()
                    },
                }
            )
        segment_churn[segment_column] = segment_rows

    if contract:
        for contract_name, group in frame.groupby(contract, dropna=False, observed=True):
            label_counts = group[target].value_counts(dropna=False)
            label_percentages = group[target].value_counts(
                normalize=True, dropna=False
            ) * 100
            churn_label = next(
                (
                    label
                    for label in label_counts.index
                    if str(label).strip().casefold() in {"yes", "1", "true"}
                ),
                None,
            )
            if churn_label is not None:
                observations.append(
                    f"{contract_name} contracts: {int(label_counts[churn_label])} of "
                    f"{len(group)} rows ({float(label_percentages[churn_label]):.2f}%) "
                    "have a positive churn label."
                )

    for key, display_name in (
        ("tenure", "Median tenure"),
        ("monthly_charges", "Median monthly charges"),
    ):
        column = resolve_column(frame.columns, ALIASES[key])
        if column:
            values = pd.to_numeric(frame[column], errors="coerce")
            medians = values.groupby(frame[target], dropna=False).median()
            for label, median in medians.items():
                if pd.notna(median):
                    unit = " months" if key == "tenure" else ""
                    observations.append(
                        f"{display_name} for {label}: {float(median):.2f}{unit}."
                    )

    return {
        "source_file": csv_path.name,
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if count > 0
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_column": target,
        "target_counts": {str(label): int(count) for label, count in counts.items()},
        "target_percentages": {
            str(label): round(float(value), 4) for label, value in percentages.items()
        },
        "categorical_cardinalities": category_cardinalities,
        "categorical_values": category_values,
        "categorical_values_omitted_above_50_categories": sorted(
            column
            for column, cardinality in category_cardinalities.items()
            if cardinality > 50
        ),
        "descriptive_observations": observations,
        "segment_churn": segment_churn,
        "schema": schema,
    }


def _save_chart(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def create_charts(frame: pd.DataFrame, target: str, output_dir: Path) -> list[str]:
    """Create descriptive charts only for fields present in the observed dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    generated: list[str] = []

    plt.figure(figsize=(7, 5))
    sns.countplot(data=frame, x=target, order=frame[target].value_counts().index)
    plt.title("Customer Churn Label Distribution")
    plt.xlabel("Churn label")
    plt.ylabel("Number of customers")
    filename = "churn_distribution.png"
    _save_chart(output_dir / filename)
    generated.append(filename)

    contract = resolve_column(frame.columns, ALIASES["contract"])
    if contract:
        plt.figure(figsize=(9, 5))
        sns.countplot(data=frame, x=contract, hue=target)
        plt.title("Churn Label by Contract Type")
        plt.xlabel("Contract type")
        plt.ylabel("Number of customers")
        plt.xticks(rotation=15, ha="right")
        filename = "churn_by_contract.png"
        _save_chart(output_dir / filename)
        generated.append(filename)

    tenure = resolve_column(frame.columns, ALIASES["tenure"])
    if tenure:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=frame,
            x=tenure,
            hue=target,
            element="step",
            stat="density",
            common_norm=False,
            bins=30,
        )
        plt.title("Tenure Distribution by Churn Label")
        plt.xlabel("Tenure (months)")
        plt.ylabel("Density")
        filename = "tenure_by_churn.png"
        _save_chart(output_dir / filename)
        generated.append(filename)

    monthly = resolve_column(frame.columns, ALIASES["monthly_charges"])
    if monthly:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=frame, x=target, y=monthly)
        plt.title("Monthly Charges by Churn Label")
        plt.xlabel("Churn label")
        plt.ylabel("Monthly charges")
        filename = "monthly_charges_by_churn.png"
        _save_chart(output_dir / filename)
        generated.append(filename)

    payment = resolve_column(frame.columns, ALIASES["payment_method"])
    if payment:
        plt.figure(figsize=(11, 5))
        sns.countplot(data=frame, x=payment, hue=target)
        plt.title("Churn Label by Payment Method")
        plt.xlabel("Payment method")
        plt.ylabel("Number of customers")
        plt.xticks(rotation=20, ha="right")
        filename = "churn_by_payment_method.png"
        _save_chart(output_dir / filename)
        generated.append(filename)

    numeric_columns = [
        column
        for column in (
            tenure,
            monthly,
            resolve_column(frame.columns, ALIASES["total_charges"]),
        )
        if column and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_columns) > 1:
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            frame[numeric_columns].corr(),
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
        )
        plt.title("Numeric Feature Correlations")
        filename = "numeric_feature_correlations.png"
        _save_chart(output_dir / filename)
        generated.append(filename)

    return generated


def run_inspection(csv_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Run data validation, write a JSON profile, and save available charts."""
    source = find_raw_csv(csv_path)
    destination = output_dir or PROJECT_ROOT / "artifacts" / "eda"
    frame, schema = load_dataset(source)
    profile = build_profile(frame, schema, source)
    profile["charts"] = create_charts(
        frame, schema["target_column_used_for_eda"], destination
    )
    destination.mkdir(parents=True, exist_ok=True)

    report_path = destination / "data_profile.json"
    try:
        profile["report_file"] = str(report_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        profile["report_file"] = report_path.name
    report_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Optional explicit CSV path; otherwise require one CSV in the raw-data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: artifacts/eda).",
    )
    arguments = parser.parse_args()

    profile = run_inspection(arguments.csv_path, arguments.output_dir)
    print(json.dumps(profile, indent=2))
    if not profile["schema"]["schema_compatible_for_training"]:
        print(
            "\nTRAINING BLOCKED: the observed dataset schema does not include the required "
            "'Churn' target column. No model was trained."
        )


if __name__ == "__main__":
    main()

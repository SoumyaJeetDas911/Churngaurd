"""Streamlit dashboard. Predictions and analytics are served by the FastAPI backend."""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.config import BACKEND_URL, MAX_BATCH_SIZE

BACKEND_URL = os.getenv("BACKEND_URL", BACKEND_URL).rstrip("/")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", str(MAX_BATCH_SIZE)))


def request_api(method: str, path: str, payload: dict[str, Any] | None = None):
    """Call the configured API and return JSON or show a user-facing error."""
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", json=payload, timeout=20)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            f"ChurnGuard API is offline at {BACKEND_URL}. Start it with "
            "`python -m uvicorn api.main:app --reload`."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("The ChurnGuard API request timed out. Check API status and retry.")
        return None
    except requests.exceptions.HTTPError:
        try:
            message = response.json().get("detail", response.text)
        except ValueError:
            message = response.text
        st.error(f"API request failed ({response.status_code}): {message}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the ChurnGuard API: {exc}")
        return None
    return response.json()


def validate_csv_schema(
    frame: pd.DataFrame, feature_names: list[str]
) -> tuple[bool, list[str]]:
    columns = set(frame.columns)
    expected = set(feature_names)
    missing = sorted(expected - columns)
    unknown = sorted(columns - expected - {"customer_label"})
    errors = []
    if missing:
        errors.append(f"Missing required feature columns: {missing}")
    if unknown:
        errors.append(f"Unexpected columns: {unknown}")
    return not errors, errors


def _render_overview():
    health = request_api("GET", "/health")
    if health:
        st.metric("Model status", "Ready" if health["model_ready"] else "Unavailable")
        st.caption(
            "Prediction history: "
            + ("available" if health["history_ready"] else "unavailable")
        )
    training = request_api("GET", "/analytics/training")
    if training and training.get("shape"):
        st.metric(
            "Training dataset rows",
            training["shape"]["rows"],
            help="Actual rows in the inspected source dataset.",
        )
        counts = training.get("target_counts", {})
        if counts:
            st.plotly_chart(
                px.bar(
                    x=list(counts),
                    y=list(counts.values()),
                    labels={"x": "Observed churn label", "y": "Customers"},
                    title="Observed training-data label counts",
                ),
                width="stretch",
            )
    summary = request_api("GET", "/analytics/summary")
    if summary:
        left, middle, right = st.columns(3)
        left.metric("Historical predictions", summary["total_predictions"])
        middle.metric(
            "Mean predicted probability",
            f"{summary['average_probability']:.1%}"
            if summary["average_probability"] is not None
            else "—",
        )
        right.metric("Predicted churn", summary["predicted_class_counts"].get("Yes", 0))
        if summary["risk_counts"]:
            chart = px.bar(
                x=list(summary["risk_counts"]),
                y=list(summary["risk_counts"].values()),
                labels={"x": "Risk category", "y": "Predictions"},
                title="Historical predictions by risk category",
            )
            st.plotly_chart(chart, width="stretch")
    metrics = request_api("GET", "/metrics")
    if metrics:
        test = metrics["held_out_test"]
        cols = st.columns(5)
        for column, name in zip(
            cols, ("roc_auc", "pr_auc", "precision", "recall", "f1")
        ):
            value = test.get(name)
            column.metric(
                name.replace("_", " ").title(),
                f"{value:.3f}" if value is not None else "—",
            )


def _render_prediction(features: dict[str, Any], label: str | None):
    result = request_api(
        "POST",
        "/predict",
        {"features": features, "customer_label": label or None},
    )
    if result is None:
        return
    st.subheader(f"Predicted churn: {result['predicted_class']}")
    left, right = st.columns(2)
    left.metric("Estimated churn probability", f"{result['probability']:.1%}")
    right.metric("Risk category", result["risk_category"].title())
    st.caption(
        f"Decision threshold: {result['threshold']:.2f}. "
        "The threshold is a configuration choice, not a universal business rule."
    )
    st.progress(result["probability"])
    if result.get("explanation"):
        st.markdown("**Model contribution (SHAP)**")
        st.caption(f"Attribution scale: {result['explanation']['scale']}.")
        st.caption(result["explanation"]["caveat"])
        st.dataframe(result["explanation"]["features"], width="stretch")
    elif result.get("explanation_warning"):
        st.warning(result["explanation_warning"])
    st.markdown("**Possible review prompts**")
    for suggestion in result["retention_suggestions"]:
        st.write(f"- {suggestion}")
    if not result["history_saved"]:
        st.warning(result.get("history_warning") or "Prediction was not saved to history.")


def _render_single_prediction(features_info: list[dict[str, Any]]):
    with st.form("single_prediction"):
        customer_label = st.text_input("Optional customer label", max_chars=128)
        values: dict[str, Any] = {}
        for spec in features_info:
            name = spec["name"]
            if spec["type"] == "number":
                value = st.number_input(name, value=0.0, format="%.4f")
            elif spec.get("categories"):
                value = st.selectbox(name, spec["categories"])
            else:
                value = st.text_input(name)
            values[name] = value
        submit = st.form_submit_button("Estimate churn risk")
    if submit:
        _render_prediction(values, customer_label)


def _render_batch_prediction(feature_names: list[str]):
    uploaded = st.file_uploader("Upload a CSV of customer features", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read this CSV: {exc}")
        return
    valid, errors = validate_csv_schema(frame, feature_names)
    if not valid:
        for error in errors:
            st.error(error)
        return
    if frame.empty:
        st.error("The uploaded CSV contains no customer rows.")
        return
    if len(frame) > MAX_BATCH_SIZE:
        st.error(f"This CSV has {len(frame)} rows; the configured limit is {MAX_BATCH_SIZE}.")
        return
    st.caption(f"{len(frame)} rows validated. The uploaded file will not be modified.")
    if st.button("Predict batch"):
        labels = (
            frame["customer_label"].tolist()
            if "customer_label" in frame
            else [None] * len(frame)
        )
        feature_frame = frame.drop(columns=["customer_label"], errors="ignore")
        records = []
        for row, label in zip(feature_frame.to_dict(orient="records"), labels):
            clean_row = {
                name: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for name, value in row.items()
            }
            clean_label = None if pd.isna(label) else label
            records.append(
                {"features": clean_row, "customer_label": clean_label}
            )
        result = request_api("POST", "/predict/batch", {"records": records})
        if result:
            output = frame.copy()
            output["probability"] = [item["probability"] for item in result["results"]]
            output["predicted_class"] = [item["predicted_class"] for item in result["results"]]
            output["risk_category"] = [item["risk_category"] for item in result["results"]]
            st.dataframe(output, width="stretch")
            st.download_button(
                "Download prediction results",
                output.to_csv(index=False).encode("utf-8"),
                file_name="churnguard_predictions.csv",
                mime="text/csv",
            )


def _render_performance():
    result = request_api("GET", "/metrics")
    if not result:
        return
    comparison = result["validation_comparison"]
    st.subheader("Validation model comparison")
    table = pd.DataFrame(
        {
            name: {
                metric: data.get(metric)
                for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy")
            }
            for name, data in comparison.items()
        }
    ).T
    st.dataframe(table, width="stretch")
    st.caption("Models are selected by validation ROC-AUC. Test metrics are from a held-out split.")
    test = result["held_out_test"]
    st.subheader("Held-out test metrics")
    st.json({key: value for key, value in test.items() if key != "classification_report"})
    matrix = test["confusion_matrix"]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted no churn", "Predicted churn"],
            y=["Actual no churn", "Actual churn"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    figure.update_layout(title="Held-out test confusion matrix")
    st.plotly_chart(figure, width="stretch")


def _render_insights():
    result = request_api("GET", "/analytics/training")
    if not result:
        st.info("Run dataset inspection to make observed training-data segments available.")
        return
    st.caption(
        "These are descriptive patterns in the training dataset, not causes or "
        "model explanations."
    )
    st.write(result.get("descriptive_observations", []))
    for segment_name, rows in result.get("segment_churn", {}).items():
        st.subheader(segment_name.replace("_", " ").title())
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _render_history():
    page = st.number_input("Page", min_value=1, value=1, step=1)
    limit = 50
    result = request_api("GET", f"/history?limit={limit}&offset={(int(page) - 1) * limit}")
    if not result:
        return
    st.caption(f"{result['total']} total predictions")
    rows = [
        {
            "id": item["id"],
            "timestamp": item["timestamp"],
            "customer_label": item["customer_label"],
            "probability": item["probability"],
            "predicted_class": item["predicted_class"],
            "risk_category": item["risk_category"],
            "model_version": item["model_version"],
        }
        for item in result["items"]
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch")
    if result["items"]:
        record_id = st.selectbox("View record details", [item["id"] for item in result["items"]])
        detail = request_api("GET", f"/history/{record_id}")
        if detail:
            st.json(detail)


def _render_about():
    info = request_api("GET", "/model/info")
    if info:
        st.subheader("Model card summary")
        st.write(f"Model type: {info['model_type']}")
        st.write(f"Trained at: {info['trained_at']}")
        st.write(f"Decision threshold: {info['decision_threshold']:.2f}")
        st.write(
            f"Validation ROC-AUC: {info['validation_roc_auc']:.3f}"
            if info["validation_roc_auc"] is not None
            else "Validation ROC-AUC unavailable"
        )
    st.markdown(
        """
        **Intended use:** educational decision support with human review only.

        **Limitations:** historical labels may not represent current customers;
        probabilities may not be calibrated; false positives and false negatives
        both matter. Explanations describe model contribution, not causation.
        Retention suggestions are hypotheses to test, not guaranteed outcomes.

        **Privacy:** prediction history stores supplied predictor attributes and
        is not access-controlled. Do not use real customer data until privacy,
        retention, encryption, access-control, and security reviews are complete.
        """
    )


def main():
    st.set_page_config(page_title="ChurnGuard", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("ChurnGuard")
    st.caption("Customer churn risk analysis and human-reviewed retention intelligence")
    navigation = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Single Customer Prediction",
            "Batch Prediction",
            "Model Performance",
            "Churn Insights",
            "Prediction History",
            "About / Model Card",
        ],
    )
    features_response = request_api("GET", "/features")
    if features_response is None and navigation not in {"Overview", "About / Model Card"}:
        return

    if navigation == "Overview":
        _render_overview()
    elif navigation == "Single Customer Prediction":
        _render_single_prediction(features_response["features"])
    elif navigation == "Batch Prediction":
        _render_batch_prediction([item["name"] for item in features_response["features"]])
    elif navigation == "Model Performance":
        _render_performance()
    elif navigation == "Churn Insights":
        _render_insights()
    elif navigation == "Prediction History":
        _render_history()
    else:
        _render_about()


if __name__ == "__main__":
    main()

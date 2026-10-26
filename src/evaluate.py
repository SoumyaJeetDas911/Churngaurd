"""Model evaluation metrics and reproducible training plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Calculate classification metrics with defined behavior for edge cases."""
    predictions = (probabilities >= threshold).astype(int)
    has_two_classes = np.unique(y_true).size == 2
    roc_auc = float(roc_auc_score(y_true, probabilities)) if has_two_classes else None
    pr_auc = float(average_precision_score(y_true, probabilities)) if has_two_classes else None
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_evaluation_plots(
    y_validation: np.ndarray,
    validation_probabilities: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    threshold: float,
    feature_names: list[str],
    importance: np.ndarray,
    output_dir: Path,
) -> list[str]:
    """Save validation comparison curves and selected-model test plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    plt.figure(figsize=(7, 6))
    for name, probabilities in validation_probabilities.items():
        if np.unique(y_validation).size == 2:
            fpr, tpr, _ = roc_curve(y_validation, probabilities)
            score = roc_auc_score(y_validation, probabilities)
            plt.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("Validation ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()
    files.append("roc_curves.png")

    plt.figure(figsize=(7, 6))
    for name, probabilities in validation_probabilities.items():
        if np.unique(y_validation).size == 2:
            precision, recall, _ = precision_recall_curve(y_validation, probabilities)
            score = average_precision_score(y_validation, probabilities)
            plt.plot(recall, precision, label=f"{name} (AP={score:.3f})")
    plt.title("Validation Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=150)
    plt.close()
    files.append("precision_recall_curves.png")

    matrix = confusion_matrix(y_test, test_probabilities >= threshold, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Held-out Test Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
    )
    for (row, column), count in np.ndenumerate(matrix):
        axis.text(column, row, str(count), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    files.append("confusion_matrix.png")

    order = np.argsort(importance)[-20:]
    plt.figure(figsize=(10, max(5, len(order) * 0.3)))
    plt.barh(np.array(feature_names)[order], importance[order])
    plt.title("Selected Model Feature Importance")
    plt.xlabel("Importance (model-specific scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()
    files.append("feature_importance.png")
    return files


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
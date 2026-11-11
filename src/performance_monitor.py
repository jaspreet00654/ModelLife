
# ============================================================
# ModelLife — Performance Monitor
# Evaluates predictions once actual outcomes are available
# ============================================================

import csv
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. Calculate performance from prediction logs
# ============================================================

def calculate_performance(
    log_path="logs/predictions.csv"
):
    """
    Calculate classification metrics using logged predictions
    that have known actual outcomes.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Prediction log not found: {log_path}"
        )

    with open(log_path, mode="r", newline="") as file:
        rows = list(csv.DictReader(file))

    # Include only records with actual outcomes.
    completed_rows = [
        row for row in rows
        if row.get("actual", "").strip() in ("0", "1")
    ]

    if not completed_rows:
        return {
            "status": "insufficient_data",
            "message": "No completed outcomes available.",
            "completed_predictions": 0
        }

    y_true = [
        int(row["actual"])
        for row in completed_rows
    ]

    y_pred = [
        int(row["prediction"])
        for row in completed_rows
    ]

    auc = (
        roc_auc_score(y_true, y_pred)
        if len(set(y_true)) == 2
        else None
    )

    return {
        "status": "success",
        "completed_predictions": len(completed_rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": auc
    }


# ============================================================
# 2. Calculate performance for a specific model version
# ============================================================

def calculate_version_performance(
    model_version,
    log_path="logs/predictions.csv"
):
    """
    Calculate metrics for one model version only,
    using predictions with known actual outcomes.
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Prediction log not found: {log_path}"
        )

    with open(log_path, mode="r", newline="") as file:
        rows = list(csv.DictReader(file))

    # Select completed predictions for the requested version.
    version_rows = [
        row for row in rows
        if row.get("model_version") == model_version
        and row.get("actual", "").strip() in ("0", "1")
    ]

    if not version_rows:
        return {
            "status": "insufficient_data",
            "model_version": model_version,
            "completed_predictions": 0
        }

    y_true = [
        int(row["actual"]) for row in version_rows
    ]

    y_pred = [
        int(row["prediction"]) for row in version_rows
    ]

    auc = (
        roc_auc_score(y_true, y_pred)
        if len(set(y_true)) == 2
        else None
    )

    return {
        "status": "success",
        "model_version": model_version,
        "completed_predictions": len(version_rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": auc
    }


# ==========================================
# Detect Performance Degradation
# ==========================================

def check_performance_degradation(
    model_version,
    min_accuracy=0.70,
    min_predictions=30,
    log_path="logs/predictions.csv"
):
    """
    Check whether a model's logged accuracy
    falls below the configured threshold.
    """

    # Calculate performance for this model version
    metrics = calculate_version_performance(
        model_version=model_version,
        log_path=log_path
    )

    # Check if enough labelled predictions exist
    if metrics["status"] != "success":
        return {
            "status": "insufficient_data",
            "model_version": model_version,
            "degradation_detected": False,
            "reason": "No completed predictions available."
        }

    if metrics["completed_predictions"] < min_predictions:
        return {
            "status": "insufficient_data",
            "model_version": model_version,
            "degradation_detected": False,
            "reason": (
                f"Need at least {min_predictions} completed "
                "predictions to evaluate performance."
            )
        }

    # Compare measured accuracy with threshold
    accuracy = metrics["accuracy"]
    degraded = accuracy < min_accuracy

    return {
        "status": "success",
        "model_version": model_version,
        "accuracy": accuracy,
        "min_accuracy": min_accuracy,
        "completed_predictions": metrics["completed_predictions"],
        "degradation_detected": degraded,
        "recommendation": (
            "Review model and consider retraining."
            if degraded
            else "Performance is within the configured threshold."
        )
    }

# ============================================================
# ModelLife — Prediction Logger
# Records model predictions for future monitoring
# ============================================================

import csv
import os
from datetime import datetime


# ============================================================
# 1. Log a single prediction
# ============================================================

def log_prediction(
    customer_id,
    prediction,
    model_version,
    log_path="logs/predictions.csv"
):
    """
    Save a prediction with its timestamp and model version.
    """

    # Create the log directory if it does not exist.
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Define the columns stored in the CSV.
    fieldnames = [
        "timestamp",
        "customer_id",
        "prediction",
        "model_version",
        "actual"
    ]

    # Check whether the log file already exists.
    file_exists = os.path.exists(log_path)

    # Append the prediction record.
    with open(log_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write column headings for a new file.
        if not file_exists or os.path.getsize(log_path) == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "customer_id": customer_id,
            "prediction": prediction,
            "model_version": model_version,
            "actual": ""
        })

    print("Prediction logged successfully.")


# ============================================================
# 2. Update a prediction with its actual outcome
# ============================================================

def update_actual_outcome(
    customer_id,
    actual,
    log_path="logs/predictions.csv"
):
    """
    Update the most recent matching prediction without
    an actual outcome.
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError("Prediction log not found.")

    with open(log_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Find the latest matching record awaiting an outcome.
    for row in reversed(rows):
        if (
            row["customer_id"] == str(customer_id)
            and row["actual"] == ""
        ):
            row["actual"] = actual

            with open(log_path, mode="w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames
                )
                writer.writeheader()
                writer.writerows(rows)

            print("Actual outcome updated successfully.")
            return

    raise ValueError(
        f"No pending prediction found for customer {customer_id}."
    )
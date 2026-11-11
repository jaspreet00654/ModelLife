
# ============================================================
# Test version-specific performance monitoring
# ============================================================

import csv
import os
import tempfile

from src.performance_monitor import (
    calculate_version_performance
)


def test_version_performance():
    rows = [
        {
            "timestamp": "2026-09-28T10:00:00",
            "customer_id": "C001",
            "prediction": "1",
            "model_version": "v1",
            "actual": "1"
        },
        {
            "timestamp": "2026-09-28T10:01:00",
            "customer_id": "C002",
            "prediction": "0",
            "model_version": "v1",
            "actual": "0"
        },
        {
            "timestamp": "2026-09-28T10:02:00",
            "customer_id": "C003",
            "prediction": "1",
            "model_version": "v2",
            "actual": "0"
        },
        {
            "timestamp": "2026-09-28T10:03:00",
            "customer_id": "C004",
            "prediction": "0",
            "model_version": "v2",
            "actual": "0"
        }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(
            temp_dir, "predictions.csv"
        )

        with open(log_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        # Evaluate each version independently.
        v1_metrics = calculate_version_performance(
            "v1", log_path
        )
        v2_metrics = calculate_version_performance(
            "v2", log_path
        )

        assert v1_metrics["completed_predictions"] == 2
        assert v1_metrics["accuracy"] == 1.0

        assert v2_metrics["completed_predictions"] == 2
        assert v2_metrics["accuracy"] == 0.5

        print("Version-specific performance test passed.")
        print("v1 metrics:", v1_metrics)
        print("v2 metrics:", v2_metrics)


if __name__ == "__main__":
    test_version_performance()
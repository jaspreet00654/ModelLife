
# ============================================================
# Test ModelLife performance monitoring
# ============================================================

import csv
import os
import tempfile

from src.performance_monitor import calculate_performance


def test_performance_monitor():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(
            temp_dir, "predictions.csv"
        )

        # Create sample predictions with known outcomes.
        rows = [
            {
                "timestamp": "2026-09-28T10:00:00",
                "customer_id": "C001",
                "prediction": 1,
                "model_version": "v2",
                "actual": "1"
            },
            {
                "timestamp": "2026-09-28T10:01:00",
                "customer_id": "C002",
                "prediction": 0,
                "model_version": "v2",
                "actual": "0"
            },
            {
                "timestamp": "2026-09-28T10:02:00",
                "customer_id": "C003",
                "prediction": 1,
                "model_version": "v2",
                "actual": "0"
            },
            {
                "timestamp": "2026-09-28T10:03:00",
                "customer_id": "C004",
                "prediction": 0,
                "model_version": "v2",
                "actual": ""
            }
        ]

        # Write sample log.
        with open(log_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        # Calculate metrics using completed outcomes only.
        metrics = calculate_performance(log_path)

        assert metrics["status"] == "success"
        assert metrics["completed_predictions"] == 3
        assert abs(metrics["accuracy"] - 2 / 3) < 1e-9
        assert abs(metrics["precision"] - 0.5) < 1e-9
        assert abs(metrics["recall"] - 1.0) < 1e-9
        assert abs(metrics["f1"] - (2 / 3)) < 1e-9
        assert metrics["roc_auc"] is not None

        print("Performance monitor test passed.")
        print("Metrics:", metrics)


if __name__ == "__main__":
    test_performance_monitor()
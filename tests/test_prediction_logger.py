
# ============================================================
# Test prediction logging and outcome updates
# ============================================================

import csv
import os
import tempfile

from src.prediction_logger import (
    log_prediction,
    update_actual_outcome
)


def test_prediction_logger():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(
            temp_dir, "predictions.csv"
        )

        # Log a prediction.
        log_prediction(
            customer_id="C1001",
            prediction=1,
            model_version="v2",
            log_path=log_path
        )

        # Add its actual outcome.
        update_actual_outcome(
            customer_id="C1001",
            actual=1,
            log_path=log_path
        )

        # Verify the saved record.
        with open(log_path, mode="r", newline="") as file:
            rows = list(csv.DictReader(file))

        assert len(rows) == 1
        assert rows[0]["prediction"] == "1"
        assert rows[0]["actual"] == "1"
        assert rows[0]["model_version"] == "v2"

        print("Prediction logger test passed.")


if __name__ == "__main__":
    test_prediction_logger()
# ==========================================
# Test: Performance from Logged Predictions
# ==========================================

import tempfile
import pandas as pd

from src.prediction_logger import update_actual_outcome
from src.performance_monitor import calculate_version_performance


# 1. Create an isolated test log
with tempfile.TemporaryDirectory() as temp_dir:
    log_path = f"{temp_dir}/predictions.csv"

    # 2. Add sample predictions
    sample = pd.DataFrame([
        {
            "timestamp": "2026-09-28T10:00:00",
            "customer_id": "C001",
            "prediction": 1,
            "model_version": "v2",
            "actual": ""
        },
        {
            "timestamp": "2026-09-28T10:01:00",
            "customer_id": "C002",
            "prediction": 0,
            "model_version": "v2",
            "actual": ""
        },
        {
            "timestamp": "2026-09-28T10:02:00",
            "customer_id": "C003",
            "prediction": 1,
            "model_version": "v1",
            "actual": ""
        }
    ])

    sample.to_csv(log_path, index=False)

    # 3. Update actual outcomes for v2
    update_actual_outcome("C001", 1, log_path)
    update_actual_outcome("C002", 1, log_path)

    # 4. Calculate v2 performance
    metrics = calculate_version_performance(
        model_version="v2",
        log_path=log_path
    )

    # 5. Verify results
    assert metrics["status"] == "success"
    assert metrics["model_version"] == "v2"
    assert metrics["completed_predictions"] == 2
    assert metrics["accuracy"] == 0.5

    print("Logged performance test passed.")
    print("v2 metrics:", metrics)
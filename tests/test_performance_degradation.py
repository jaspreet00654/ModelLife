# ==========================================
# Test: Performance Degradation Detection
# ==========================================

import tempfile
import pandas as pd

from src.performance_monitor import (
    check_performance_degradation
)


with tempfile.TemporaryDirectory() as temp_dir:
    log_path = f"{temp_dir}/predictions.csv"

    # Create 30 labelled predictions:
    # 20 correct and 10 incorrect = 66.7% accuracy
    rows = []

    for i in range(30):
        actual = 1 if i < 15 else 0
        prediction = actual if i < 20 else 1 - actual

        rows.append({
            "timestamp": f"2026-09-28T10:{i:02d}:00",
            "customer_id": f"C{i:03d}",
            "prediction": prediction,
            "model_version": "v2",
            "actual": actual
        })

    pd.DataFrame(rows).to_csv(log_path, index=False)

    result = check_performance_degradation(
        model_version="v2",
        min_accuracy=0.70,
        min_predictions=30,
        log_path=log_path
    )

    assert result["status"] == "success"
    assert result["completed_predictions"] == 30
    assert result["accuracy"] < 0.70
    assert result["degradation_detected"] is True

    print("Performance degradation test passed.")
    print("Result:", result)
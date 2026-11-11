# ==========================================
# Test: Update Actual Customer Outcome
# ==========================================

import tempfile
import pandas as pd

from src.prediction_logger import update_actual_outcome


# 1. Create a temporary prediction log
with tempfile.TemporaryDirectory() as temp_dir:
    log_path = f"{temp_dir}/predictions.csv"

    # 2. Add a sample pending prediction
    sample = pd.DataFrame([{
        "timestamp": "2026-09-28T21:52:25",
        "customer_id": "DEMO001",
        "prediction": 1,
        "model_version": "v2",
        "actual": ""
    }])

    sample.to_csv(log_path, index=False)

    # 3. Record the actual outcome
    update_actual_outcome(
        customer_id="DEMO001",
        actual=1,
        log_path=log_path
    )

    # 4. Verify the outcome was saved
    result = pd.read_csv(log_path)

    assert result.loc[0, "actual"] == 1

    print("Actual outcome update test passed.")
    print(result.to_string(index=False))
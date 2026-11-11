
# ============================================================
# Test prediction service with a trained model
# ============================================================

import os
import tempfile

from src.data_loader import load_and_split_data
from src.preprocessing import build_preprocessor
from src.train import train_model
from src.prediction_service import predict_and_log


def test_prediction_service():
    # Load and prepare data.
    data = load_and_split_data(
        "data/Telco-Customer-Churn.csv"
    )

    x_train = data["x_model_train"]
    y_train = data["y_model_train"]
    x_test = data["x_test"]

    # Train a pipeline using the project's preprocessor.
    preprocessor = build_preprocessor(x_train)
    model = train_model(preprocessor, x_train, y_train)

    # Use one raw customer record for prediction.
    customer = x_test.iloc[[0]]

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(
            temp_dir, "predictions.csv"
        )

        prediction = predict_and_log(
            model=model,
            input_data=customer,
            customer_id="TEST001",
            model_version="test_v1",
            log_path=log_path
        )

        # Verify the prediction and log entry.
        assert prediction in (0, 1)
        assert os.path.exists(log_path)

        with open(log_path, "r", newline="") as file:
            import csv
            rows = list(csv.DictReader(file))

        assert len(rows) == 1
        assert rows[0]["customer_id"] == "TEST001"
        assert rows[0]["model_version"] == "test_v1"
        assert int(rows[0]["prediction"]) == prediction
        assert rows[0]["actual"] == ""

        print("Prediction service test passed.")
        print("Predicted churn:", prediction)


if __name__ == "__main__":
    test_prediction_service()
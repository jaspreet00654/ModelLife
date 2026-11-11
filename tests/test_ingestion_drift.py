
import tempfile
from pathlib import Path

from src.data_loader import load_and_split_data
from src.data_ingestion import load_production_batch
from src.drift_detection import detect_data_drift


def test_ingestion_with_drift_detection():
    # Load reference and simulated production data
    datasets = load_and_split_data()
    x_model_train = datasets["x_model_train"]
    x_production = datasets["x_production"]

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "drifted_production.csv"

        # Simulate a change in incoming production data
        drifted_data = x_production.copy()
        drifted_data["MonthlyCharges"] += 50

        # Save the drifted batch as an external CSV
        drifted_data.to_csv(csv_path, index=False)

        # Ingest and validate the external CSV
        incoming_data = load_production_batch(
            file_path=csv_path,
            required_columns=list(x_model_train.columns)
        )

        # Detect drift against the reference dataset
        drift_report = detect_data_drift(
            reference_data=x_model_train,
            new_data=incoming_data,
            columns=["tenure", "MonthlyCharges", "TotalCharges"]
        )

        # Verify drift detection
        assert drift_report["drift_results"]["MonthlyCharges"][
            "drift_detected"
        ] is True

        assert drift_report["retraining_required"] is True

        print("Ingestion and drift detection test passed.")
        print("Drift report:", drift_report)


if __name__ == "__main__":
    test_ingestion_with_drift_detection()
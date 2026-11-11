
import tempfile
from pathlib import Path

import pandas as pd

from src.data_ingestion import load_production_batch


def test_valid_production_batch():
    # Create a temporary CSV for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "production.csv"

        expected_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

        test_data = pd.DataFrame({
            "TotalCharges": [100.0, 200.0],
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0]
        })

        test_data.to_csv(file_path, index=False)

        # Load and validate the batch
        result = load_production_batch(
            file_path,
            expected_columns
        )

        assert list(result.columns) == expected_columns
        assert result.shape == (2, 3)

        print("Valid production batch test passed.")


def test_missing_column():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "invalid.csv"

        test_data = pd.DataFrame({
            "tenure": [12],
            "MonthlyCharges": [50.0]
        })

        test_data.to_csv(file_path, index=False)

        try:
            load_production_batch(
                file_path,
                ["tenure", "MonthlyCharges", "TotalCharges"]
            )
        except ValueError as error:
            assert "TotalCharges" in str(error)
            print("Missing column test passed.")
        else:
            raise AssertionError("Expected a schema validation error.")


if __name__ == "__main__":
    test_valid_production_batch()
    test_missing_column()
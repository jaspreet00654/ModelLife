
import tempfile
from pathlib import Path

from src.data_loader import load_and_split_data
from src.data_ingestion import load_production_batch


def test_external_production_csv():
    # Load datasets returned as a dictionary
    datasets = load_and_split_data()

    x_model_train = datasets["x_model_train"]
    x_production = datasets["x_production"]

    # Create a temporary CSV for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "production_batch.csv"

        # Save production features to CSV
        x_production.to_csv(csv_path, index=False)

        # Load and validate the external production CSV
        loaded_data = load_production_batch(
            file_path=csv_path,
            required_columns=list(x_model_train.columns)
        )

        # Verify the loaded data
        assert loaded_data.shape == x_production.shape
        assert list(loaded_data.columns) == list(x_model_train.columns)

        print("External production CSV test passed.")
        print(f"Loaded batch shape: {loaded_data.shape}")


if __name__ == "__main__":
    test_external_production_csv()
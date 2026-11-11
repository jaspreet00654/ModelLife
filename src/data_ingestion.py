
import pandas as pd
from pathlib import Path


def load_production_batch(file_path, required_columns):
    """
    Load a production CSV batch and validate its feature columns.

    Args:
        file_path: Path to the production CSV file.
        required_columns: Expected feature column names.

    Returns:
        A DataFrame with columns in the expected order.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty or its schema is invalid.
    """
    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.is_file():
        raise FileNotFoundError(f"Production file not found: {file_path}")

    # Load the production data
    data = pd.read_csv(file_path)

    # Check for empty data
    if data.empty:
        raise ValueError("Production batch is empty.")

    # Check for duplicate column names
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names found in production data.")

    # Check whether expected columns are present
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    # Check for unexpected columns
    extra_columns = [
        column for column in data.columns
        if column not in required_columns
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            f"Invalid production schema. "
            f"Missing columns: {missing_columns}. "
            f"Unexpected columns: {extra_columns}."
        )

    # Arrange columns in the expected model-input order
    data = data[required_columns]

    print("Production batch loaded and validated successfully.")
    print(f"Batch shape: {data.shape}")

    return data
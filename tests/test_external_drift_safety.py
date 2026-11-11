
import tempfile
from pathlib import Path

import pandas as pd

from src.lifecycle import run_lifecycle


def test_external_drift_without_labels():
    # Create reference and drifted production data
    reference_data = pd.DataFrame({
        "tenure": [10, 20, 30, 40, 50] * 20,
        "MonthlyCharges": [20, 30, 40, 50, 60] * 20,
        "TotalCharges": [200, 400, 600, 800, 1000] * 20
    })

    new_data = reference_data.copy()
    new_data["MonthlyCharges"] += 100

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Use isolated registry and model artifact paths
        metadata_path = temp_dir / "test_metadata.json"
        candidate_path = temp_dir / "candidate.pkl"
        production_path = temp_dir / "current_model.pkl"

        # Run lifecycle without actual labels
        result = run_lifecycle(
            current_model=None,
            preprocessor=None,
            reference_data=reference_data,
            new_data=new_data,
            new_labels=None,
            test_data=None,
            test_labels=None,
            current_metrics=None,
            candidate_version="v_test",
            metadata_path=str(metadata_path),
            candidate_path=str(candidate_path),
            production_path=str(production_path)
        )

        # Verify drift was detected and retraining was blocked
        assert result["status"] == "drift_detected_labels_unavailable"
        assert result["drift"]["retraining_required"] is True

        # Verify no model artifacts or registry were created
        assert not candidate_path.exists()
        assert not production_path.exists()
        assert not metadata_path.exists()

        print("External drift safety test passed.")
        print("Status:", result["status"])
        print("Retraining blocked: actual labels unavailable.")
        print("No model or registry artifacts created.")


if __name__ == "__main__":
    test_external_drift_without_labels()
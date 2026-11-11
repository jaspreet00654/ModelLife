
import os
import tempfile

from src.data_loader import load_and_split_data
from src.preprocessing import build_preprocessor
from src.train import train_model, evaluate_model
from src.lifecycle import run_lifecycle


def test_lifecycle_no_drift():
    """Verify that no drift skips retraining and promotion."""

    data = load_and_split_data()
    preprocessor = build_preprocessor(data["x_model_train"])

    current_model = train_model(
        preprocessor,
        data["x_model_train"],
        data["y_model_train"]
    )

    current_metrics = evaluate_model(
        current_model,
        data["x_test"],
        data["y_test"]
    )

    # Simulate a distribution shift on a copy
    new_data = data["x_production"].copy()
    new_data["MonthlyCharges"] += 50

    # Use an isolated temporary registry path.
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = os.path.join(temp_dir, "test_registry.json")
        candidate_path = os.path.join(temp_dir, "candidate.pkl")

        result = run_lifecycle(
            current_model=current_model,
            preprocessor=preprocessor,
            reference_data=data["x_model_train"],
            new_data=new_data,
            new_labels=data["y_production"],
            test_data=data["x_test"],
            test_labels=data["y_test"],
            current_metrics=current_metrics,
            candidate_version="test_v1",
            metadata_path=metadata_path,
            candidate_path=candidate_path
        )

        assert result["drift"]["retraining_required"]
        assert result["status"] == "rejected"
        assert not result["validation"]["approved"]
        assert (
            result["validation"]["f1_improvement"] < 0
            or result["validation"]["auc_improvement"] < 0
        )
        assert not os.path.exists(candidate_path)
        assert not os.path.exists(metadata_path)

        print("Drift-triggered rejection test passed.")


if __name__ == "__main__":
    test_lifecycle_no_drift()

# ============================================================
# Test: Approved candidate promotion using temporary paths
# ============================================================

import tempfile
import os
import joblib

from unittest.mock import patch
from src.lifecycle import run_lifecycle
from src.model_registry import save_model, save_metadata


def test_approved_candidate_promotion():
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Create isolated paths for this test
        metadata_path = os.path.join(temp_dir, "registry.json")
        candidate_path = os.path.join(temp_dir, "candidate.pkl")
        production_path = os.path.join(temp_dir, "current_model.pkl")

        # 2. Prepare an existing current model entry
        current_model_path = os.path.join(temp_dir, "old_model.pkl")
        save_model("old-model-placeholder", current_model_path)

        save_metadata(
            {
                "model_version": "v1",
                "artifact_path": current_model_path,
                "status": "current",
                "metrics": {}
            },
            metadata_path
        )

        # 3. Load data and build the preprocessing pipeline
        from src.data_loader import load_and_split_data
        from src.preprocessing import build_preprocessor
        from src.train import train_model, evaluate_model

        data = load_and_split_data()
        preprocessor = build_preprocessor(data["x_model_train"])

        current_model = train_model(
            preprocessor,
            data["x_model_train"],
            data["y_model_train"]
        )
        current_metrics = evaluate_model(
            current_model,
            data["x_test"],
            data["y_test"]
        )

        # 4. Simulate drift in the new batch
        new_data = data["x_production"].copy()
        new_data["MonthlyCharges"] += 50

        # 5. Force approval for testing the promotion branch
        with patch(
            "src.lifecycle.request_approval",
            return_value=True
        ):
            result = run_lifecycle(
                current_model=current_model,
                preprocessor=preprocessor,
                reference_data=data["x_model_train"],
                new_data=new_data,
                new_labels=data["y_production"],
                test_data=data["x_test"],
                test_labels=data["y_test"],
                current_metrics=current_metrics,
                candidate_version="v_test",
                metadata_path=metadata_path,
                candidate_path=candidate_path,
                production_path=production_path
            )

        # 6. Verify successful promotion and isolated artifact
        assert result["status"] == "promoted"
        assert os.path.exists(candidate_path)
        assert os.path.exists(production_path)

        with open(metadata_path, "r") as file:
            registry = __import__("json").load(file)

        promoted = next(
            model for model in registry["models"]
            if model["model_version"] == "v_test"
        )
        assert promoted["status"] == "current"
        assert promoted["production_artifact"] == production_path

        print("Approved candidate promotion test passed.")


if __name__ == "__main__":
    test_approved_candidate_promotion()
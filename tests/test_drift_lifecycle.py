
# ============================================================
# ModelLife — Isolated Drift Lifecycle Test
# Tests drift detection and candidate validation safely
# ============================================================

import os
import tempfile
import shutil

from src.data_loader import load_and_split_data
from src.preprocessing import build_preprocessor
from src.train import train_model, evaluate_model
from src.lifecycle import run_lifecycle


def test_drift_lifecycle():
    # --------------------------------------------------------
    # 1. Load data and train a baseline model
    # --------------------------------------------------------
    data = load_and_split_data("data/Telco-Customer-Churn.csv")

    x_model_train = data["x_model_train"]
    y_model_train = data["y_model_train"]
    x_production = data["x_production"]
    y_production = data["y_production"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    preprocessor = build_preprocessor(x_model_train)

    current_model = train_model(
        preprocessor,
        x_model_train,
        y_model_train
    )

    current_metrics = evaluate_model(
        current_model,
        x_test,
        y_test
    )

    # --------------------------------------------------------
    # 2. Simulate drift on a copy of production data
    # --------------------------------------------------------
    new_data = x_production.copy()
    new_data["MonthlyCharges"] += 50

    # --------------------------------------------------------
    # 3. Use temporary paths to protect real model files
    # --------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = os.path.join(
            temp_dir, "test_metadata.json"
        )
        candidate_path = os.path.join(
            temp_dir, "test_candidate.pkl"
        )
        production_path = os.path.join(
            temp_dir, "test_current_model.pkl"
        )

        # ----------------------------------------------------
        # 4. Run the lifecycle
        # ----------------------------------------------------
        result = run_lifecycle(
            current_model=current_model,
            preprocessor=preprocessor,
            reference_data=x_model_train,
            new_data=new_data,
            new_labels=y_production,
            test_data=x_test,
            test_labels=y_test,
            current_metrics=current_metrics,
            candidate_version="v_test",
            metadata_path=metadata_path,
            candidate_path=candidate_path,
            production_path=production_path
        )

        # ----------------------------------------------------
        # 5. Verify the lifecycle outcome
        # ----------------------------------------------------
        assert result["status"] in (
            "rejected",
            "promoted"
        ), f"Unexpected lifecycle result: {result}"

        if result["status"] == "rejected":
            print("Drift-triggered rejection test passed.")

        elif result["status"] == "promoted":
            assert os.path.exists(production_path)
            print("Drift-triggered promotion test passed.")

        print("Lifecycle test result:", result["status"])


if __name__ == "__main__":
    test_drift_lifecycle()
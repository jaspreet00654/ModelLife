
import json
import os

from src.drift_detection import detect_data_drift
from src.retraining import train_candidate_model
from src.train import evaluate_model
from src.model_evaluation import compare_models, validate_candidate
from src.approval import request_approval
from src.model_registry import (
    save_model,
    save_metadata,
    get_current_model,
    promote_model
)


def run_lifecycle(
    current_model,
    preprocessor,
    reference_data,
    new_data,
    new_labels,
    test_data,
    test_labels,
    current_metrics,
    candidate_version,
    metadata_path="models/model_metadata.json",
    candidate_path=None,
    production_path="models/current_model.pkl"
):
    """Run drift detection, candidate validation and approval."""

    # 1. Detect data drift
    drift_report = detect_data_drift(reference_data, new_data)

    print("Drift report:", drift_report)

    if not drift_report["retraining_required"]:
        print("No drift detected. Retraining is not required.")
        return {"status": "no_drift", "drift": drift_report}

    if new_labels is None:
        return {
            "status": "drift_detected_labels_unavailable",
            "message": (
                "Data drift detected, but actual labels are unavailable. "
                "Supervised retraining cannot proceed."
            ),
            "drift": drift_report
        }

    if candidate_path is None:
        candidate_path = f"models/model_{candidate_version}.pkl"

   
    
# Initialize an empty registry for a new metadata file
    registry = {"models": []}

    # Load existing registry if the file exists
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as file:
         registry = json.load(file)

    # Get registered models safely
    existing_models = registry.get("models", [])

        # Prevent reusing the currently active production version
    active_model = next(
            (
                model for model in existing_models
                if model.get("status") == "current"
            ),
            None
        )

    if (
            active_model
            and active_model.get("model_version") == candidate_version
        ):
            print("Candidate version is already active.")
            return {
                "status": "active_version_conflict",
                "version": candidate_version
            }

    # Prevent reusing any version already registered
    if any(
        model.get("model_version") == candidate_version
        for model in existing_models
    ):
        print("Candidate version already exists in registry.")
        return {
            "status": "version_conflict",
            "version": candidate_version,
            "candidate_path": candidate_path
        }

    if os.path.exists(candidate_path):
        return {
            "status": "artifact_exists",
            "version": candidate_version,
            "candidate_path": candidate_path
        }

    # Re-evaluate the current model on the untouched held-out test data.
    evaluated_current_metrics = evaluate_model(
        current_model,
        test_data,
        test_labels
    )

    if current_metrics != evaluated_current_metrics:
        print("Provided current metrics differ from held-out evaluation; using evaluated metrics.")

    current_metrics = evaluated_current_metrics

    # 2. Train and evaluate candidate
    candidate_model = train_candidate_model(
        preprocessor,
        new_data,
        new_labels
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        test_data,
        test_labels
    )

    # 3. Compare and validate
    comparison = compare_models(
        current_metrics,
        candidate_metrics
    )

    validation = validate_candidate(comparison)

    print("Candidate metrics:", candidate_metrics)
    print("Validation:", validation)

    # 4. Ask for human approval
    approved = request_approval(
        candidate_version,
        validation
    )

    if not approved:
        return {
            "status": "rejected",
            "drift": drift_report,
            "comparison": comparison,
            "validation": validation
        }

    # 5. Save and promote approved candidate
    save_model(candidate_model, candidate_path)

    candidate_metadata = {
    "model_version": candidate_version,
    "artifact_path": candidate_path,
    "status": "candidate",
    "metrics": candidate_metrics
}

    save_metadata(candidate_metadata, metadata_path)

    promoted = promote_model(
        metadata_path,
        candidate_version,
        current_path=production_path
    )
    if not promoted:
        return {
            "status": "promotion_failed",
            "version": candidate_version,
            "candidate_status": "candidate",
            "candidate_path": candidate_path,
            "metadata_path": metadata_path
        }

    return {
        "status": "promoted",
        "version": candidate_version,
        "metrics": candidate_metrics,
        "comparison": comparison
    }
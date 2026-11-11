import os
import json
import shutil
import joblib


def save_model(model, model_path):
    """Save a model artifact to disk."""
    model_directory = os.path.dirname(model_path)
    if model_directory:
        os.makedirs(model_directory, exist_ok=True)
    joblib.dump(model, model_path)
    return True


def save_metadata(model_metadata, metadata_path):
    """Add or update model metadata in the registry."""
    registry = {"models": []}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as file:
            registry = json.load(file)

    registry.setdefault("models", [])

    model_version = model_metadata.get("model_version")

    # Update an existing version instead of duplicating it
    existing_index = next(
        (
            index for index, model in enumerate(registry["models"])
            if model.get("model_version") == model_version
        ),
        None
    )

    if existing_index is not None:
        registry["models"][existing_index].update(model_metadata)
    else:
        registry["models"].append(model_metadata)

    metadata_directory = os.path.dirname(metadata_path)
    if metadata_directory:
        os.makedirs(metadata_directory, exist_ok=True)

    with open(metadata_path, "w") as file:
        json.dump(registry, file, indent=4)

    return True

def get_current_model(metadata_path):
    """Return the registry entry marked as the current model."""
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r") as file:
        registry = json.load(file)

    return next(
        (
            model for model in registry.get("models", [])
            if model.get("status") == "current"
        ),
        None,
    )

# ============================================================ 
# Promote Model 
# Copy approved model artifact to current production path 
# ============================================================ 
 
def promote_model(
    metadata_path,
    candidate_version,
    current_path="models/current_model.pkl"
):
    """Promote a registered candidate model to production.""" 
 
    if not os.path.exists(metadata_path): 
        print("Model registry not found.") 
        return False 
 
    with open(metadata_path, "r") as file: 
        registry = json.load(file) 
 
    models = registry.get("models", []) 
 
    candidate = next( 
        ( 
            model for model in models 
            if model.get("model_version") == candidate_version 
        ), 
        None 
    ) 
 
    if candidate is None: 
        print(f"Model {candidate_version} not found.") 
        return False 
 
    artifact_path = candidate.get("artifact_path") 
 
    if not artifact_path or not os.path.exists(artifact_path): 
        print("Candidate model artifact not found.") 
        return False 
 
    
 
    # Copy candidate artifact to production path 
        # Create the production directory if it does not exist
    production_directory = os.path.dirname(current_path)
    if production_directory:
        os.makedirs(production_directory, exist_ok=True)

    # Copy candidate artifact to production path
    shutil.copy2(artifact_path, current_path)
    
 
    # Update registry statuses 
    for model in models: 
        if model.get("status") == "current": 
            model["status"] = "archived" 
 
    candidate["status"] = "current" 
    candidate["production_artifact"] = current_path 
 
    with open(metadata_path, "w") as file: 
        json.dump(registry, file, indent=4) 
 
    print(f"Model {candidate_version} promoted to production.") 
    print(f"Active artifact: {current_path}") 
 
    return True 
 
 
# ============================================================ 
# Rollback Model 
# Restore a previous model artifact to production 
# ============================================================ 
 
def rollback_model(metadata_path, version): 
    """Restore a previous model version as production.""" 

    if not os.path.exists(metadata_path): 
        print("Model registry not found.") 
        return False 
 
    with open(metadata_path, "r") as file: 
        registry = json.load(file) 
 
    models = registry.get("models", []) 
 
    target = next( 
        ( 
            model for model in models 
            if model.get("model_version") == version 
        ), 
        None 
    ) 
 
    if target is None: 
        print(f"Model version {version} not found.") 
        return False 
 
    artifact_path = target.get("artifact_path")
 
    if not artifact_path or not os.path.exists(artifact_path):
        print(f"Artifact for {version} not found.") 
        return False 
 
    current_path = "models/current_model.pkl" 
 
    # Restore selected model artifact 
    shutil.copy2(artifact_path, current_path) 
 
    # Update registry statuses 
    for model in models: 
        if model.get("status") == "current": 
            model["status"] = "archived" 
 
    target["status"] = "current" 
    target["production_artifact"] = current_path 
 
    with open(metadata_path, "w") as file: 
        json.dump(registry, file, indent=4) 
 
    print(f"Rollback completed. Model {version} is now current.") 
    print(f"Active artifact: {current_path}") 
 
    return True 

# ============================================================
# ModelLife — Prediction Service
# Generates predictions and logs them automatically
# ============================================================

from src.prediction_logger import log_prediction


# ============================================================
# 1. Predict and log
# ============================================================

def predict_and_log(
    model,
    input_data,
    customer_id,
    model_version,
    log_path="logs/predictions.csv"
):
    """
    Generate a prediction using the supplied model,
    then record it with its model version.
    """

    # Generate prediction using the complete ML pipeline.
    prediction = int(model.predict(input_data)[0])

    # Log the prediction for future monitoring.
    log_prediction(
        customer_id=customer_id,
        prediction=prediction,
        model_version=model_version,
        log_path=log_path
    )

    return prediction
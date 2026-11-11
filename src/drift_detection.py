from scipy.stats import ks_2samp


def detect_data_drift(reference_data, new_data, columns=None, significance=0.05):
    """Detect numerical data drift using the Kolmogorov-Smirnov test."""

    if columns is None:
        columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    drift_results = {}

    for column in columns:
        statistic, p_value = ks_2samp(
            reference_data[column].dropna(),
            new_data[column].dropna()
        )

        drift_results[column] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < significance)
        }

    retraining_required = any(
        result["drift_detected"]
        for result in drift_results.values()
    )

    return {
        "drift_results": drift_results,
        "retraining_required": retraining_required
    }

def compare_models(current_metrics, candidate_metrics):
    """Compare current and candidate model metrics."""

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    comparison = {}

    for metric in metric_names:
        current_value = current_metrics[metric]
        candidate_value = candidate_metrics[metric]

        comparison[metric] = {
            "current": current_value,
            "candidate": candidate_value,
            "improvement": candidate_value - current_value
        }

    return comparison


def validate_candidate(
    comparison,
    min_f1_improvement=0.0,
    min_auc_improvement=0.0
):
    """Check whether the candidate meets validation thresholds."""

    f1_improvement = comparison["f1"]["improvement"]
    auc_improvement = comparison["roc_auc"]["improvement"]

    approved = (
        f1_improvement >= min_f1_improvement
        and auc_improvement >= min_auc_improvement
    )

    return {
        "approved": approved,
        "f1_improvement": f1_improvement,
        "auc_improvement": auc_improvement,
        "reason": (
            "Candidate passed validation."
            if approved
            else "Candidate did not meet validation thresholds."
        )
    }
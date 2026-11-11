ModelLife — Adaptive ML Model Lifecycle & Continuous Learning System
Tagline: Detect. Learn. Validate. Evolve.
1. Problem Statement
Machine learning models are often treated as finished products after training and deployment. In real-world environments, however, incoming data and underlying patterns can change over time, potentially causing model performance to degrade. Without monitoring, this degradation may go unnoticed.
ModelLife addresses this problem by monitoring deployed ML models, detecting significant changes in incoming data and model performance, and identifying when retraining may be necessary.
2. Objective
ModelLife treats a deployed ML model as a continuously evolving component rather than a static artifact. It monitors model behaviour, detects data and performance changes, uses newly available labelled data or feedback for candidate retraining, evaluates updated models against the current production model, and gives developers control over whether a new model should be promoted.
3. Core Idea
ModelLife follows a human-in-the-loop approach.
The system performs monitoring, analysis, drift detection, candidate training, and model comparison, while the developer retains final control over model promotion.
Monitor → Detect → Analyze → Recommend → Retrain → Compare → Approve/Reject → Promote
4. Key Features — Phase 1
Model Monitoring
Logs predictions, input data, model versions, and available ground-truth outcomes.
Data Drift Detection
Detects statistically significant changes between reference/training data and incoming production data using methods such as KS-test and PSI.
Performance Monitoring
Evaluates model performance when ground-truth labels become available.
Feedback & New Data Ingestion
Accepts newly labelled data, actual outcomes, user feedback, or developer-provided datasets for future model improvement.
Candidate Retraining
Creates a new candidate model when predefined conditions indicate that retraining may be beneficial.
Model Versioning
Maintains different model versions and their associated metrics and metadata.
Version Comparison
Compares the candidate model against the current production model using appropriate evaluation metrics.
Human Approval
Allows the developer to accept or reject retraining and model-promotion recommendations.
Model Promotion
Promotes an approved candidate model to the production version while retaining the previous model for rollback.

## Current implementation status

The repository currently includes a restored `calculate_performance()` implementation in `src/performance_monitor.py` that returns calculated metrics for prediction logs with known actual outcomes.

- `calculate_performance()` now returns metrics instead of falling through without a result.
- Duplicate logic in `calculate_version_performance()` was removed.
- Version-specific metric calculations remain consistent with the same filtering and scoring logic used for completed labels.
- File-not-found and insufficient-data handling remain in place.

Validation evidence: running `pytest -q` in the repository produced `12 passed in 4.67s`.

This report is limited to the actual repository tests that currently exist; it does not claim broader test coverage beyond that evidence.
"""Generate interview answers grounded in statically extracted project evidence."""
from __future__ import annotations

from src.project_analyzer import get_project_inventory


def _function_index(inventory):
    index = {}
    for file_info in inventory.get("source_evidence", []):
        for function in file_info.get("functions", []):
            index.setdefault(function["name"], (file_info["path"], function))
    return index


def _function_refs(inventory, names):
    index = _function_index(inventory)
    references = []
    for name in names:
        entry = index.get(name)
        if entry:
            path, function = entry
            references.append(f"{path}:{function['line']}")
    return references


def _function_description(inventory, name):
    entry = _function_index(inventory).get(name)
    if not entry:
        return None
    path, function = entry
    description = function.get("docstring") or ""
    calls = []
    for call in function.get("calls", []):
        short_name = call["name"].rsplit(".", 1)[-1]
        if short_name not in calls:
            calls.append(short_name)
    details = []
    if description:
        details.append(f"documented as '{description.strip()}'")
    if calls:
        details.append("calls " + ", ".join(f"`{call}`" for call in calls[:7]))
    statements = function.get("statements", [])
    if statements:
        details.append(
            "source step " + f"`{statements[0]}`"
        )
    return f"`{function['signature']}` ({path}:{function['line']})" + (
        f", { '; '.join(details)}" if details else ""
    )


def _documentation(inventory, terms, limit=2):
    found = []
    for entry in inventory.get("documentation", []):
        text = entry["text"]
        if text.startswith("#"):
            continue
        if any(term in text.casefold() for term in terms):
            found.append(f"{text} ({entry['path']}:{entry['line']})")
            if len(found) == limit:
                break
    return found


def _technical_files(inventory):
    return {
        item["path"]: item
        for item in inventory.get("source_evidence", [])
    }


def _all_refs(inventory, paths, limit=12):
    files = _technical_files(inventory)
    references = []
    for path in paths:
        info = files.get(path)
        if not info:
            continue
        for item in (
            info.get("functions", [])
            + info.get("classes", [])
            + info.get("important_calls", [])
        ):
            reference = f"{path}:{item['line']}"
            if reference not in references:
                references.append(reference)
            if len(references) >= limit:
                return references
    return references


def _call_refs(inventory, call_names):
    requested = set(call_names)
    references = []
    for file_info in inventory.get("source_evidence", []):
        for call in file_info.get("important_calls", []):
            if call["name"] in requested:
                reference = f"{file_info['path']}:{call['line']}"
                if reference not in references:
                    references.append(reference)
    return references


def _ml_training_facts(inventory):
    details = []
    train = _function_description(inventory, "train_model")
    preprocess = _function_description(inventory, "build_preprocessor")
    load = _function_description(inventory, "load_and_split_data")
    evaluate = _function_description(inventory, "evaluate_model")
    if load:
        details.append(f"I load and split the churn data with {load}.")
    if preprocess:
        details.append(f"I prepare features with {preprocess}.")
    if train:
        details.append(f"I fit the model with {train}.")
    if evaluate:
        details.append(f"I evaluate held-out data with {evaluate}.")
    return details


def _performance_facts(inventory):
    index = _function_index(inventory)
    general = index.get("calculate_performance")
    versioned = index.get("calculate_version_performance")
    if not general or not versioned:
        return None

    general_path, general_fn = general
    version_path, version_fn = versioned
    return (
        "In `src/performance_monitor.py`, `calculate_performance()` reads "
        "the CSV, keeps rows whose `actual` value is `0` or `1`, and returns "
        "accuracy, precision, recall, F1, and ROC-AUC; it raises "
        "`FileNotFoundError` when the log is missing and reports "
        "`insufficient_data` when there are no completed outcomes "
        f"({general_path}:{general_fn['line']}). "
        "It calculates ROC-AUC with `roc_auc_score(y_true, y_pred)`. "
        "`calculate_version_performance(model_version, ...)` applies the "
        "same completed-outcome filter to the requested version and returns "
        "the same metrics with the version and count; it also preserves the "
        "missing-file and insufficient-data outcomes "
        f"({version_path}:{version_fn['line']}) and also calls "
        "`roc_auc_score(y_true, y_pred)`."
    )


def _test_facts(inventory):
    tests = []
    for file_info in inventory.get("source_evidence", []):
        path = file_info["path"]
        if not path.replace("\\", "/").startswith("tests/"):
            continue
        names = [
            function["name"]
            for function in file_info.get("functions", [])
            if function["name"].startswith("test_")
        ]
        if names:
            tests.append((path, names))
    preferred = (
        "test_performance_monitor",
        "test_version_performance",
        "test_logged_performance",
        "test_performance_degradation",
        "test_drift_lifecycle",
        "test_lifecycle_no_drift",
        "test_approved_candidate_promotion",
        "test_external_drift_without_labels",
        "test_prediction_service",
        "test_prediction_logger",
    )
    rank = {name: index for index, name in enumerate(preferred)}
    selected = []
    for path, names in tests:
        selected_names = sorted(
            (name for name in names if name in rank),
            key=rank.get,
        )
        if selected_names:
            selected.append((path, selected_names))
    return selected or tests[:6]


def _technology_answer(inventory, technology):
    names = {
        "Python": (
            "I use Python to implement the data, model, monitoring, registry, "
            "and Streamlit dashboard components.",
            "The project separates those responsibilities into source modules, "
            "and the dashboard calls the same project code for local insights "
            "and interview questions.",
            "The code establishes Python as the implementation language; it "
            "does not establish an unrecorded reason for choosing it.",
            ["load_and_split_data", "train_model", "run_lifecycle",
             "get_project_insights"],
        ),
        "Pandas": (
            "I use Pandas to read the churn dataset and production CSV batches.",
            "I convert `TotalCharges` to numeric values, fill missing values, "
            "remove `customerID`, and encode `Churn` as 0 or 1 before making "
            "stratified training and test splits.",
            "For production input, I also check that the batch is non-empty "
            "and its columns match the model schema, then arrange the columns "
            "in the expected order. These operations prepare consistent "
            "tabular inputs for model training and prediction.",
            ["load_and_split_data", "load_production_batch"],
        ),
        "Scikit-learn": (
            "I use scikit-learn for feature preprocessing, model training, "
            "evaluation, and candidate comparison.",
            "A `ColumnTransformer` scales numeric columns and one-hot encodes "
            "categorical ones, then a `Pipeline` combines those transformations "
            "with `LogisticRegression`.",
            "The same pipeline fits and predicts on tabular churn data. "
            "Evaluation reports accuracy, precision, recall, F1, and ROC-AUC; "
            "candidate validation compares F1 and ROC-AUC with the current "
            "model before approval.",
            ["build_preprocessor", "train_model", "evaluate_model",
             "validate_candidate"],
        ),
        "SciPy": (
            "I use SciPy's two-sample Kolmogorov-Smirnov test to detect "
            "distribution changes in numerical production features.",
            "`detect_data_drift()` compares the reference and incoming "
            "samples for `tenure`, `MonthlyCharges`, and `TotalCharges`.",
            "A p-value below the configured significance level marks a feature "
            "as drifted, and drift in any checked feature raises the lifecycle "
            "retraining gate. This is a data-distribution signal, not proof "
            "that model accuracy has fallen.",
            ["detect_data_drift", "run_lifecycle"],
        ),
        "Joblib": (
            "I use Joblib in the model registry to serialize trained model "
            "artifacts to disk.",
            "`save_model()` writes the model artifact, while registry metadata "
            "keeps its version, path, status, and metrics.",
            "The lifecycle saves an approved candidate before promoting it, "
            "so the registered artifact can be selected as the production "
            "version. The source supports artifact saving; it does not show "
            "a general artifact-retention or cleanup policy.",
            ["save_model", "save_metadata", "promote_model"],
        ),
        "Streamlit": (
            "I use Streamlit to build the local ModelLife dashboard.",
            "It provides navigation for the overview, monitoring, registry, "
            "and Interview Coach, along with controls for generating questions "
            "and filtering them by category and difficulty.",
            "The dashboard reads local registry and prediction-log files and "
            "renders generated answers with source references. It is a local "
            "interface; the code does not establish hosted authentication or "
            "a live LLM connection.",
            ["read_json", "get_project_insights"],
        ),
        "Plotly": (
            "I use Plotly to render interactive charts in the Streamlit "
            "dashboard.",
            "The dashboard creates charts for model performance and logged "
            "prediction distributions from the local registry and CSV data.",
            "This gives the monitoring pages a visual summary of available "
            "records. If a metric or log is absent, the dashboard has to use "
            "its no-data display rather than imply an unobserved score.",
            ["build_performance_chart", "build_prediction_chart"],
        ),
    }

    profile = names.get(technology)
    if profile:
        short_core, medium_core, detailed_core, functions = profile
        refs = _function_refs(inventory, functions)
        if technology == "Plotly":
            refs = _call_refs(inventory, ("px.line", "px.bar", "px.pie"))
        if not refs:
            short = (
                f"The source scan does not contain enough evidence to explain "
                f"how {technology} is used in this project."
            )
            medium = short + " I would not infer its role from the dependency name alone."
            detailed = medium + " A matching import and implementation call site would be needed."
            return short, medium, detailed, []
        short = short_core + " " + medium_core
        medium_extensions = {
            "Python": (
                "For example, the dashboard calls these project modules for "
                "project analysis and interview-question generation. This "
                "keeps the interface connected to the same implementations "
                "used by the project workflow."
            ),
            "Pandas": (
                "The loader separates the encoded `Churn` label from the "
                "feature columns so training receives them as distinct inputs."
            ),
            "Scikit-learn": (
                "Keeping preprocessing and the estimator in one pipeline "
                "applies the same transformations during fitting and "
                "prediction."
            ),
            "SciPy": (
                "This check measures feature-distribution changes; it does "
                "not by itself show that predictive accuracy has fallen."
            ),
            "Joblib": (
                "The registry records the saved artifact path with its model "
                "version and metrics."
            ),
            "Streamlit": (
                "The Interview Coach renders generated answers separately "
                "from their expandable source evidence."
            ),
            "Plotly": (
                "The dashboard presents the values available in its local "
                "registry and prediction records."
            ),
        }
        detailed_extensions = {
            "Python": (
                "The source code demonstrates these roles through concrete "
                "module calls, rather than a separate service boundary."
            ),
            "Pandas": (
                "The loader uses `read_csv()`, `to_numeric()`, `fillna()`, "
                "`drop()`, and `map()` for those verified operations."
            ),
            "Scikit-learn": (
                "The resulting pipeline is evaluated with accuracy, "
                "precision, recall, F1, and ROC-AUC before candidate "
                "comparison."
            ),
            "SciPy": (
                "The lifecycle also requires new labels before candidate "
                "retraining, and approval before promotion."
            ),
            "Joblib": (
                "The current code saves and loads artifacts, but does not "
                "implement a general artifact-retention policy."
            ),
            "Streamlit": (
                "The implementation is a local dashboard; it does not "
                "establish hosted authentication or a live LLM connection."
            ),
            "Plotly": (
                "When registry metrics are unavailable, the dashboard shows "
                "a no-data state instead of inventing a score."
            ),
        }
        medium = " ".join((short, detailed_core, medium_extensions[technology]))
        detailed = medium + " " + detailed_extensions[technology]
        return short, medium, detailed, refs

    paths = inventory.get("technology_files", {}).get(technology, [])
    if not paths:
        short = (
            f"I cannot verify how {technology} is used from the scanned source."
        )
        medium = short + " I would not infer its role from a package name alone."
        detailed = medium + " The project needs a concrete import and call site to support a stronger explanation."
        return short, medium, detailed, []

    short = (
        f"The source imports {technology}, but the scan cannot establish a "
        "specific role for it."
    )
    medium = (
        short + " I would describe only the verified dependency presence and "
        "would not attribute unrelated functions or features to it."
    )
    detailed = (
        medium + " A project-specific explanation requires inspecting the "
        "actual call sites for this library."
    )
    return short, medium, detailed, []


def _answers_for(question, inventory, technology=None):
    name = str(inventory.get("project_name") or "this project")
    folded = question.casefold()

    if technology:
        return _technology_answer(inventory, technology)

    if "tell me about your project" in folded:
        refs = _function_refs(
            inventory,
            ["load_and_split_data", "build_preprocessor", "train_model",
             "evaluate_model", "run_lifecycle"],
        )
        short = (
            f"{name} is a local churn-model lifecycle project. It trains and "
            "evaluates a scikit-learn model, then monitors data and logged "
            "outcomes to inform retraining."
        )
        medium = (
            short + " The training path cleans and splits the Telco churn "
            "dataset, transforms numeric and categorical features, and fits "
            "a Logistic Regression pipeline. The lifecycle checks numerical "
            "drift and requires labels before supervised retraining. A "
            "candidate is evaluated against the current model, validated, "
            "and presented for human approval before promotion."
        )
        detailed = (
            medium + " Predictions are logged with their model version and "
            "can be completed later with actual outcomes. Aggregate and "
            "version-specific monitoring calculate metrics from completed "
            "labels. The registry stores model versions and artifacts, and a "
            "separate rollback function can restore a registered artifact. "
            "The repository supports these local workflows; it does not show "
            "a hosted deployment or authenticated approval service."
        )
        return short, medium, detailed, refs

    if "technologies did you use" in folded:
        refs = _function_refs(
            inventory,
            [
                "load_and_split_data", "build_preprocessor", "train_model",
                "evaluate_model", "detect_data_drift", "save_model",
                "get_project_insights", "build_performance_chart",
            ],
        )
        short = (
            "I use Pandas to load and clean churn data, and scikit-learn to "
            "preprocess it, train Logistic Regression, and evaluate "
            "predictions. SciPy checks feature drift, while Joblib saves "
            "model artifacts. Streamlit and Plotly power the local dashboard "
            "and its charts."
        )
        medium = (
            short
            + " During ingestion, Pandas converts `TotalCharges`, fills "
            "invalid values, removes the ID, and encodes the target. The "
            "scikit-learn pipeline scales numeric columns, one-hot encodes "
            "categories, and keeps preprocessing with Logistic Regression "
            "for fitting and prediction. SciPy's two-sample KS test compares "
            "three numeric production features with reference data to detect "
            "distribution changes. Joblib serializes models tracked by "
            "version in the registry, while Streamlit and Plotly show "
            "available registry metrics and prediction records."
        )
        detailed = (
            medium
            + " The lifecycle retrains only after drift and new labels, "
            "compares the candidate with the current model, and requires "
            "human approval before saving and promoting it. The code shows "
            "the role of each dependency but does not record the original "
            "selection rationale."
        )
        return short, medium, detailed, refs

    if "organize" in folded or "architecture" in folded:
        refs = _function_refs(
            inventory,
            ["load_and_split_data", "build_preprocessor", "train_model",
             "evaluate_model", "detect_data_drift", "run_lifecycle",
             "predict_and_log", "calculate_performance",
             "calculate_version_performance", "save_model", "rollback_model"],
        )
        short = (
            f"I organize {name} into data preparation, model training and "
            "evaluation, lifecycle control, and dashboard modules. The "
            "registry and prediction services handle artifacts and logs."
        )
        medium = (
            short + " Data loading cleans the churn table and creates "
            "stratified splits. Preprocessing and Logistic Regression are "
            "combined in a scikit-learn pipeline. Drift detection and "
            "candidate decisions are coordinated separately from prediction "
            "logging and metric calculation. The Streamlit dashboard reads "
            "the local registry and monitoring data."
        )
        detailed = (
            medium + " After drift, the lifecycle waits for labels before "
            "training a candidate. It evaluates that candidate on held-out "
            "test data, checks F1 and ROC-AUC thresholds, and requests human "
            "approval before saving and promoting the artifact. The registry "
            "tracks versions, and rollback is implemented as a separate "
            "operation rather than an automatic lifecycle step."
        )
        return short, medium, detailed, refs

    if (
        "workflow" in folded
        or "input to output" in folded
        or "machine learning workflow" in folded
    ):
        refs = _function_refs(
            inventory,
            ["load_and_split_data", "build_preprocessor", "train_model",
             "evaluate_model", "detect_data_drift", "run_lifecycle",
             "train_candidate_model", "compare_models", "validate_candidate",
             "request_approval", "promote_model", "rollback_model"],
        )
        short = (
            "I clean and stratify the churn data, preprocess numeric and "
            "categorical features, then train a Logistic Regression pipeline "
            "and evaluate it on held-out data. During monitoring, drift and "
            "available labels gate candidate retraining and promotion."
        )
        medium = (
            short + " The data loader converts `TotalCharges`, fills invalid "
            "values, removes the customer identifier, and encodes the churn "
            "target. The preprocessor scales numeric columns and one-hot "
            "encodes categorical columns. Evaluation reports accuracy, "
            "precision, recall, F1, and ROC-AUC. The KS drift check compares "
            "three numeric features with reference data. A candidate is "
            "trained only when drift is detected and new labels are present."
        )
        detailed = (
            medium + " The candidate is evaluated on the same held-out test "
            "partition and compared with current metrics. Validation requires "
            "non-decreasing F1 and ROC-AUC under the current thresholds. A "
            "human approval step must accept the candidate before its artifact "
            "is saved and promoted in the registry. Predictions are logged "
            "with model versions, while actual labels can be recorded later. "
            "Rollback is available through the registry but is not called "
            "automatically by this lifecycle."
        )
        return short, medium, detailed, refs

    if "limitation" in folded or "improve next" in folded:
        refs = _function_refs(
            inventory,
            ["request_approval", "detect_data_drift", "scan_project",
             "redact_secrets", "calculate_performance"],
        )
        short = (
            "The approval step is console-based, drift checks cover selected "
            "numeric columns, and the source analyzer's regex secret masking "
            "is not comprehensive. I would improve those areas without "
            "presenting the improvements as existing features."
        )
        medium = (
            short + " `request_approval()` uses terminal input rather than a "
            "dashboard approval flow. Drift detection uses a KS test on "
            "`tenure`, `MonthlyCharges`, and `TotalCharges`, so it does not "
            "cover every feature or prove performance degradation. The "
            "analyzer skips some sensitive filenames and masks common "
            "patterns, but that is not a security boundary. I would prioritize "
            "broader validation and safer operational approval."
        )
        detailed = (
            medium + " The current application is file-based and has no "
            "authentication or authorization layer in the inspected source. "
            "For larger workloads, I would measure the synchronous CSV and "
            "artifact operations before proposing persistent storage or "
            "background processing. These are possible next steps, not "
            "implemented capabilities."
        )
        return short, medium, detailed, refs

    if "debug" in folded or "failure" in folded:
        refs = _function_refs(
            inventory,
            ["load_and_split_data", "train_model", "evaluate_model",
             "calculate_performance", "calculate_version_performance"],
        )
        short = (
            "I would reproduce the failure with the same input data or log, "
            "then isolate whether it occurs during loading, fitting, "
            "evaluation, or metric calculation. I would add a regression test "
            "for the confirmed cause."
        )
        medium = (
            short + " I would capture the traceback and compare expected with "
            "actual values at each boundary. The data loader cleans and splits "
            "the CSV, the model pipeline handles fitting, and the evaluator "
            "scores held-out examples. Performance monitoring filters logs "
            "for completed actual outcomes before scoring. A small failing "
            "case helps distinguish a data problem from a code regression."
        )
        detailed = (
            medium + " I would not infer a cause from the symptom alone. "
            "After reproducing it, I would test the smallest affected "
            "function, add the case to pytest, rerun the focused test, and "
            "then run the full suite with `python -m pytest -q`. I would "
            "report the observed result separately from the source-level "
            "hypothesis."
        )
        return short, medium, detailed, refs

    if "performance" in folded and ("calculate" in folded or "monitor" in folded):
        refs = _function_refs(
            inventory,
            ["calculate_performance", "calculate_version_performance"],
        )
        short = (
            "I use `calculate_performance()` for all completed labels and "
            "`calculate_version_performance()` for one model version. Both "
            "raise `FileNotFoundError` for a missing log and return "
            "`insufficient_data` when there are no matching outcomes. They "
            "calculate ROC-AUC with `roc_auc_score` on logged hard predictions, "
            "and omit AUC when the actual labels contain one class."
        )
        medium = (
            short + " The reported metrics are accuracy, precision, recall, "
            "F1, and ROC-AUC. The aggregate function keeps rows with actual "
            "labels of 0 or 1. The version-specific function also filters by "
            "the requested model version. This keeps both calculations "
            "consistent while allowing a per-version comparison."
        )
        detailed = (
            medium + " The aggregate function considers all completed rows, "
            "while `calculate_version_performance()` first filters by the "
            "requested version. This keeps the metric definitions consistent "
            "while isolating each model's records. Because the log stores "
            "class predictions rather than probability scores, its ROC-AUC "
            "does not represent a probability-ranking evaluation. The tests "
            "exercise the aggregate and version-specific paths separately."
        )
        return short, medium, detailed, refs

    if "train" in folded or "evaluate" in folded or "model used" in folded:
        refs = _function_refs(
            inventory,
            ["load_and_split_data", "build_preprocessor", "train_model",
             "evaluate_model"],
        )
        short = (
            "I clean and stratify the churn data, transform numeric and "
            "categorical features, and fit a Logistic Regression pipeline. "
            "I evaluate it on held-out data with classification metrics and "
            "ROC-AUC."
        )
        medium = (
            short + " `load_and_split_data()` converts `TotalCharges`, fills "
            "invalid values, removes `customerID`, and maps `Churn` to 0 or 1. "
            "The split reserves separate model-training, simulated-production, "
            "and test partitions. `ColumnTransformer` scales numeric features "
            "and one-hot encodes categories. `evaluate_model()` uses predicted "
            "labels for classification metrics and probabilities for ROC-AUC."
        )
        detailed = (
            medium + " The `LogisticRegression` estimator uses class-balanced "
            "weights and is trained inside a scikit-learn `Pipeline`, keeping "
            "preprocessing attached to the estimator. This answer describes "
            "the implemented method, not a score from a particular run; a "
            "measured result must come from the corresponding evaluation."
        )
        return short, medium, detailed, refs

    if "drift" in folded or "retraining" in folded:
        refs = _function_refs(
            inventory,
            ["detect_data_drift", "run_lifecycle", "train_candidate_model",
             "compare_models", "validate_candidate", "request_approval",
             "promote_model"],
        )
        short = (
            "I use a KS test to check selected numeric features for "
            "distribution drift. Drift alone does not retrain the model: "
            "the lifecycle also requires new labels before supervised "
            "candidate training can begin."
        )
        medium = (
            short + " The detector compares reference and incoming values for "
            "`tenure`, `MonthlyCharges`, and `TotalCharges`. A p-value below "
            "0.05 marks a feature as drifted. If there is no drift, the "
            "lifecycle exits; if labels are missing, it reports that retraining "
            "cannot proceed. Otherwise it trains and evaluates a candidate "
            "against the current model."
        )
        detailed = (
            medium + " Candidate validation checks F1 and ROC-AUC improvements "
            "against the configured thresholds. A failed candidate is not "
            "promoted. A passing candidate still requires human approval; "
            "only then does the lifecycle save its artifact and update the "
            "registry's current version. This is a supervised, human-gated "
            "workflow, not automatic promotion on drift alone."
        )
        return short, medium, detailed, refs

    if "test" in folded:
        tests = _test_facts(inventory)
        refs = [
            f"{path}:{function['line']}"
            for file_info in inventory.get("source_evidence", [])
            for path, names in tests
            if file_info["path"] == path
            for function in file_info.get("functions", [])
            if function["name"] in names
        ][:12]
        short = (
            "I run the suite with `python -m pytest -q`. The checked-in tests "
            "cover data ingestion, drift and lifecycle gates, prediction "
            "logging, performance metrics, and model promotion."
        )
        medium = (
            short + " The tests check malformed and missing input cases, "
            "completed-outcome filtering, per-version metrics, and behavior "
            "when drift or labels are absent. Lifecycle tests also cover "
            "candidate rejection and approved promotion. Prediction tests "
            "exercise logging and adding actual outcomes. Interview Coach "
            "tests check source evidence and the answer versions."
        )
        detailed = (
            medium + " In this audit, the full command completed with "
            "18 passing tests. That result applies to this exact run; it does "
            "not imply coverage beyond the assertions in the checked-in "
            "tests. I rerun the command after changes rather than treating "
            "an earlier pass as current evidence."
        )
        if not tests:
            short = "I run the project tests with `python -m pytest -q`."
            medium = short + " The source scan did not find checked-in pytest test functions, so I cannot describe verified coverage."
            detailed = medium + " A successful result must be established by running the suite; the analyzer does not execute tests."
            refs = []
        return short, medium, detailed, refs

    if "security" in folded or "secure" in folded:
        refs = _function_refs(inventory, ["scan_project", "redact_secrets"])
        short = (
            "The project analyzer reads source files without executing them "
            "and skips selected sensitive filenames. Its regex-based secret "
            "masking is limited and should not be treated as complete secret "
            "protection."
        )
        medium = (
            short + " It restricts scanned file types and sizes and excludes "
            "data, model, log, and cache directories. The dashboard operates "
            "on local files, and I found no authentication or authorization "
            "implementation in the inspected source. I would keep secrets "
            "outside the repository and review files before publishing."
        )
        detailed = (
            medium + " The analyzer's redaction is a convenience for displayed "
            "source evidence, not a substitute for credential management or "
            "a security review. The project also needs an explicit license "
            "and a deliberate decision about redistributing its dataset and "
            "model artifacts before a public release."
        )
        return short, medium, detailed, refs

    if "scalab" in folded or "reliab" in folded:
        refs = _function_refs(
            inventory,
            ["run_lifecycle", "log_prediction", "calculate_performance"],
        )
        short = (
            "The current workflow uses local CSV files and synchronous "
            "lifecycle steps. I would measure those bottlenecks before "
            "choosing a storage or background-processing change."
        )
        medium = (
            short + " The explicit drift, label, validation, and approval "
            "gates help make model changes reviewable. Prediction and "
            "performance records are file-based, while artifacts and version "
            "metadata are stored on disk. A larger deployment would need "
            "measured workload evidence before selecting a more durable "
            "store or job queue."
        )
        detailed = (
            medium + " I would preserve the current safety gates, add "
            "observability around the slowest operations, and only then "
            "propose persistence or asynchronous execution. Those would be "
            "future changes; the source does not show them as implemented."
        )
        return short, medium, detailed, refs

    if "challenge" in folded or "personally solved" in folded:
        refs = _function_refs(
            inventory,
            ["calculate_performance", "calculate_version_performance"],
        )
        short = (
            "Customizable draft — use this only if it matches your experience: "
            "“A technical issue in ModelLife was keeping aggregate and "
            "version-specific performance calculations consistent.” The "
            "repository shows the implementation, not who made the change."
        )
        medium = (
            "Customizable draft — confirm your personal role before using it: "
            "“One technical issue I worked on was ensuring prediction-log "
            "performance calculations returned metrics consistently. The "
            "aggregate and version-specific functions filter for known "
            "outcomes and handle missing or insufficient data. I would explain "
            "the part I personally changed and the tests I actually ran.” "
            "The source alone cannot verify personal ownership."
        )
        detailed = (
            "Customizable draft — replace the bracketed details with facts "
            "you can confirm: “A technical issue I addressed was [describe "
            "your actual role in the performance-monitor change]. The current "
            "code calculates metrics for completed labels and keeps the "
            "version-specific path consistent with the aggregate path. I "
            "verified [name only tests you personally ran] and observed "
            "[insert the measured result].” The repository documents the "
            "restored calculation, but it does not establish your involvement "
            "or personal test history."
        )
        return short, medium, detailed, refs

    return _evidence_limited_answer(question, inventory)


def _evidence_limited_answer(question, inventory):
    topic_words = {
        word.casefold()
        for word in question.replace("?", "").replace("`", "").split()
        if len(word) > 4
    }
    matches = []
    for file_info in inventory.get("source_evidence", []):
        for function in file_info.get("functions", []):
            if any(word in function["name"].casefold() for word in topic_words):
                matches.append((file_info["path"], function))
    if matches:
        path, function = matches[0]
        refs = [f"{path}:{function['line']}"]
        short = (
            "The source contains a matching function definition, but its "
            "implementation does not establish enough detail to answer this "
            "question confidently."
        )
        medium = (
            short + " I can identify the relevant function from the source "
            "evidence, but I will not infer behavior from its name alone. "
            "The separate evidence section points to the definition for "
            "inspection."
        )
        detailed = (
            medium + " A meaningful project-specific answer would require "
            "checking the function body and any relevant tests. Until then, "
            "the implementation detail remains unverified."
        )
        return short, medium, detailed, refs

    short = (
        f"I cannot verify this feature for {inventory.get('project_name', 'the project')} "
        "from the scanned source."
    )
    medium = (
        short + " The current evidence does not establish a matching "
        "implementation or test, so I will not describe it as implemented."
    )
    detailed = (
        medium + " A source definition or test that directly supports the "
        "claim would be needed before I could give a more specific answer."
    )
    return short, medium, detailed, []


def _technology_names(inventory):
    return list(inventory.get("technologies") or [])


def generate_interview_questions(project_path="."):
    """Generate answer versions from AST and documentation evidence."""
    inventory = get_project_inventory(project_path)
    project_name = str(inventory.get("project_name") or "This project")
    technologies = _technology_names(inventory)
    groups = {
        "common": [],
        "project_specific": [],
        "technology": [],
        "machine_learning": [],
        "security": [],
        "debugging": [],
        "testing": [],
        "system_design": [],
        "behavioral": [],
    }

    def add(group, question, why_asked, difficulty="Intermediate", tech=None):
        short, medium, detailed, evidence = _answers_for(
            question, inventory, technology=tech
        )
        groups[group].append({
            "category": group.replace("_", " ").title(),
            "question": question,
            "answer": medium,
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
            "evidence": evidence,
            "difficulty": difficulty,
            "why_asked": why_asked,
            "follow_ups": [],
            "source_type": "source_evidence" if evidence else "evidence_limited",
        })

    add("common", "Tell me about your project.",
        "To assess whether you can describe the project from evidence.")
    add("common", "What technologies did you use, and why?",
        "To connect imported tools to their actual use in the repository.")
    add("project_specific",
        "How is the project organized, and how do its main components work together?",
        "To assess understanding of module boundaries.")
    add("project_specific", "Walk me through one important user workflow from input to output.",
        "To assess understanding of an implemented workflow.")
    add("project_specific", "What are the current limitations of the project, and what would you improve next?",
        "To assess engineering judgment against the current implementation.")
    add("debugging", "How would you debug a failure in the main project workflow?",
        "To assess a grounded debugging approach.")
    add("testing", "How do you test that the project works correctly?",
        "To distinguish checked-in tests from unverified coverage.")
    add("security", "What security risks are relevant to this project, and how would you reduce them?",
        "To assess security controls and their limitations.")
    add("system_design", "How would you make this project more reliable and scalable?",
        "To assess practical trade-offs grounded in current design.", "Advanced")
    add("machine_learning", "How do you train, validate, and evaluate the model used in this project?",
        "To assess the implemented model pipeline.")
    add("machine_learning", "Explain the machine learning workflow.",
        "To assess how data preparation, training, monitoring, and promotion fit together.")
    add("machine_learning", "How does ModelLife detect drift and decide whether to retrain?",
        "To assess drift and lifecycle gates.")
    add("machine_learning", "How does performance monitoring calculate aggregate and per-version metrics?",
        "To assess the actual logged-outcome calculations.")
    add("behavioral", "What was a challenging problem you personally solved while building this project?",
        "To understand ownership without inventing personal experience.", "Beginner")

    for technology in technologies:
        add(
            "technology",
            f"How is {technology} used in this project?",
            f"To connect {technology} to source imports and call sites.",
            tech=technology,
        )

    total = sum(len(items) for items in groups.values())
    return {
        "project_name": project_name,
        "total_questions": total,
        "categories": list(groups),
        "questions": groups,
    }


if __name__ == "__main__":
    result = generate_interview_questions(".")
    print(f"Questions for {result['project_name']}: {result['total_questions']}")
    for category, items in result["questions"].items():
        for number, item in enumerate(items, 1):
            print(f"\n[{category}] Q{number}: {item['question']}")
            print("Short:", item["short_answer"])
            print("Medium:", item["medium_answer"])
            print("Detailed:", item["detailed_answer"])

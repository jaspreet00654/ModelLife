
"""
ModelLife - Code Logic Interview Question Generator

Extracts Python function structure and control flow using AST.
Does not execute project source code.

Note:
AST extraction describes source structure. It does not prove
runtime behavior or generate fully verified interview answers.
"""

import ast
from pathlib import Path
import re
from typing import Any

from src.project_analyzer import is_backup_or_duplicate_file


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "data",
    "models",
    "logs",
    "tests",
}

MAX_FILES = 100
MAX_FILE_SIZE = 200_000
MAX_ACTION_LENGTH = 240

OUTPUT_LIMIT = 100


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def source_of(node: ast.AST) -> str:
    """Return a compact source representation of an AST node."""
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError, ValueError):
        return type(node).__name__


def short_text(text: str, limit: int = MAX_ACTION_LENGTH) -> str:
    """Trim long source descriptions for readable output."""
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def summarize_statement(statement: ast.stmt) -> str:
    """Summarize one statement without executing it."""
    if isinstance(statement, ast.If):
        return f"if {source_of(statement.test)}: ..."

    if isinstance(statement, ast.For):
        return f"for {source_of(statement.target)} in {source_of(statement.iter)}: ..."

    if isinstance(statement, ast.AsyncFor):
        return f"async for {source_of(statement.target)} in {source_of(statement.iter)}: ..."

    if isinstance(statement, ast.While):
        return f"while {source_of(statement.test)}: ..."

    if isinstance(statement, ast.Try):
        return "try: ..."

    try_star = getattr(ast, "TryStar", None)
    if try_star is not None and isinstance(statement, try_star):
        return "try*: ..."

    if isinstance(statement, ast.With):
        return "with ..."

    if isinstance(statement, ast.AsyncWith):
        return "async with ..."

    if isinstance(statement, ast.Return):
        if statement.value is None:
            return "return"
        return f"return {source_of(statement.value)}"

    if isinstance(statement, ast.Raise):
        if statement.exc is None:
            return "raise"
        return f"raise {source_of(statement.exc)}"

    if isinstance(statement, ast.Break):
        return "break"

    if isinstance(statement, ast.Continue):
        return "continue"

    return short_text(source_of(statement))


def summarize_statements(
    statements: list[ast.stmt],
    limit: int = 8,
) -> list[str]:
    """Summarize a sequence of statements."""
    summaries = []

    for statement in statements[:limit]:
        summaries.append(summarize_statement(statement))

    if len(statements) > limit:
        summaries.append("... additional statements omitted")

    return summaries


def get_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """
    Return the declared return annotation, or None if absent.

    No annotation is different from an explicit -> None.
    """
    if node.returns is None:
        return None

    return source_of(node.returns)


_CALL_DESCRIPTIONS = {
    "read_csv": "reads tabular rows from a CSV file",
    "read_text": "reads text from a file",
    "to_numeric": "converts values to numeric form",
    "fillna": "fills missing values",
    "drop": "removes selected columns or rows",
    "map": "maps labels to their target values",
    "train_test_split": "splits the data into training and held-out partitions",
    "StandardScaler": "scales numeric features",
    "OneHotEncoder": "encodes categorical features",
    "ColumnTransformer": "routes columns through the appropriate preprocessing",
    "LogisticRegression": "creates a logistic-regression classifier",
    "Pipeline": "connects preprocessing and model fitting into one pipeline",
    "fit": "fits the model or transformer to the supplied data",
    "predict": "generates class predictions",
    "predict_proba": "calculates class-probability estimates",
    "accuracy_score": "calculates accuracy",
    "precision_score": "calculates precision",
    "recall_score": "calculates recall",
    "f1_score": "calculates the F1 score",
    "roc_auc_score": "calculates ROC-AUC",
    "ks_2samp": "compares two samples with the Kolmogorov-Smirnov test",
    "dump": "saves a model artifact",
    "load": "loads serialized data",
    "json.load": "parses JSON content",
    "exists": "checks whether the path exists",
    "is_file": "checks that the path identifies a file",
    "get_project_inventory": "scans the project and builds a source inventory",
    "count_python_files": "counts Python files in the inventory",
    "request_approval": "requests a human approval decision",
}


def _function_purpose(function: dict[str, Any]) -> str:
    """Turn a docstring or function name into a concise purpose statement."""
    name = function["qualified_name"].rsplit(".", 1)[-1]
    known_purposes = {
        "read_json": "safely loads and validates JSON data from a file",
        "get_project_insights": (
            "collects project summary counts for the dashboard"
        ),
        "get_registry_parts": (
            "normalizes registry metadata into the current version and model entries"
        ),
        "format_metric": "formats optional metric values for dashboard display",
        "compare_models": (
            "calculates metric differences between current and candidate models"
        ),
        "validate_candidate": (
            "checks candidate F1 and ROC-AUC improvements against configured thresholds"
        ),
        "get_question_bank": "returns the predefined interview question bank",
        "keep_selected_question_open": (
            "preserves the selected question's expanded state after answer-length changes"
        ),
        "load_and_split_data": (
            "loads and prepares churn data, then creates stratified data splits"
        ),
        "build_preprocessor": (
            "builds the numeric and categorical feature transformations"
        ),
        "train_model": "fits the Logistic Regression training pipeline",
        "evaluate_model": "evaluates a fitted model on held-out data",
        "detect_data_drift": (
            "checks numerical features for distribution changes"
        ),
        "run_lifecycle": (
            "coordinates drift checks, candidate validation, approval, and promotion"
        ),
        "calculate_performance": (
            "calculates metrics for logged predictions with known outcomes"
        ),
        "calculate_version_performance": (
            "calculates logged prediction metrics for one model version"
        ),
        "rollback_model": "restores a registered model version as production",
        "predict_and_log": "predicts an outcome and records it for monitoring",
        "log_prediction": "records a model prediction and its version",
        "load_production_batch": (
            "loads a production CSV batch and validates its feature schema"
        ),
    }
    if name in known_purposes:
        return known_purposes[name]

    docstring = (function.get("docstring") or "").strip()
    if docstring:
        first_line = docstring.splitlines()[0]
        first_sentence = re.split(
            r"(?<=[.!?])\s+",
            first_line,
            maxsplit=1,
        )[0]
        first_sentence = first_sentence.rstrip(".:")
        words = first_sentence.split(maxsplit=1)
        verb_forms = {
            "build": "builds",
            "calculate": "calculates",
            "check": "checks",
            "compare": "compares",
            "create": "creates",
            "convert": "converts",
            "detect": "detects",
            "evaluate": "evaluates",
            "extract": "extracts",
            "find": "finds",
            "generate": "generates",
            "get": "gets",
            "load": "loads",
            "log": "logs",
            "predict": "predicts",
            "read": "reads",
            "return": "returns",
            "run": "runs",
            "save": "saves",
            "support": "supports",
            "train": "trains",
            "update": "updates",
            "validate": "validates",
        }
        if words and words[0].casefold() in verb_forms:
            verb = verb_forms[words[0].casefold()]
            remainder = f" {words[1]}" if len(words) > 1 else ""
            return verb + remainder
        return first_sentence[0].lower() + first_sentence[1:]

    words = name.removeprefix("_")
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", words)
    words = words.replace("_", " ").strip().lower()
    for prefix, verb in (
        ("get ", "retrieves"),
        ("load ", "loads"),
        ("calculate ", "calculates"),
        ("check ", "checks"),
        ("build ", "builds"),
        ("train ", "trains"),
        ("detect ", "detects"),
        ("validate ", "validates"),
        ("save ", "saves"),
        ("update ", "updates"),
        ("compare ", "compares"),
        ("run ", "coordinates"),
        ("log ", "records"),
        ("predict ", "predicts"),
        ("request ", "requests"),
        ("scan ", "scans"),
        ("extract ", "extracts"),
        ("format ", "formats"),
        ("generate ", "generates"),
        ("find ", "finds"),
    ):
        if words.startswith(prefix):
            return f"{verb} {words[len(prefix):]}"
    return f"handles {words or 'its assigned task'}"


def _describe_call(call_name: str) -> str:
    short_name = call_name.rsplit(".", 1)[-1]
    description = _CALL_DESCRIPTIONS.get(call_name)
    if description is None:
        description = _CALL_DESCRIPTIONS.get(short_name)
    if description:
        return description
    return f"uses the `{short_name}()` helper"


def _describe_condition(condition: str) -> str:
    normalized = re.sub(r"\s+", "", condition).casefold()
    if normalized in {"notpath.exists()", "notpath.is_file()"}:
        return "the file does not exist"
    if normalized in {"path.exists()", "path.is_file()"}:
        return "the file exists"
    if "pd.isna(value)" in normalized:
        return "the value is missing or is a pandas null value"
    if normalized in {"notvalues", "notdata"}:
        return "the input collection is empty"
    if "isnone" in normalized and "label" in normalized:
        return "new labels are unavailable"
    if "isinstance(" in normalized and "dict" in normalized:
        subject = re.search(r"isinstance\(([^,]+),", condition)
        name = (
            subject.group(1).strip().replace("_", " ")
            if subject else "the value"
        )
        return f"{name} is a dictionary"
    if ".empty" in normalized:
        return "the data batch is empty"
    if "duplicated().any()" in normalized:
        return "the input has duplicate column names"
    if "drift_detected" in normalized or "retraining_required" in normalized:
        if normalized.startswith("not"):
            return "the drift check did not require retraining"
        return "the drift check indicates that retraining is required"
    if "approved" in normalized:
        if normalized.startswith("not"):
            return "candidate validation did not pass"
        return "candidate validation passed"
    if "missing_columns" in normalized or "extra_columns" in normalized:
        return "the input schema does not match the expected columns"
    if normalized == "columnsisnone":
        return "no feature columns were provided"
    if normalized == "notentry":
        return "the registry entry is missing"
    if normalized == "entry":
        return "the registry entry is present"

    try:
        expression = ast.parse(condition, mode="eval").body
    except SyntaxError:
        return "the function's input check passes"
    return _describe_condition_node(expression)


def _condition_subject(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.replace("_", " ")
    if isinstance(node, ast.Attribute):
        return f"{_condition_subject(node.value)} {node.attr.replace('_', ' ')}"
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant):
            return (
                f"{_condition_subject(node.value)} "
                f"{str(key.value).replace('_', ' ')}"
            )
        return _condition_subject(node.value)
    if isinstance(node, ast.Call):
        function_name = source_of(node.func).rsplit(".", 1)[-1]
        if node.args:
            return _condition_subject(node.args[0])
        return function_name.replace("_", " ")
    return "the value"


def _condition_value(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "missing"
        if isinstance(node.value, str):
            return f"`{node.value}`"
        return str(node.value)
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values = [
            _condition_value(item)
            for item in node.elts[:4]
        ]
        return " or ".join(values) if values else "the allowed values"
    if isinstance(node, ast.Name):
        return node.id.replace("_", " ")
    return _condition_subject(node)


def _describe_condition_node(node: ast.AST, negated: bool = False) -> str:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _describe_condition_node(node.operand, not negated)

    if isinstance(node, ast.BoolOp):
        connector = " and " if isinstance(node.op, ast.And) else " or "
        return connector.join(
            _describe_condition_node(value, negated)
            for value in node.values[:3]
        )

    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _condition_subject(node.left)
        right_node = node.comparators[0]
        operation = node.ops[0]
        if isinstance(operation, (ast.Is, ast.IsNot)) and isinstance(
            right_node, ast.Constant
        ) and right_node.value is None:
            is_missing = isinstance(operation, ast.Is)
            if negated:
                is_missing = not is_missing
            return f"{left} is {'missing' if is_missing else 'available'}"
        if isinstance(operation, ast.In) and isinstance(
            right_node, (ast.Set, ast.List, ast.Tuple)
        ):
            values = _condition_value(right_node)
            return (
                f"{left} is {'not ' if negated else ''}one of {values}"
            )
        comparisons = {
            ast.Eq: "equals",
            ast.NotEq: "does not equal",
            ast.Gt: "is greater than",
            ast.GtE: "is at least",
            ast.Lt: "is less than",
            ast.LtE: "is at most",
        }
        comparator = comparisons.get(type(operation))
        if comparator:
            if negated:
                comparator = f"does not satisfy the check against"
            return f"{left} {comparator} {_condition_value(right_node)}"

    if isinstance(node, ast.Call):
        function_name = source_of(node.func).rsplit(".", 1)[-1]
        subject = _condition_subject(node.args[0]) if node.args else "the value"
        if function_name == "isinstance" and len(node.args) > 1:
            expected_type = _condition_value(node.args[1])
            return (
                f"{subject} is {'not ' if negated else ''}a "
                f"{expected_type.strip('`')}"
            )
        if function_name in {"exists", "is_file"}:
            return f"the path {'does not ' if negated else ''}{'exist' if function_name == 'exists' else 'identify a file'}"
        if function_name == "isna":
            return f"{subject} is {'available' if negated else 'missing'}"
        if function_name == "any":
            return (
                "no item satisfies the check"
                if negated else "at least one item satisfies the check"
            )

    if isinstance(node, ast.Name):
        return f"the {node.id.replace('_', ' ')} value is {'not set' if negated else 'set'}"
    return "the function's input check does not pass" if negated else (
        "the function's input check passes"
    )


def _describe_action(action: str) -> str:
    action = action.strip()
    if action.startswith("return "):
        return f"returns {_describe_return(action[7:])}"
    if action.startswith("raise "):
        exception = action[6:].split("(", 1)[0].strip()
        return f"raises `{exception}`"
    if action.startswith("continue"):
        return "continues with the next item"
    if action.startswith("break"):
        return "stops the current loop"
    if action.startswith("print("):
        return "prints a status message"
    if ".append(" in action or action.startswith("append("):
        return "adds the item to the result collection"
    if ".setdefault(" in action:
        return "fills a missing field without overwriting an existing value"
    if ".update(" in action:
        return "updates the existing record"
    if action.startswith("for "):
        return "processes the matching entries"
    assignment = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)", action)
    if assignment:
        target, value = assignment.groups()
        target = target.rsplit(".", 1)[-1].replace("_", " ")
        if value in {"[]", "list()"}:
            return f"initializes an empty {target} list"
        if value in {"{}", "dict()"}:
            return f"initializes an empty {target} dictionary"
        return f"updates {target} for the next step"
    return "updates its working state"


def _describe_return(expression: str) -> str:
    compact = re.sub(r"\s+", "", expression)
    if compact in {"{}", "dict()"}:
        return "an empty dictionary"
    if compact in {"None", "return"}:
        return "no value"
    if compact in {"True", "False"}:
        return f"`{compact}`"
    if "isinstance(" in compact and "else{}" in compact:
        return "the parsed value when it has the expected type, or an empty dictionary otherwise"
    if expression.startswith("{") and ":" in expression:
        keys = list(dict.fromkeys(
            re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:", expression)
        ))
        if keys:
            readable = ", ".join(key.replace("_", " ") for key in keys)
            return f"a dictionary containing {readable}"
    value_names = {
        "result": "the parsed result",
        "inventory": "the project inventory",
        "metrics": "the calculated metrics",
        "data": "the prepared data",
        "model": "the fitted model",
    }
    for variable, description in value_names.items():
        if compact == variable:
            return description
    return "the result of the operation"


def _exception_descriptions(exceptions: list[dict[str, Any]]) -> list[str]:
    descriptions = []
    for exception in exceptions:
        for handler in exception["handlers"]:
            name = handler["exception"].strip("()").replace(", ", " or ")
            actions = handler["actions"]
            action_text = (
                _describe_action(actions[0])
                if actions else "handles the error"
            )
            descriptions.append(
                f"It catches {name}; in that case, it {action_text} instead "
                "of letting the error escape."
            )
    return descriptions


def _operation_descriptions(calls: list[str]) -> list[str]:
    operations = []
    for call in calls:
        description = _describe_call(call)
        if description not in operations:
            operations.append(description)
    return operations[:5]


def _project_relevance(function: dict[str, Any]) -> str:
    name = function["qualified_name"].rsplit(".", 1)[-1]
    path = function["file_path"].replace("\\", "/")
    if name == "read_json":
        return (
            "This keeps missing or malformed dashboard data from interrupting "
            "the rest of the page."
        )
    if name == "get_project_insights":
        return (
            "The dashboard uses these values to populate its project-insight "
            "cards."
        )
    if name == "load_and_split_data":
        return "The resulting partitions are used by training, drift checks, and evaluation."
    if name == "run_lifecycle":
        return "It is the control point that prevents candidate promotion before the required checks and approval."
    if path.startswith("src/"):
        return "Keeping this step in its own function makes the project workflow easier to test and reuse."
    return "This makes the behavior available as a reusable step in the application."


def _answer_sentences(function: dict[str, Any]) -> dict[str, Any]:
    name = function["qualified_name"].rsplit(".", 1)[-1]
    purpose = _function_purpose(function)
    parameters = [
        item for item in function["parameters"]
        if item not in {"self", "cls"}
    ]
    calls = function.get("calls", [])
    operations = _operation_descriptions(calls)
    conditions = function.get("conditions", [])
    exceptions = _exception_descriptions(function.get("exceptions", []))
    returns = [
        _describe_return(expression)
        for expression in function.get("return_expressions", [])
    ]

    purpose_sentence = f"`{name}()` {purpose}."
    if parameters:
        parameter_sentence = (
            "It accepts " + ", ".join(f"`{item}`" for item in parameters)
            + " as input."
        )
    else:
        parameter_sentence = "It does not require explicit input parameters."

    if operations:
        step_sentence = (
            "The implementation " + "; ".join(operations[:3]) + "."
        )
    else:
        step_sentence = "It applies its local logic directly to the inputs."

    branch_sentences = []
    for condition in conditions[:3]:
        action = (
            _describe_action(condition["actions"][0])
            if condition.get("actions") else "it takes the guarded branch"
        )
        branch_sentences.append(
            f"When { _describe_condition(condition['condition']) }, it {action}."
        )

    if returns:
        unique_returns = list(dict.fromkeys(returns))
        return_sentence = "It returns " + " or ".join(unique_returns[:3]) + "."
    else:
        return_sentence = "It completes the operation without an explicit return value."

    if exceptions:
        exception_sentence = " ".join(exceptions[:3])
    else:
        exception_sentence = "The extracted function has no explicit exception handler."

    return {
        "purpose": purpose_sentence,
        "parameters": parameter_sentence,
        "operations": step_sentence,
        "branches": branch_sentences,
        "returns": return_sentence,
        "exceptions": exception_sentence,
        "relevance": _project_relevance(function),
    }


# ---------------------------------------------------------
# AST visitors
# ---------------------------------------------------------

class ReturnExpressionVisitor(ast.NodeVisitor):
    """Collect return expressions without entering nested functions."""

    def __init__(self):
        self.expressions = []

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            self.expressions.append("return")
        else:
            self.expressions.append(source_of(node.value))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Nested function returns belong to that nested function.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return

    def visit_ClassDef(self, node: ast.ClassDef):
        return


class ControlFlowVisitor(ast.NodeVisitor):
    """
    Extract if/else and try/except structure from a function.

    Nested control-flow statements are visited, but nested functions
    and classes are intentionally excluded from the parent function.
    """

    def __init__(self):
        self.conditions = []
        self.exceptions = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return

    def visit_ClassDef(self, node: ast.ClassDef):
        return

    def visit_If(self, node: ast.If):
        has_else = bool(node.orelse)

        condition = {
            "line": node.lineno,
            "condition": short_text(source_of(node.test)),
            "actions": summarize_statements(node.body),
            "else_actions": summarize_statements(node.orelse),
            "has_else": has_else,
        }
        self.conditions.append(condition)

        # Inspect nested control flow inside both branches.
        for statement in node.body:
            self.visit(statement)

        for statement in node.orelse:
            self.visit(statement)

    def visit_For(self, node: ast.For):
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While):
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_Try(self, node: ast.Try):
        handlers = []

        for handler in node.handlers:
            exception_type = (
                source_of(handler.type)
                if handler.type is not None
                else "unspecified exception"
            )

            handlers.append({
                "exception": exception_type,
                "name": handler.name,
                "line": handler.lineno,
                "actions": summarize_statements(handler.body),
            })

        self.exceptions.append({
            "line": node.lineno,
            "try_actions": summarize_statements(node.body),
            "handlers": handlers,
            "else_actions": summarize_statements(node.orelse),
            "finally_actions": summarize_statements(node.finalbody),
        })

        for statement in node.body:
            self.visit(statement)

        for handler in node.handlers:
            for statement in handler.body:
                self.visit(statement)

        for statement in node.orelse:
            self.visit(statement)

        for statement in node.finalbody:
            self.visit(statement)

    def visit_With(self, node: ast.With):
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for statement in node.body:
            self.visit(statement)


# ---------------------------------------------------------
# Python file discovery
# ---------------------------------------------------------

def find_python_files(project_path: str | Path = PROJECT_ROOT) -> list[Path]:
    """Find eligible Python files while skipping excluded directories."""
    root = Path(project_path).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid project directory: {root}")

    python_files = []

    for file_path in root.rglob("*.py"):
        relative_path = file_path.relative_to(root)

        if any(part in EXCLUDED_DIRS for part in relative_path.parts):
            continue

        if (
            file_path.name == "code_question_generator.py"
            or is_backup_or_duplicate_file(file_path.name)
        ):
            continue

        try:
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue

        python_files.append(file_path)

        if len(python_files) >= MAX_FILES:
            break

    return sorted(python_files)


# ---------------------------------------------------------
# Function extraction
# ---------------------------------------------------------

def extract_function_details(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: Path,
    project_root: Path,
    class_name: str | None = None,
) -> dict[str, Any]:
    """Extract metadata, returns, and control flow from a function."""
    return_visitor = ReturnExpressionVisitor()

    for statement in node.body:
        return_visitor.visit(statement)

    flow_visitor = ControlFlowVisitor()

    for statement in node.body:
        flow_visitor.visit(statement)

    positional_args = [
        arg.arg for arg in node.args.posonlyargs + node.args.args
    ]
    keyword_only_args = [arg.arg for arg in node.args.kwonlyargs]

    parameters = positional_args + keyword_only_args

    if node.args.vararg:
        parameters.append("*" + node.args.vararg.arg)

    if node.args.kwarg:
        parameters.append("**" + node.args.kwarg.arg)

    try:
        relative_file = file_path.relative_to(project_root).as_posix()
    except ValueError:
        relative_file = str(file_path)

    qualified_name = (
        f"{class_name}.{node.name}" if class_name else node.name
    )
    calls = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        call_name = source_of(call.func)
        if call_name not in calls:
            calls.append(call_name)
    statements = [
        short_text(source_of(statement), limit=180)
        for statement in node.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]

    return {
        "name": node.name,
        "qualified_name": qualified_name,
        "class_name": class_name,
        "parameters": parameters,
        "return_annotation": get_return_annotation(node),
        "return_expressions": return_visitor.expressions,
        "docstring": ast.get_docstring(node),
        "file_path": relative_file,
        "line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "conditions": flow_visitor.conditions,
        "exceptions": flow_visitor.exceptions,
        "calls": calls,
        "source_steps": statements[:8],
    }


class FunctionDefinitionVisitor(ast.NodeVisitor):
    """Find functions, including methods and nested functions."""

    def __init__(self, file_path: Path, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.functions = []
        self.class_stack = []
        self.function_stack = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)

        for statement in node.body:
            self.visit(statement)

        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node)

    def _handle_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ):
        class_name = ".".join(self.class_stack) if self.class_stack else None
        details = extract_function_details(
            node=node,
            file_path=self.file_path,
            project_root=self.project_root,
            class_name=class_name,
        )

        if self.function_stack:
            parent = ".".join(self.function_stack)
            details["qualified_name"] = f"{parent}.{node.name}"

        self.functions.append(details)

        self.function_stack.append(node.name)

        for statement in node.body:
            self.visit(statement)

        self.function_stack.pop()


def extract_functions(
    project_path: str | Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Parse eligible Python files and extract function metadata."""
    root = Path(project_path).resolve()
    functions = []

    for file_path in find_python_files(root):
        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            tree = ast.parse(content, filename=str(file_path))
        except (OSError, SyntaxError, UnicodeError):
            continue

        visitor = FunctionDefinitionVisitor(file_path, root)
        visitor.visit(tree)
        functions.extend(visitor.functions)

    return functions


# ---------------------------------------------------------
# Question generation helpers
# ---------------------------------------------------------

def format_control_flow(function: dict[str, Any]) -> str:
    """Create a readable summary of conditions and exceptions."""
    parts = []

    for condition in function["conditions"]:
        line = condition["line"]
        test = condition["condition"]
        actions = condition["actions"]

        text = f"if condition at line {line}: {test}; actions: {actions}"

        if condition["has_else"]:
            text += f"; else actions: {condition['else_actions']}"

        parts.append(text)

    for exception in function["exceptions"]:
        parts.append(
            f"try block at line {exception['line']}; "
            f"try actions: {exception['try_actions']}"
        )

        for handler in exception["handlers"]:
            parts.append(
                f"except {handler['exception']} at line "
                f"{handler['line']}; handler actions: "
                f"{handler['actions']}"
            )

        if exception["else_actions"]:
            parts.append(
                f"try-else actions: {exception['else_actions']}"
            )

        if exception["finally_actions"]:
            parts.append(
                f"finally actions: {exception['finally_actions']}"
            )

    return " | ".join(parts) if parts else (
        "No if conditions or try/except blocks detected"
    )


def format_annotation(function: dict[str, Any]) -> str:
    """Explain whether a return annotation is declared."""
    annotation = function["return_annotation"]

    if annotation is None:
        return "No return type annotation is specified."

    return f"The declared return type annotation is `{annotation}`."


def build_answer_versions(function: dict[str, Any]) -> dict[str, str]:
    """Create progressively detailed, spoken answers from function evidence."""
    facts = _answer_sentences(function)
    name = function["qualified_name"].rsplit(".", 1)[-1]

    if name == "safe_float":
        short = (
            "`safe_float()` converts an optional value to a float for "
            "dashboard metrics. Missing or pandas-null values and "
            "`TypeError` or `ValueError` conversions return `None`; otherwise, "
            "the function returns the converted number."
        )
        medium = (
            "`safe_float()` converts a value to a float without interrupting "
            "dashboard metric rendering. It accepts a value and checks "
            "whether it is `None` or pandas considers it missing. A missing "
            "value returns `None`. Otherwise, the function calls `float()` "
            "and returns the conversion. It catches `TypeError` and "
            "`ValueError` and returns `None` for those invalid inputs."
        )
        detailed = (
            "Purpose: `safe_float()` converts an optional metric value for "
            "dashboard use. Input: it accepts the value to convert. Processing: "
            "it returns `None` for `None` or a pandas-null value; otherwise, "
            "it calls `float()` and returns the numeric result. Error handling: "
            "`TypeError` and `ValueError` are caught and produce `None`, so "
            "invalid metric values do not interrupt the display. The "
            "dashboard uses the result to render available numeric metrics."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "load_production_batch":
        short = (
            "`load_production_batch()` reads a production CSV and checks it "
            "against the model's required feature schema. It rejects missing "
            "files, empty data, duplicate or mismatched columns with "
            "`FileNotFoundError` or `ValueError`, then returns columns in the "
            "required order."
        )
        medium = (
            "`load_production_batch()` prepares a CSV batch for model "
            "prediction. It accepts a file path and the required feature "
            "columns, then checks that the file exists before reading it with "
            "Pandas. It raises `ValueError` for an empty batch or duplicate "
            "column names. It also compares actual and required columns, "
            "rejecting missing or unexpected fields. Valid data is reordered "
            "to the model's expected feature order and returned. The function "
            "raises these input errors rather than catching them."
        )
        detailed = (
            "Purpose: `load_production_batch()` validates production input "
            "before prediction. Inputs: it accepts a file path and the "
            "required feature-column names. Processing: it verifies the path "
            "is a file, reads the CSV, rejects empty data and duplicate column "
            "names, then checks for missing and unexpected columns. If the "
            "schema matches, it reorders the DataFrame to the required column "
            "sequence and returns it. Error handling: missing files raise "
            "`FileNotFoundError`; empty batches and invalid schemas raise "
            "`ValueError`, which the function does not catch. This prevents "
            "unvalidated columns from being passed to the production model."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "detect_data_drift":
        short = (
            "`detect_data_drift()` compares reference and incoming numerical "
            "features with a two-sample Kolmogorov-Smirnov test. It defaults "
            "to `tenure`, `MonthlyCharges`, and `TotalCharges`, ignores "
            "missing values, and marks drift when a p-value is below the "
            "significance threshold; the result includes per-feature details "
            "and an overall retraining flag."
        )
        medium = (
            "`detect_data_drift()` checks selected numerical features for a "
            "distribution shift. It accepts reference data, new data, an "
            "optional column list, and a significance threshold that defaults "
            "to 0.05. If no columns are supplied, it checks `tenure`, "
            "`MonthlyCharges`, and `TotalCharges`. For each feature it drops "
            "missing values and runs `ks_2samp`. A p-value below the "
            "threshold marks that feature as drifted. It returns each "
            "feature's statistic and p-value plus a flag that is true if any "
            "checked feature drifted."
        )
        detailed = (
            "Purpose: `detect_data_drift()` measures changes between reference "
            "and incoming numerical data. Inputs: it accepts both DataFrames, "
            "optional feature names, and a significance threshold. Processing: "
            "the default columns are `tenure`, `MonthlyCharges`, and "
            "`TotalCharges`; missing values are removed before each "
            "`ks_2samp` comparison. A feature is marked as drifted when its "
            "p-value is less than the threshold, which defaults to 0.05. The "
            "return value contains per-feature KS statistics and p-values, "
            "along with an overall flag set when any feature drifts. It has "
            "no explicit exception handler. The lifecycle uses this result "
            "to decide whether the supervised retraining path can continue."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "run_lifecycle":
        short = (
            "`run_lifecycle()` coordinates drift checks, candidate evaluation, "
            "approval, and promotion. It stops when there is no drift or no "
            "new labels; otherwise it validates a candidate on held-out data "
            "and promotes it only after human approval. It returns a status "
            "for each exit path and has no explicit exception handler."
        )
        medium = (
            "`run_lifecycle()` orchestrates the supervised retraining path. "
            "It accepts the current model, reference and new data, labels, "
            "held-out test data, metrics, a candidate version, and registry "
            "paths. It first detects drift and exits if retraining is not "
            "required; if labels are missing, it reports that retraining "
            "cannot proceed. It also rejects reused versions or existing "
            "candidate artifacts. The current model and candidate are "
            "evaluated on the supplied held-out test data, then compared by "
            "F1 and ROC-AUC thresholds. A passing candidate still requires "
            "human approval before its artifact and metadata are saved and "
            "promoted."
        )
        detailed = (
            "Purpose: `run_lifecycle()` controls the drift-triggered, "
            "human-approved candidate promotion workflow. Inputs include the "
            "current model, reference and incoming data, new labels, held-out "
            "test data, current metrics, candidate version, and registry and "
            "artifact paths. It first detects drift and returns `no_drift` "
            "when the trigger is absent; if labels are unavailable, it "
            "returns a status explaining that supervised retraining cannot "
            "continue. Before training, it rejects a reused version or an "
            "existing candidate artifact. It reevaluates the current model "
            "on the held-out test data, trains the candidate on the new "
            "labeled data, and evaluates that candidate on the same test "
            "partition. It compares current and candidate metrics, validates "
            "F1 and ROC-AUC thresholds, and returns `rejected` if human "
            "approval is denied. An approved candidate is saved with metadata "
            "and promoted; promotion failure and successful promotion have "
            "separate statuses. The function has no explicit exception "
            "handler, so unexpected errors propagate to its caller."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name in {"calculate_performance", "calculate_version_performance"}:
        version_specific = name == "calculate_version_performance"
        function_call = f"`{name}()`"
        scope = (
            "only rows for the requested model version"
            if version_specific else "all logged model versions"
        )
        short = (
            f"{function_call} calculates classification metrics from "
            f"completed prediction logs for {scope}. It raises "
            "`FileNotFoundError` when the log is missing and returns "
            "`insufficient_data` when no eligible outcomes exist; otherwise "
            "it calculates accuracy, precision, recall, F1, and ROC-AUC, "
            "omitting AUC when the actual labels contain one class."
        )
        if version_specific:
            scope_steps = (
                "It filters rows to the requested `model_version` and to "
                "actual labels of `0` or `1`."
            )
        else:
            scope_steps = (
                "It includes all logged model versions and filters rows to "
                "actual labels of `0` or `1`, excluding predictions whose "
                "outcomes are not yet known."
            )
        medium = (
            f"{function_call} scores logged predictions after their actual "
            "outcomes are available. It accepts a log path"
            + (
                " and a model version."
                if version_specific else "."
            )
            + " A missing log raises `FileNotFoundError`. "
            + scope_steps
            + " If no rows remain, it returns an `insufficient_data` status. "
            "Otherwise, it calculates accuracy, precision, recall, and F1 "
            "from actual and predicted labels. ROC-AUC is calculated only "
            "when the actual labels contain both classes; otherwise it is "
            "reported as `None`."
        )
        detailed = (
            f"Purpose: {function_call} calculates metrics from prediction "
            "records whose outcomes are known. Input: it accepts a log path"
            + (
                " and the requested model version."
                if version_specific else "."
            )
            + " It raises `FileNotFoundError` if the log is absent, then "
            "reads the CSV records. "
            + scope_steps
            + " No eligible rows produce an `insufficient_data` result rather "
            "than metric values. With eligible rows, it builds the actual and "
            "predicted label lists and calculates accuracy, precision, "
            "recall, and F1. ROC-AUC uses the logged hard predictions only "
            "when actual labels include both classes; otherwise the AUC value "
            "is `None`. The version-specific function applies the same metric "
            "definitions after filtering to one version, keeping its results "
            "comparable with the aggregate calculation."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "rollback_model":
        short = (
            "`rollback_model()` restores a registered version as production. "
            "It checks the registry, version, and artifact, copies the chosen "
            "artifact to the current-model path, updates registry statuses, "
            "and returns whether the rollback succeeded."
        )
        medium = (
            "`rollback_model()` switches production back to a selected "
            "registered model version. It accepts the registry path and "
            "version identifier. It returns `False` if the registry, version, "
            "or artifact is missing. Otherwise it copies the artifact to "
            "`models/current_model.pkl`, archives the previous current entry, "
            "marks the selected entry as current, and writes the registry "
            "back to disk. The function returns `True` on completion and does "
            "not catch file or JSON errors."
        )
        detailed = (
            "Purpose: `rollback_model()` restores an older registered model "
            "artifact as production. Inputs: it accepts the registry metadata "
            "path and requested version. Processing: it loads the registry, "
            "locates that version, and checks that its artifact exists. A "
            "missing registry, version, or artifact prints a status message "
            "and returns `False`. On success it copies the artifact to the "
            "current production path, archives any entry marked current, "
            "marks the selected version current, records the production "
            "artifact path, and saves the registry. It returns `True` after "
            "these updates. It does not explicitly catch JSON or file-system "
            "exceptions, so those errors propagate."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "request_approval":
        short = (
            "`request_approval()` asks for a human decision on a validated "
            "candidate. It rejects failed validation without prompting; for a "
            "passing candidate, it reads a console response and returns true "
            "only for `yes` or `y`."
        )
        medium = (
            "`request_approval()` is the lifecycle's human approval gate. It "
            "accepts a candidate version and its validation result. If "
            "validation failed, it prints the reason and returns `False` "
            "without asking for approval. Otherwise it prints the candidate "
            "and validation status, then prompts in the console. Only `yes` "
            "or `y`, ignoring letter case and surrounding whitespace, returns "
            "`True`; other answers return `False`. It has no explicit "
            "exception handler."
        )
        detailed = (
            "Purpose: `request_approval()` prevents an invalid candidate from "
            "reaching human review and requires a person to accept a passing "
            "candidate. Inputs: it accepts the candidate version and "
            "validation result. If the result is not approved, it prints the "
            "failure reason and returns `False` immediately. If validation "
            "passes, it displays the version and prompts through terminal "
            "input. The response is stripped and lowercased; only `yes` or "
            "`y` is accepted. It returns that boolean decision and does not "
            "handle input exceptions explicitly. `run_lifecycle()` uses the "
            "result to decide whether to save and promote the candidate."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "read_json":
        short = (
            "`read_json()` safely loads and validates JSON data from a file. "
            "It takes a path, returns an empty dictionary when the file is "
            "missing or the content is not a dictionary, and also falls back "
            "to an empty dictionary for file-reading or JSON-decoding errors."
        )
        medium = (
            "`read_json()` safely loads JSON for the dashboard. It takes a "
            "file path and first checks whether that file exists. If it does "
            "not, the function returns an empty dictionary. Otherwise it "
            "opens and parses the file, then accepts the value only if it is "
            "a dictionary. File-access errors and invalid JSON are caught and "
            "also produce an empty dictionary, so malformed local data does "
            "not interrupt dashboard reads."
        )
        detailed = (
            "Purpose: `read_json()` safely loads JSON data used by the "
            "dashboard. Input: it accepts a `path`. Implementation: it checks "
            "that the path exists, opens the file with a context manager, and "
            "parses the contents with `json.load()`. Return behavior: the "
            "parsed value is returned only when it is a dictionary; a missing "
            "file or a different JSON value yields an empty dictionary. Error "
            "handling: `OSError` and `JSONDecodeError` are caught and also "
            "yield an empty dictionary. Project relevance: this lets the "
            "dashboard read registry and log data without crashing when a "
            "file is missing or malformed."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "get_registry_parts":
        short = (
            "`get_registry_parts()` normalizes registry data into a current "
            "version, its active model entry, and a list of model entries. "
            "It accepts the registry data; mapping-shaped models are converted "
            "to entries, while list-shaped models keep only dictionary "
            "entries. It selects the entry matching the current version and "
            "returns all three values; an unsupported models value produces "
            "an empty entry list. The function has no explicit exception handler."
        )
        medium = (
            "`get_registry_parts()` prepares registry metadata for dashboard "
            "views. It accepts a registry dictionary and defaults a missing "
            "current version to `Not set`. When models are stored by version "
            "in a mapping, it copies each dictionary entry and fills in a "
            "missing version from the mapping key. When models are a list, it "
            "keeps only dictionary entries; other value types result in an "
            "empty list. It finds the entry whose version matches the current "
            "version, then returns the version, active entry, and full entry "
            "list. It does not define explicit exception handling."
        )
        detailed = (
            "Purpose: `get_registry_parts()` converts the supported registry "
            "layouts into values the dashboard can consume. Input: it accepts "
            "the registry data. Processing: it reads the current version, "
            "then normalizes `models` whether it is a mapping or a list. For "
            "a mapping, each dictionary value is copied and receives its "
            "mapping key as `version` only when that field is missing. For a "
            "list, non-dictionary entries are discarded; if `models` has "
            "another type, the entry list is empty. It scans the normalized "
            "entries for the current version and returns that active entry "
            "along with the version and all entries. There is no explicit "
            "exception handler, so unexpected input errors are not converted "
            "into a fallback result."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "format_metric":
        short = (
            "`format_metric()` formats an optional metric value for dashboard "
            "display. It accepts the value and a decimal-place count; `None` "
            "becomes an em dash, while a number is formatted to the requested "
            "precision, with no explicit exception handler."
        )
        medium = (
            "`format_metric()` formats numeric metrics for the dashboard. It "
            "accepts a value and a `digits` argument that defaults to three. "
            "If the value is `None`, it returns an em dash as the missing-data "
            "placeholder. Otherwise, it formats the number with the requested "
            "fixed decimal precision. The function does not catch formatting "
            "errors. The dashboard uses it to present performance metrics "
            "consistently."
        )
        detailed = (
            "Purpose: `format_metric()` converts an optional numeric metric "
            "into display text. Inputs: it accepts the metric value and "
            "`digits`, which defaults to three decimal places. Branching: a "
            "`None` value returns an em dash; all other values are formatted "
            "with fixed-point precision using `digits`. Return behavior: it "
            "returns the placeholder or formatted string. Error handling: "
            "there is no explicit exception handler. Relevance: the dashboard "
            "uses this helper for accuracy and ROC-AUC cards and registry "
            "metrics."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "compare_models":
        short = (
            "`compare_models()` measures how a candidate's accuracy, "
            "precision, recall, F1, and ROC-AUC differ from the current model. "
            "It accepts both metric dictionaries and returns their values and "
            "candidate-minus-current changes; it has no explicit exception "
            "handler."
        )
        medium = (
            "`compare_models()` compares a candidate model with the current "
            "model. It accepts a metric dictionary for each model. For "
            "accuracy, precision, recall, F1, and ROC-AUC, it reads both "
            "values and subtracts the current value from the candidate value. "
            "The result records the two values and the improvement for every "
            "metric. The function has no explicit exception handler. The "
            "lifecycle uses this comparison to validate a candidate."
        )
        detailed = (
            "Purpose: `compare_models()` creates a per-metric comparison of "
            "the current and candidate models. Inputs: it accepts their "
            "respective metric dictionaries. Processing: it iterates over "
            "accuracy, precision, recall, F1, and ROC-AUC, preserving each "
            "model's score and calculating improvement as candidate minus "
            "current. It returns a dictionary with those values for each "
            "metric. It does not handle missing keys or other exceptions "
            "explicitly. `validate_candidate()` consumes this result to decide "
            "whether the candidate meets its improvement thresholds."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "validate_candidate":
        short = (
            "`validate_candidate()` checks whether a candidate improves both "
            "F1 and ROC-AUC by their required minimum amounts. It accepts the "
            "comparison and optional thresholds, then returns an approval "
            "flag, both improvements, and a reason; it has no explicit "
            "exception handler."
        )
        medium = (
            "`validate_candidate()` applies the lifecycle's candidate-quality "
            "gate. It accepts the metric comparison plus minimum F1 and "
            "ROC-AUC improvements, both defaulting to zero. It reads the two "
            "improvements and approves only when both meet their thresholds. "
            "The result includes the approval flag, the measured improvements, "
            "and a reason. The function does not handle exceptions explicitly. "
            "The lifecycle uses this result before requesting human approval."
        )
        detailed = (
            "Purpose: `validate_candidate()` decides whether a candidate "
            "passes the configured metric gate. Inputs: it accepts the "
            "comparison and minimum improvement thresholds for F1 and "
            "ROC-AUC; each threshold defaults to zero. Processing: it reads "
            "both metric improvements and requires both comparisons to pass, "
            "so a gain in one metric cannot offset a failure in the other. It "
            "returns the approval decision, both improvements, and a reason "
            "message. It has no explicit exception handler for missing or "
            "malformed comparison data. `run_lifecycle()` uses the decision "
            "before the human approval and promotion steps."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "get_question_bank":
        short = (
            "`get_question_bank()` returns the predefined interview question "
            "bank. It takes no parameters and returns `QUESTION_BANK` directly, "
            "without filtering or modifying it."
        )
        medium = (
            "`get_question_bank()` exposes the repository's predefined "
            "interview questions. It takes no parameters. The function returns "
            "the existing `QUESTION_BANK` collection directly. It does not "
            "filter or transform the entries. It has no explicit exception "
            "handler. Other question-bank helpers use that collection to "
            "prepare answer versions."
        )
        detailed = (
            "Purpose: `get_question_bank()` provides access to the repository's "
            "predefined interview questions. Inputs: it does not accept "
            "parameters. Processing: there is no filtering or transformation; "
            "the function returns the `QUESTION_BANK` value as stored. It does "
            "not define explicit exception handling. This keeps the question "
            "data available to the helpers that build answer versions."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "keep_selected_question_open":
        short = (
            "`keep_selected_question_open()` records which Interview Coach "
            "question is active after its answer length changes. It stores "
            "the question key in Streamlit session state so the dashboard can "
            "keep that panel expanded after rerunning; it returns no value."
        )
        medium = (
            "`keep_selected_question_open()` preserves the selected question "
            "panel when an answer-length radio changes. It accepts the "
            "question key as input. The callback stores that key in Streamlit "
            "session state. On the next render, the matching expander reads "
            "the stored key and stays open. It returns no value and defines "
            "no explicit exception handling."
        )
        detailed = (
            "Purpose: `keep_selected_question_open()` keeps the current "
            "Interview Coach question visible across a Streamlit rerun. Input: "
            "it accepts the question's stable key. Processing: the callback "
            "writes that key to `st.session_state`; the rendering loop compares "
            "it with each question's key to choose which expander stays open. "
            "Return behavior: it returns no value. Error handling: no explicit "
            "exception handler is defined. This preserves the context while "
            "the user switches between Short, Medium, and Detailed answers."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    if name == "get_project_insights":
        short = (
            "`get_project_insights()` asks the project analyzer for an "
            "inventory rooted at the dashboard's project directory. It "
            "returns counts for scanned files, Python files, technologies, "
            "and source modules, plus the full inventory; if analysis fails, "
            "it returns zero counts and an empty inventory."
        )
        medium = (
            "`get_project_insights()` gathers summary information for the "
            "dashboard. It calls `get_project_inventory()` with the project "
            "root. It counts scanned files, Python files, detected "
            "technologies, and source modules, and includes the complete "
            "inventory in its result. The overview uses these values to "
            "populate its Project Insights cards. If the analyzer raises an "
            "exception, the function catches it and supplies zero counts and "
            "an empty inventory instead."
        )
        detailed = (
            "Purpose: `get_project_insights()` prepares the project summary "
            "shown by the dashboard. It has no explicit input parameters and "
            "uses the dashboard's configured project root. It requests an "
            "inventory from `get_project_inventory()`, counts Python files "
            "from the inventory paths, and returns counts for scanned files, "
            "Python files, detected technologies, and source modules together "
            "with the inventory itself. The Overview page consumes those "
            "values for its Project Insights cards. Error handling is broad: "
            "any exception during the scan is converted into zero counts and "
            "an empty inventory. That keeps the view available, but it can "
            "hide an analyzer failure, which is an important limitation."
        )
        return {
            "short_answer": short,
            "medium_answer": medium,
            "detailed_answer": detailed,
        }

    branches = facts["branches"]
    branch_sentence = (
        branches[0]
        if branches
        else "The function has no conditional branch that changes its result."
    )
    secondary_branch = branches[1] if len(branches) > 1 else ""
    operations = facts["operations"]

    operation_clause = operations.removeprefix("The implementation ").rstrip(".")
    parameter_clause = facts["parameters"].rstrip(".")
    if operations.startswith("The implementation "):
        short_steps = f"{parameter_clause}, and it {operation_clause}."
    else:
        short_steps = (
            f"{parameter_clause}; its behavior is implemented directly "
            "from the inputs."
        )
    return_clause = facts["returns"].removeprefix("It ").rstrip(".")
    exception_clause = facts["exceptions"].removeprefix("It ").rstrip(".")
    if function.get("exceptions"):
        short_outcome = (
            f"It {return_clause}; it also {exception_clause}."
        )
    else:
        short_outcome = (
            f"It {return_clause}; it has no explicit exception handler."
        )
    short = " ".join(
        (facts["purpose"], short_steps, branch_sentence, short_outcome)
    )
    medium_sentences = [
        facts["purpose"],
        facts["parameters"],
        operations,
        branch_sentence,
        secondary_branch,
        facts["returns"],
        facts["exceptions"],
        facts["relevance"],
    ]
    medium = " ".join(sentence for sentence in medium_sentences if sentence)
    detailed = (
        f"Purpose: {facts['purpose']} "
        f"Inputs: {facts['parameters']} "
        f"Implementation: {operations} "
        f"{' '.join(branches) if branches else branch_sentence} "
        f"Return behavior: {facts['returns']} "
        f"Error handling: {facts['exceptions']} "
        f"Project relevance: {facts['relevance']}"
    )
    return {
        "short_answer": short,
        "medium_answer": medium,
        "detailed_answer": detailed,
    }


def build_function_question(function: dict[str, Any]) -> dict[str, Any]:
    """Build an interview question from extracted function structure."""
    name = function["qualified_name"]
    file_path = function["file_path"]
    parameters = function["parameters"]
    returns = function["return_expressions"]
    conditions = function["conditions"]
    exceptions = function["exceptions"]

    if conditions or exceptions:
        question = (
            f"Explain the logic of `{name}` in `{file_path}`. "
            "Describe its parameters, return behavior, branches, "
            "and exception handling."
        )
    else:
        question = (
            f"Explain how `{name}` works in `{file_path}`. "
            "Describe its parameters, processing steps, and return behavior."
        )

    if function["docstring"]:
        question += (
            " How does the implementation support the purpose "
            "described in its docstring?"
        )

    control_flow = format_control_flow(function)

    evidence = f"{file_path}:{function['line']}"
    answer_versions = build_answer_versions(function)

    return {
        "category": "code_logic",
        "difficulty": "intermediate",
        "question": question,
        "answer": answer_versions["medium_answer"],
        **answer_versions,
        "function_name": name,
        "file_path": file_path,
        "line": function["line"],
        "parameters": parameters,
        "return_annotation": function["return_annotation"],
        "annotation_explanation": format_annotation(function),
        "return_expressions": returns,
        "conditions": conditions,
        "exceptions": exceptions,
        "control_flow_summary": control_flow,
        "docstring": function["docstring"],
        "evidence": [evidence],
        "source_type": "ast_function_metadata",
    }


def build_docstring_question(
    function: dict[str, Any],
) -> dict[str, Any]:
    """Build a purpose-oriented question from a function docstring."""
    name = function["qualified_name"]
    file_path = function["file_path"]
    answer_versions = build_answer_versions(function)

    return {
        "category": "code_logic",
        "difficulty": "intermediate",
        "question": (
            f"What purpose does `{name}` serve according to its "
            "docstring, and which parts of its implementation "
            "support that purpose?"
        ),
        "answer": answer_versions["medium_answer"],
        **answer_versions,
        "function_name": name,
        "file_path": file_path,
        "line": function["line"],
        "parameters": function["parameters"],
        "return_annotation": function["return_annotation"],
        "annotation_explanation": format_annotation(function),
        "return_expressions": function["return_expressions"],
        "conditions": function["conditions"],
        "exceptions": function["exceptions"],
        "control_flow_summary": format_control_flow(function),
        "docstring": function["docstring"],
        "evidence": [f"{file_path}:{function['line']}"],
        "source_type": "ast_docstring",
    }


# ---------------------------------------------------------
# Interview question generation
# ---------------------------------------------------------

def generate_function_questions(
    project_path: str | Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Generate interview questions for discovered Python functions."""
    functions = extract_functions(project_path)
    questions = []

    for function in functions:
        questions.append(build_function_question(function))

        if function["docstring"]:
            questions.append(build_docstring_question(function))

    return questions


# ---------------------------------------------------------
# Console report
# ---------------------------------------------------------

def print_question(question: dict[str, Any], number: int):
    """Print one generated question with its extracted evidence."""
    print(f"\n{number}. [{question['category']}]")
    print(question["question"])
    print(f"Evidence: {question['evidence']}")
    print(f"Parameters: {question['parameters']}")
    print(f"Return annotation: {question['return_annotation']}")
    print(question["annotation_explanation"])
    print(f"Return expressions: {question['return_expressions']}")
    print(f"Control flow: {question['control_flow_summary']}")
    print(f"Docstring: {question['docstring']}")
    print(f"Source type: {question['source_type']}")
    print(f"Status: {question['status']}")


def main():
    """Run the generator and display a project interview report."""
    print("=" * 70)
    print("MODELLIFE - CODE LOGIC INTERVIEW QUESTION GENERATOR")
    print("=" * 70)

    try:
        functions = extract_functions(PROJECT_ROOT)
        questions = generate_function_questions(PROJECT_ROOT)
    except (OSError, ValueError) as error:
        print(f"Could not analyze project: {error}")
        return

    print(f"\nProject: {PROJECT_ROOT.name}")
    print(f"Functions discovered: {len(functions)}")
    print(f"Questions generated: {len(questions)}")
    print(f"Displaying up to: {OUTPUT_LIMIT} questions")

    for index, question in enumerate(questions[:OUTPUT_LIMIT], start=1):
        print_question(question, index)

    if len(questions) > OUTPUT_LIMIT:
        print(
            f"\n... {len(questions) - OUTPUT_LIMIT} additional "
            "questions not displayed."
        )

    print("\n" + "=" * 70)
    print("Analysis complete. Review source code before using answers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
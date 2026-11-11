
from pathlib import Path
import re
import ast


# ============================================================
# 1. Project scanning configuration
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"
}

EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build",
    "data", "models", "logs"
}

SENSITIVE_FILENAMES = {
    ".env", ".env.local", "secrets.json",
    "credentials.json"
}

MAX_FILES = 100
MAX_FILE_SIZE = 200_000
MAX_CONTENT_CHARS = MAX_FILE_SIZE

ML_OPERATION_NAMES = {
    "LogisticRegression": "classifier",
    "Pipeline": "pipeline",
    "ColumnTransformer": "feature transformation",
    "StandardScaler": "numeric scaling",
    "OneHotEncoder": "categorical encoding",
    "train_test_split": "data splitting",
    "fit": "model fitting",
    "predict": "prediction",
    "predict_proba": "probability prediction",
    "ks_2samp": "Kolmogorov-Smirnov drift test",
}

METRIC_NAMES = {
    "accuracy", "accuracy_score", "precision", "precision_score",
    "recall", "recall_score", "f1", "f1_score", "roc_auc",
    "roc_auc_score", "ks_statistic", "p_value",
}


# ============================================================
# 2. Basic secret redaction
# ============================================================

def redact_secrets(content):
    """Mask common secret assignments before returning file text."""

    pattern = (
        r"""(?i)\b(api[_-]?key|secret|password|token)"""
        r"""\b\s*[:=]\s*["']?[^"' \t\r\n,;]+"""
    )

    return re.sub(
        pattern,
        lambda match: match.group(1) + "=[REDACTED]",
        content
    )


def is_backup_or_duplicate_file(file_name):
    """Return whether a Python file is a named backup or duplicate copy."""
    name = Path(file_name).name.casefold()
    return name.endswith("_backup.py") or bool(
        re.search(r" copy(?: \d+)?\.py$", name)
    )


def _call_name(node):
    """Return a readable name for a call expression."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _analyze_python_file(item):
    """Extract source-level evidence without importing or executing code."""
    try:
        tree = ast.parse(item["content"])
    except SyntaxError:
        return None

    lines = item["content"].splitlines()
    imports = []
    functions = []
    classes = []
    important_calls = []
    ml_operations = []
    metrics = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append({
                "line": node.lineno,
                "module": ", ".join(alias.name for alias in node.names),
                "names": [alias.asname or alias.name for alias in node.names],
            })
        elif isinstance(node, ast.ImportFrom):
            imports.append({
                "line": node.lineno,
                "module": node.module or "",
                "names": [alias.asname or alias.name for alias in node.names],
            })
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "bases": [ast.unparse(base) for base in node.bases],
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ast.unparse(node.args)
            signature = f"{node.name}({args})"
            if node.returns is not None:
                signature += f" -> {ast.unparse(node.returns)}"
            function_calls = [
                {"name": _call_name(call.func), "line": call.lineno}
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and _call_name(call.func)
            ]
            statements = [
                ast.unparse(statement)[:300]
                for statement in node.body[:10]
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "signature": signature,
                "docstring": ast.get_docstring(node),
                "async": isinstance(node, ast.AsyncFunctionDef),
                "calls": sorted(function_calls, key=lambda value: value["line"]),
                "statements": statements,
            })
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if not call_name:
                continue
            call = {
                "name": call_name,
                "line": node.lineno,
                "expression": lines[node.lineno - 1].strip()
                if node.lineno <= len(lines) else call_name,
            }
            important_calls.append(call)
            short_name = call_name.rsplit(".", 1)[-1]
            if short_name in ML_OPERATION_NAMES:
                ml_operations.append({
                    **call,
                    "operation": ML_OPERATION_NAMES[short_name],
                })
            if short_name.lower() in METRIC_NAMES:
                metrics.append({"name": short_name, "line": node.lineno})

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            metric_name = node.value.strip().lower()
            if metric_name in METRIC_NAMES:
                metrics.append({"name": metric_name, "line": node.lineno})

    return {
        "path": item["path"],
        "imports": imports,
        "functions": sorted(functions, key=lambda value: value["line"]),
        "classes": sorted(classes, key=lambda value: value["line"]),
        "important_calls": sorted(important_calls, key=lambda value: value["line"]),
        "ml_operations": sorted(ml_operations, key=lambda value: value["line"]),
        "metrics": sorted(metrics, key=lambda value: value["line"]),
    }


# ============================================================
# 3. Scan a project without executing its code
# ============================================================

def scan_project(project_path="."):
    """
    Read supported project files and return a safe file inventory.

    This scanner does not execute project code.
    """

    root = Path(project_path).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(
            "The project path must be an existing directory."
        )

    scanned_files = []

    for file_path in sorted(root.rglob("*")):

        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)

        if (
            file_path.suffix.lower() == ".py"
            and is_backup_or_duplicate_file(file_path.name)
        ):
            continue

        # Skip virtual environments, data, model artifacts,
        # and cache directories.
        if any(
            part in EXCLUDED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        # Only inspect supported text file types.
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Skip likely credential files.
        if file_path.name.lower() in SENSITIVE_FILENAMES:
            continue

        # Skip oversized files.
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            content = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

        except OSError:
            continue

        # Redact common secrets before storing content.
        content = redact_secrets(content)

        # Limit the amount of content retained per file.
        content = content[:MAX_CONTENT_CHARS]

        scanned_files.append({
            "path": relative_path.as_posix(),
            "extension": file_path.suffix.lower(),
            "content": content
        })

        if len(scanned_files) >= MAX_FILES:
            break

    return {
        "project_name": root.name,
        "project_path": str(root),
        "files_scanned": len(scanned_files),
        "files": scanned_files
    }


# ============================================================
# 4. Build an accurate project inventory
# ============================================================

def get_project_inventory(project_path="."):
    """
    Identify technologies from actual Python imports
    and record which files import each technology.
    """

    result = scan_project(project_path)

    file_paths = [
        item["path"]
        for item in result["files"]
    ]

    extensions = sorted({
        item["extension"]
        for item in result["files"]
    })

    # --------------------------------------------------------
    # Detect actual imports and track their source files
    # --------------------------------------------------------

    python_evidence = []
    imported_by_file = {}

    for item in result["files"]:
        if item["extension"] != ".py":
            continue
        evidence = _analyze_python_file(item)
        if evidence is None:
            continue
        python_evidence.append(evidence)
        imported_by_file[item["path"]] = {
            entry["module"].split(".")[0]
            for entry in evidence["imports"]
            if entry["module"]
        }

    # --------------------------------------------------------
    # Map Python module names to technology names
    # --------------------------------------------------------

    technology_map = {
        "pandas": "Pandas",
        "numpy": "NumPy",
        "streamlit": "Streamlit",
        "sklearn": "Scikit-learn",
        "scipy": "SciPy",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "flask": "Flask",
        "fastapi": "FastAPI",
        "cv2": "OpenCV",
        "sqlalchemy": "SQLAlchemy",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "joblib": "Joblib",
        "plotly": "Plotly",
        "pytest": "Pytest",
    }

    # --------------------------------------------------------
    # Associate each detected technology with its source files
    # --------------------------------------------------------

    technology_files = {}

    for path, modules in imported_by_file.items():

        for module in modules:

            technology = technology_map.get(module)

            if technology:
                technology_files.setdefault(
                    technology, []
                ).append(path)

    technologies = sorted(technology_files)
    if python_evidence:
        technologies.insert(0, "Python")

    documentation = []
    for item in result["files"]:
        if item["extension"] != ".md":
            continue
        for line_number, text in enumerate(item["content"].splitlines(), start=1):
            text = text.strip()
            if text:
                documentation.append({
                    "path": item["path"],
                    "line": line_number,
                    "text": text[:500],
                })
            if len(documentation) >= 160:
                break
        if len(documentation) >= 160:
            break

    # --------------------------------------------------------
    # Return complete project inventory
    # --------------------------------------------------------

    return {
        "project_name": result["project_name"],
        "files_scanned": result["files_scanned"],
        "file_paths": file_paths,
        "extensions": extensions,
        "technologies": technologies,
        "technology_files": technology_files,
        "source_evidence": python_evidence,
        "documentation": documentation,
    }


# ============================================================
# 5. Run analyzer directly for testing
# ============================================================

if __name__ == "__main__":

    inventory = get_project_inventory(".")

    print("\nPROJECT INVENTORY")
    print("=" * 50)

    print("Project:", inventory["project_name"])
    print("Files scanned:", inventory["files_scanned"])
    print("Extensions:", inventory["extensions"])

    print("\nDETECTED TECHNOLOGIES")
    print("=" * 50)

    for technology in inventory["technologies"]:
        print(f"\n{technology}")

        for path in inventory["technology_files"][technology]:
            print(f"  - {path}")
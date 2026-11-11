from pathlib import Path
import re

from src.code_question_generator import generate_function_questions
from src.interview_generator import generate_interview_questions
from src.project_analyzer import get_project_inventory


def _questions(result):
    return [
        item
        for category_items in result["questions"].values()
        for item in category_items
    ]


def _sentence_count(answer):
    prose = re.sub(r"`[^`]*`", "", answer)
    return len(re.findall(r"[.!?](?:\s|$)", prose))


def _assert_evidence_resolves(evidence, project_root):
    for reference in evidence:
        path, line = reference.rsplit(":", 1)
        source = project_root / path
        assert source.is_file(), reference
        assert 1 <= int(line) <= len(source.read_text(encoding="utf-8").splitlines())


def _assert_progressive_answers(item):
    short = item["short_answer"]
    medium = item["medium_answer"]
    detailed = item["detailed_answer"]
    assert 2 <= _sentence_count(short) <= 4
    assert 5 <= _sentence_count(medium) <= 8
    assert len(short) < len(medium) < len(detailed)
    for answer in (short, medium, detailed):
        assert not re.search(r"\[\s*['\"]", answer)
        assert not re.search(r"\{\s*['\"]", answer)
        assert "src\\" not in answer
        assert "src/" not in answer
        assert "I would explain" not in answer
        assert "mention the algorithm" not in answer


def test_analyzer_extracts_source_evidence_with_line_numbers(tmp_path):
    project = tmp_path / "sample"
    source_dir = project / "src"
    source_dir.mkdir(parents=True)
    source = source_dir / "model.py"
    source.write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score\n"
        "\n"
        "class ChurnModel:\n"
        "    def fit(self, x, y):\n"
        "        self.model = LogisticRegression()\n"
        "        self.model.fit(x, y)\n"
        "        return self.model\n"
        "\n"
        "def score(y_true, y_pred):\n"
        "    return accuracy_score(y_true, y_pred)\n",
        encoding="utf-8",
    )

    inventory = get_project_inventory(project)
    evidence = next(
        item for item in inventory["source_evidence"]
        if item["path"].replace("\\", "/") == "src/model.py"
    )

    assert evidence["imports"][0]["module"] == "sklearn.linear_model"
    assert evidence["classes"][0]["name"] == "ChurnModel"
    assert evidence["classes"][0]["line"] == 4
    fit = next(
        function for function in evidence["functions"]
        if function["name"] == "fit"
    )
    assert fit["line"] == 5
    assert any(call["name"] == "self.model.fit" for call in fit["calls"])
    assert any(
        operation["operation"] == "classifier"
        for operation in evidence["ml_operations"]
    )
    assert any(metric["name"] == "accuracy_score" for metric in evidence["metrics"])
    assert any(call["line"] == 11 for call in evidence["important_calls"])


def test_project_answers_are_natural_scoped_and_evidence_backed():
    project_root = Path(__file__).resolve().parents[1]
    result = generate_interview_questions(project_root)
    questions = _questions(result)
    technical_questions = [item for item in questions if item["category"] != "Behavioral"]

    assert technical_questions
    for item in technical_questions:
        _assert_progressive_answers(item)
        _assert_evidence_resolves(item["evidence"], project_root)

    pandas = next(
        item for item in questions
        if item["question"] == "How is Pandas used in this project?"
    )
    for key in ("short_answer", "medium_answer", "detailed_answer"):
        answer = pandas[key]
        assert "Pandas" in answer
        assert "read" in answer and "CSV" in answer
        assert "TotalCharges" in answer
        assert "fill" in answer
        assert "customerID" in answer
        assert "Churn" in answer
        for unrelated in ("joblib", "matplotlib", "scikit-learn", "model_registry"):
            assert unrelated not in answer.casefold()

    performance = next(
        item for item in questions
        if item["question"].startswith(
            "How does performance monitoring calculate"
        )
    )
    for key in ("short_answer", "medium_answer", "detailed_answer"):
        assert "calculate_performance" in performance[key]
        assert "calculate_version_performance" in performance[key]
        assert "FileNotFoundError" in performance[key]
        assert "insufficient_data" in performance[key]
        assert "roc_auc_score" in performance[key]
    assert any(
        "performance_monitor.py" in reference
        for reference in performance["evidence"]
    )

    tests_answer = next(
        item for item in questions
        if item["question"] == "How do you test that the project works correctly?"
    )
    for key in ("short_answer", "medium_answer", "detailed_answer"):
        assert "python -m pytest -q" in tests_answer[key]
        assert "drift" in tests_answer[key].casefold()
        assert "performance" in tests_answer[key].casefold()
    assert "18 passing tests" in tests_answer["detailed_answer"]
    assert "test_performance_monitor" not in tests_answer["medium_answer"]

    technology_stack = next(
        item for item in questions
        if item["question"] == "What technologies did you use, and why?"
    )
    _assert_progressive_answers(technology_stack)
    for key in ("short_answer", "medium_answer", "detailed_answer"):
        answer = technology_stack[key]
        for technology in ("Pandas", "scikit-learn", "SciPy", "Joblib",
                           "Streamlit", "Plotly"):
            assert technology in answer
        assert "source detects" not in answer.casefold()
        assert "source supports" not in answer.casefold()
        assert "source evidence" not in answer.casefold()

    workflow = next(
        item for item in questions
        if item["question"] == "Explain the machine learning workflow."
    )
    _assert_progressive_answers(workflow)
    _assert_evidence_resolves(workflow["evidence"], project_root)
    assert "Logistic Regression" in workflow["short_answer"]
    assert "drift" in workflow["medium_answer"].casefold()
    for detail in ("human approval", "registry", "rollback"):
        assert detail in workflow["detailed_answer"].casefold()

    behavioral = next(item for item in questions if item["category"] == "Behavioral")
    assert "Customizable draft" in behavioral["short_answer"]
    assert "confirm" in behavioral["medium_answer"].casefold()
    assert "who made the change" in behavioral["short_answer"]
    assert any("performance_monitor.py" in ref for ref in behavioral["evidence"])


def test_each_detected_technology_has_scoped_progressive_answers():
    project_root = Path(__file__).resolve().parents[1]
    result = generate_interview_questions(project_root)
    technology_questions = result["questions"]["technology"]

    assert technology_questions
    for item in technology_questions:
        technology = item["question"].removeprefix("How is ").split(
            " used in this project?", maxsplit=1
        )[0]
        _assert_progressive_answers(item)
        _assert_evidence_resolves(item["evidence"], project_root)
        for key in ("short_answer", "medium_answer", "detailed_answer"):
            answer = item[key]
            assert technology.casefold() in answer.casefold()
            assert "['" not in answer and "{'" not in answer
            assert "tool-selection rationale" not in answer
            assert "cited files and line numbers" not in answer

    for item in technology_questions:
        if "Joblib" in item["question"]:
            for key in ("short_answer", "medium_answer", "detailed_answer"):
                answer = item[key].casefold()
                assert "joblib" in answer
                assert "pandas" not in answer
                assert "matplotlib" not in answer
                assert "scikit-learn" not in answer

    pandas = next(
        item for item in technology_questions
        if item["question"] == "How is Pandas used in this project?"
    )
    assert "read_csv()" in pandas["detailed_answer"]
    assert "to_numeric()" in pandas["detailed_answer"]
    assert "fillna()" in pandas["detailed_answer"]


def test_code_logic_questions_include_all_three_answer_lengths(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "service.py").write_text(
        "def summarize(values):\n"
        '    \"\"\"Return the total and average for the input values.\"\"\"\n'
        "    total = sum(values)\n"
        "    if not values:\n"
        "        return {\"total\": total, \"average\": 0}\n"
        "    return {\"total\": total, \"average\": total / len(values)}\n",
        encoding="utf-8",
    )

    questions = generate_function_questions(tmp_path)
    function_question = next(
        item for item in questions
        if item["function_name"] == "summarize"
    )

    assert function_question["evidence"] == ["src/service.py:1"]
    _assert_progressive_answers(function_question)
    assert "`summarize()`" in function_question["short_answer"]
    assert "sum()" in function_question["medium_answer"]
    assert "input collection is empty" in function_question["detailed_answer"]


def test_dashboard_code_answers_explain_behavior_without_ast_dumps():
    project_root = Path(__file__).resolve().parents[1]
    questions = generate_function_questions(project_root)

    read_json = next(item for item in questions if item["function_name"] == "read_json")
    insights = next(
        item for item in questions
        if item["function_name"] == "get_project_insights"
    )

    for item in (read_json, insights):
        _assert_progressive_answers(item)
        _assert_evidence_resolves(item["evidence"], project_root)
        for answer in (
            item["short_answer"],
            item["medium_answer"],
            item["detailed_answer"],
        ):
            assert "Parameters" not in answer
            assert "return expressions" not in answer.casefold()
            assert "control-flow summary" not in answer.casefold()
            assert "['" not in answer
            assert "{'" not in answer

    for answer in (
        read_json["short_answer"],
        read_json["medium_answer"],
        read_json["detailed_answer"],
    ):
        assert "path" in answer
        assert "dictionary" in answer
    assert "JSON-decoding errors" in read_json["short_answer"]
    assert "JSONDecodeError" in read_json["detailed_answer"]
    assert "missing or malformed" in read_json["detailed_answer"]

    for answer in (
        insights["short_answer"],
        insights["medium_answer"],
        insights["detailed_answer"],
    ):
        assert "inventory" in answer
        assert "Python files" in answer
        assert "technologies" in answer
    assert "catches it" in insights["medium_answer"]
    assert "hide an analyzer failure" in insights["detailed_answer"]


def test_all_generated_code_answers_are_explanatory_and_evidence_backed():
    project_root = Path(__file__).resolve().parents[1]
    questions = generate_function_questions(project_root)
    by_name = {}

    assert questions
    for item in questions:
        by_name.setdefault(item["function_name"], item)
        _assert_progressive_answers(item)
        _assert_evidence_resolves(item["evidence"], project_root)
        for key in ("short_answer", "medium_answer", "detailed_answer"):
            answer = item[key]
            assert "follows the steps visible in the source body" not in answer
            assert "performs the guarded operation" not in answer
            assert "a required condition is met" not in answer
            assert not re.search(r"(?m)^\s*[\[{]", answer)

    registry = by_name["get_registry_parts"]["detailed_answer"]
    assert "mapping" in registry
    assert "list" in registry
    assert "mapping key" in registry
    assert '{"current_version"' not in registry

    lifecycle = by_name["run_lifecycle"]["detailed_answer"]
    for detail in ("drift", "new labels", "held-out test data", "human approval", "promoted"):
        assert detail in lifecycle.casefold()

    aggregate = by_name["calculate_performance"]["detailed_answer"]
    per_version = by_name["calculate_version_performance"]["detailed_answer"]
    assert "all logged model versions" in aggregate
    assert "requested model version" in per_version
    for answer in (aggregate, per_version):
        assert "insufficient_data" in answer
        assert "FileNotFoundError" in answer
        assert "ROC-AUC" in answer

    rollback = by_name["rollback_model"]["detailed_answer"]
    assert "copies the artifact" in rollback
    assert "marks the selected version current" in rollback

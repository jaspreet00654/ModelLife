

from src.project_analyzer import get_project_inventory
from src.interview_question_bank import get_question_bank



def _build_answer_versions(item, project_name, technologies):
    """Create three deterministic answer drafts without executing project code."""
    question = str(item.get("question", "")).strip()
    question_lower = question.lower()
    guidance = str(item.get("answer", "")).strip()
    source_type = str(item.get("source_type", "general"))
    tech_list = ", ".join(technologies) if technologies else "the project's detected Python tools"

    # Preserve carefully authored versions from the curated question bank.
    short = item.get("short_answer")
    medium = item.get("medium_answer")
    detailed = item.get("detailed_answer")

    if not short or not medium or not detailed:
        if "tell me about your project" in question_lower:
            short = short or (
                f"My project is {project_name}. It is a Python-based project that "
                f"uses {tech_list}. Its main goal is to organize the project's "
                "machine-learning workflow and supporting tools."
            )
            medium = medium or (
                f"My project is {project_name}. It brings together data handling, "
                "machine-learning development, and supporting project tools. "
                f"The repository scan detected {tech_list}. Its key workflow includes "
                "preparing data, training and evaluating a model, monitoring for data "
                "drift, and managing candidate model versions. I would demonstrate "
                "each feature using its source file and explain which parts I personally built."
            )
            detailed = detailed or (
                f"My project is {project_name}. It is intended to make the machine-learning "
                "lifecycle easier to manage rather than treating model training as a one-time task. "
                "The project includes data preparation, model training and evaluation, statistical "
                "drift checks, candidate-model evaluation, human approval, and model-version "
                "management, along with project analysis and interview-question generation. "
                f"The repository analyzer detected these technologies: {tech_list}. "
                "I would walk through the relevant modules to show how the components connect, "
                "and I would distinguish completed functionality from planned improvements. "
                "I would also describe my individual contribution accurately and support claims "
                "with code and test results."
            )
        elif "technologies did you use" in question_lower or "technical stack" in question_lower:
            short = short or (
                f"The repository scan detected {tech_list}. Pandas supports tabular data work, "
                "Scikit-learn supports preprocessing and machine learning, SciPy provides "
                "statistical tools, and Joblib can persist model artifacts. Matplotlib and "
                "Seaborn support visual analysis where used."
            )
            medium = medium or (
                f"ModelLife is primarily written in Python, and the repository scan detected "
                f"{tech_list}. Pandas is useful for loading and transforming tabular data; "
                "Scikit-learn provides preprocessing, model training, and evaluation tools; "
                "SciPy provides statistical tests used in drift analysis; and Joblib supports "
                "saving and loading model artifacts. Matplotlib and Seaborn are visualization "
                "libraries for exploratory analysis. I would point to the matching source files "
                "to show exactly where each library is used."
            )
            detailed = detailed or (
                f"I chose the project's tools according to the tasks they support. The repository "
                f"scan detected {tech_list}. Pandas is suited to working with tabular datasets. "
                "Scikit-learn provides the preprocessing and model-building components, including "
                "classification algorithms and evaluation metrics. SciPy provides statistical "
                "functions that can be used for drift tests, while Joblib supports persistence of "
                "Python model artifacts. Matplotlib and Seaborn can produce plots for exploratory "
                "data analysis and monitoring. Detection by the analyzer does not prove that every "
                "library is used in the core application, so I would verify each claim in the cited "
                "files and avoid claiming a library is central if it appears only in a utility or test."
            )
        elif "complete modellife workflow" in question_lower or "end-to-end" in question_lower or "end to end" in question_lower:
            short = short or (
                "ModelLife loads and preprocesses churn data, trains and evaluates a baseline model, "
                "checks incoming data for drift, and can train a candidate when labelled data is available. "
                "The candidate is evaluated and a human approval step controls promotion."
            )
            medium = medium or (
                "The workflow starts with loading and preprocessing the churn dataset. A Logistic Regression "
                "baseline is trained and evaluated using multiple metrics. Statistical tests check whether "
                "incoming data has drifted from reference data. If drift is detected and suitable labels are "
                "available, the system can train and evaluate a candidate model. A human approval step then "
                "controls whether the candidate is promoted or rejected, with model versions and predictions "
                "recorded by the supporting components."
            )
            detailed = detailed or (
                "ModelLife starts by loading the Telco Customer Churn data and preparing the features and target. "
                "Numerical and categorical features are processed before a Logistic Regression baseline is trained. "
                "The model is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. A drift-detection "
                "component uses statistical tests to compare incoming data with reference data. Drift indicates a "
                "distribution change; it does not by itself prove that predictive performance has fallen. If drift "
                "is detected and suitable labelled data is available, the system can train a candidate model and "
                "compare it with the current model. Human approval controls promotion or rejection, while registry, "
                "prediction logging, and monitoring components support version tracking and operational review. "
                "The exact behavior of rollback and promotion should be verified against the current implementation."
            )
        else:
            # Conservative, reusable answer framing for generated questions.
            topic = question.rstrip("?").strip()
            base = guidance or "I would explain the relevant implementation and verify it against the source code."
            short = short or (
                f"For {project_name}, I would answer '{topic}' by focusing on the relevant component, "
                "what it does, and why it is needed. I would support the explanation with the cited source files."
            )
            medium = medium or (
                f"In {project_name}, I would explain '{topic}' by first describing the problem or responsibility, "
                "then the approach used, and finally how it fits into the wider project. "
                f"The current question guidance is: {base} "
                "I would use the listed source evidence to confirm implementation details rather than assume them."
            )
            detailed = detailed or (
                f"To answer '{topic}', I would start with the purpose of the component and the problem it solves. "
                "Next, I would describe the relevant inputs, the main processing steps, and the output or decision "
                "it produces. I would then explain how it connects to the rest of the project and mention important "
                "trade-offs, limitations, or failure cases where relevant. "
                f"The current generator's guidance is: {base} "
                "Because this answer is generated from templates rather than a language model that deeply analyzes "
                "the source, I would verify specific behavior against the cited files before presenting it as fact."
            )

    item["short_answer"] = short
    item["medium_answer"] = medium
    item["detailed_answer"] = detailed
    return item

def generate_interview_questions(project_path="."):
    """
    Generate interview questions using:
    1. Project inventory
    2. Detected technologies
    3. Curated interview question bank

    This version uses static templates and project inventory.
    It does not execute project code or use an LLM.
    """

    # --------------------------------------------------
    # 1. ANALYZE PROJECT
    # --------------------------------------------------

    inventory = get_project_inventory(project_path)

    project_name = inventory["project_name"]
    technologies = inventory["technologies"]
    file_paths = inventory["file_paths"]
    technology_files = inventory["technology_files"]

    # --------------------------------------------------
    # 2. CREATE QUESTION CATEGORIES
    # --------------------------------------------------

    questions = {
        "common": [],
        "project_specific": [],
        "technology": [],
        "backend": [],
        "ai_agents": [],
        "machine_learning": [],
        "rag": [],
        "system_design": [],
        "security": [],
        "debugging": [],
        "behavioral": []
    }

    # --------------------------------------------------
    # 3. COMMON INTERVIEW QUESTIONS
    # --------------------------------------------------

    questions["common"].append({
        "question": "Tell me about your project.",
        "answer": (
            f"My project is {project_name}. "
            f"It contains {inventory['files_scanned']} scanned files "
            f"and uses technologies including "
            f"{', '.join(technologies) if technologies else 'not yet identified'}. "
            "I would explain its main problem, architecture, features, "
            "and my contribution."
        ),
        "evidence": file_paths[:5],
        "difficulty": "Beginner",
        "why_asked": (
            "To understand your project, your contribution, "
            "and how clearly you can explain your work."
        ),
        "follow_ups": [
            "What problem does your project solve?",
            "Who would use this application?",
            "What are its main features?",
            "What was your individual contribution?"
        ],
        "source_type": "project"
    })

    questions["common"].append({
        "question": "What technologies did you use, and why?",
        "answer": (
            "The project analyzer detected these technologies: "
            f"{', '.join(technologies) if technologies else 'None detected'}. "
            "Explain the purpose of each technology and how it is used "
            "in the implementation."
        ),
        "evidence": [
            path for path in file_paths
            if path.endswith(".py")
        ][:10],
        "difficulty": "Beginner",
        "why_asked": (
            "To check whether you understand your technical stack "
            "rather than just listing technologies."
        ),
        "follow_ups": [
            "Why did you choose these technologies?",
            "What alternatives did you consider?",
            "Which technology was the most challenging to use?"
        ],
        "source_type": "project"
    })

    # --------------------------------------------------
    # 4. PROJECT STRUCTURE QUESTIONS
    # --------------------------------------------------

    source_files = [
        path for path in file_paths
        if path.startswith("src\\") or path.startswith("src/")
    ]

    if source_files:
        questions["project_specific"].append({
            "question": "How have you organized your project?",
            "answer": (
                f"The project has a src directory containing "
                f"{len(source_files)} scanned source files. "
                "Explain the responsibility of each module "
                "and how they work together."
            ),
            "evidence": source_files,
            "difficulty": "Intermediate",
            "why_asked": (
                "To assess your understanding of modular design "
                "and the project's architecture."
            ),
            "follow_ups": [
                "Why did you separate the code into modules?",
                "How do the modules communicate?",
                "Which module controls the main workflow?",
                "How would you make the architecture easier to maintain?"
            ],
            "source_type": "project"
        })

    # --------------------------------------------------
    # 5. TECHNOLOGY QUESTION TEMPLATES
    # --------------------------------------------------

    technology_questions = {
        "Streamlit": {
            "question": "How does Streamlit help in your project?",
            "answer": (
                "Streamlit is detected in the project. Explain which "
                "dashboard features you implemented and how users "
                "interact with them."
            ),
            "difficulty": "Beginner",
            "why_asked": "To check your understanding of the dashboard.",
            "follow_ups": [
                "How does Streamlit display dynamic data?",
                "How would you connect the dashboard to a backend API?",
                "How would you manage user sessions?",
                "What are the limitations of Streamlit?"
            ]
        },

        "Pandas": {
            "question": "Where and why do you use Pandas?",
            "answer": (
                "Pandas is detected. Explain how your project loads, "
                "cleans, transforms, or analyzes tabular data."
            ),
            "difficulty": "Beginner",
            "why_asked": "To test your practical data-handling knowledge.",
            "follow_ups": [
                "How do you handle missing values?",
                "How do you detect duplicate records?",
                "What is the difference between a Series and DataFrame?",
                "Why use Pandas instead of ordinary Python lists?"
            ]
        },

        "Scikit-learn": {
            "question": "How is Scikit-learn used in your project?",
            "answer": (
                "Scikit-learn is detected. Explain the models, "
                "preprocessing, training, and evaluation steps "
                "that your implementation uses."
            ),
            "difficulty": "Intermediate",
            "why_asked": "To assess your practical machine learning knowledge.",
            "follow_ups": [
                "Why did you use Logistic Regression?",
                "What is the purpose of a preprocessing pipeline?",
                "How do you prevent data leakage?",
                "What is the difference between fit and transform?"
            ]
        },

        "SciPy": {
            "question": "Why is SciPy included in your project?",
            "answer": (
                "SciPy is detected. Explain which statistical "
                "or scientific functions your code uses."
            ),
            "difficulty": "Intermediate",
            "why_asked": "To assess your understanding of statistical tools.",
            "follow_ups": [
                "What is the Kolmogorov-Smirnov test?",
                "What does a p-value represent?",
                "What does the KS statistic measure?",
                "What are the limitations of this drift test?"
            ]
        },

        "Joblib": {
            "question": "Why might Joblib be useful in an ML project?",
            "answer": (
                "Joblib is detected. Explain how your project "
                "uses it to save or load model artifacts."
            ),
            "difficulty": "Beginner",
            "why_asked": "To check your understanding of model persistence.",
            "follow_ups": [
                "What is model serialization?",
                "Why do we save a trained model?",
                "What is a model artifact?",
                "Why should you avoid loading untrusted serialized files?"
            ]
        },

        "Matplotlib": {
            "question": "How do you use Matplotlib?",
            "answer": (
                "Matplotlib is detected. Describe the plots "
                "and visualizations implemented in your project."
            ),
            "difficulty": "Beginner",
            "why_asked": "To assess your visualization knowledge.",
            "follow_ups": [
                "How do you create a bar chart?",
                "How can you add labels and a title?",
                "How can plots help with model monitoring?"
            ]
        },

        "Seaborn": {
            "question": "What is Seaborn used for in your project?",
            "answer": (
                "Seaborn is detected. Explain any statistical "
                "visualizations or exploratory data analysis "
                "that use it."
            ),
            "difficulty": "Beginner",
            "why_asked": "To check your knowledge of statistical visualization.",
            "follow_ups": [
                "What is a heatmap?",
                "How can you visualize correlations?",
                "How is Seaborn different from Matplotlib?"
            ]
        }
    }

    # --------------------------------------------------
    # 6. GENERATE TECHNOLOGY QUESTIONS WITH MATCHED EVIDENCE
    # --------------------------------------------------

    for technology in technologies:
        if technology in technology_questions:
            item = technology_questions[technology]

            questions["technology"].append({
                "question": item["question"],
                "answer": item["answer"],
                "evidence": technology_files.get(technology, []),
                "difficulty": item["difficulty"],
                "why_asked": item["why_asked"],
                "follow_ups": item["follow_ups"],
                "source_type": "project"
            })

    # --------------------------------------------------
    # 7. ADD THE CURATED INTERVIEW QUESTION BANK
    # --------------------------------------------------

    category_mapping = {
        "Project Logic": "project_specific",
        "Backend": "backend",
        "Machine Learning": "machine_learning",
        "AI Agents": "ai_agents",
        "RAG and Knowledge Retrieval": "rag",
        "System Design": "system_design",
        "Security": "security",
        "Debugging": "debugging",
        "Behavioral and Project Defense": "behavioral"
    }

    for item in get_question_bank():
        bank_category = item.get("category", "common")
        category = category_mapping.get(bank_category, "common")

        # Add category if it is not already present
        if category not in questions:
            questions[category] = []

        # Add the question to its category
        questions[category].append({
            "question": item["question"],
            "answer": item["answer"],
            "short_answer": item.get("short_answer"),
            "medium_answer": item.get("medium_answer"),
            "detailed_answer": item.get("detailed_answer"),
            "evidence": item.get("evidence", []),
            "difficulty": item.get("difficulty", "Intermediate"),
            "why_asked": item.get(
                "why_asked",
                "To assess your technical understanding."
            ),
            "follow_ups": item.get("follow_ups", []),
            "source_type": item.get("source_type", "general")
        })

    # --------------------------------------------------
    # 8. ADD SHORT, MEDIUM, AND DETAILED ANSWERS
    # --------------------------------------------------

    for category_items in questions.values():
        for question_item in category_items:
            _build_answer_versions(question_item, project_name, technologies)

    # --------------------------------------------------
    # 9. CALCULATE QUESTION COUNT
    # --------------------------------------------------

    total_questions = sum(
        len(group)
        for group in questions.values()
    )

    # --------------------------------------------------
    # 10. RETURN THE COMPLETE QUESTION SET
    # --------------------------------------------------

    return {
        "project_name": project_name,
        "total_questions": total_questions,
        "categories": list(questions.keys()),
        "questions": questions
    }


# --------------------------------------------------
# 11. RUN GENERATOR DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":
    result = generate_interview_questions(".")

    print("\n" + "=" * 60)
    print(f"INTERVIEW QUESTIONS FOR: {result['project_name']}")
    print(f"TOTAL QUESTIONS: {result['total_questions']}")
    print("=" * 60)

    for category, items in result["questions"].items():
        if not items:
            continue

        print(f"\n{'=' * 60}")
        print(f"--- {category.upper()} ---")
        print("=" * 60)

        for index, item in enumerate(items, start=1):
            print(f"\nQ{index}: {item['question']}")
            print(f"Difficulty: {item['difficulty']}")
            print(f"Why asked: {item['why_asked']}")
            print(f"\nShort answer:\n{item['short_answer']}")
            print(f"\nMedium answer:\n{item['medium_answer']}")
            print(f"\nLong / detailed answer:\n{item['detailed_answer']}")
            print(f"\nEvidence files: {item['evidence']}")

            if item["follow_ups"]:
                print("\nPossible follow-up questions:")
                for follow_up in item["follow_ups"]:
                    print(f"  - {follow_up}")

            print(f"\nSource type: {item['source_type']}")
            


from src.project_analyzer import get_project_inventory
from src.interview_question_bank import get_question_bank



def _build_answer_versions(item, project_name, technologies):
    """Build three answer drafts with topic-specific, honest guidance.

    These are deterministic templates, not LLM-generated or source-verified
    explanations. Existing authored answer versions are preserved.
    """
    question = str(item.get("question", "")).strip()
    q = question.lower()
    guidance = str(item.get("answer", "")).strip()
    tech_list = ", ".join(technologies) if technologies else "the project's detected tools"

    short = item.get("short_answer")
    medium = item.get("medium_answer")
    detailed = item.get("detailed_answer")

    # Topic-specific answers for dynamically generated questions.
    topic_answers = None

    if "scikit-learn" in q or "sklearn" in q:
        topic_answers = (
            "I use Scikit-learn for the machine-learning workflow in ModelLife. It provides preprocessing tools, Logistic Regression, and evaluation metrics.",
            "In ModelLife, Scikit-learn supports preprocessing and model development. Numerical features are scaled and categorical features are encoded through the preprocessing pipeline. Logistic Regression is used for churn prediction, and the model is evaluated with accuracy, precision, recall, F1-score, and ROC-AUC. These metrics help compare a candidate model with the current model. I would show the relevant preprocessing, training, and evaluation files to demonstrate the exact implementation.",
            "Scikit-learn provides several core building blocks in ModelLife. The preprocessing stage prepares numerical and categorical columns, including scaling numerical features and encoding categorical features. The training stage uses Logistic Regression for the churn classification task. Evaluation uses accuracy, precision, recall, F1-score, and ROC-AUC, giving a broader view than accuracy alone. During candidate evaluation, these metrics help assess whether a new model is suitable for promotion. The precise comparison rules and thresholds should be confirmed in the current evaluation code, and I would use the cited source files to demonstrate the implementation rather than claiming features that are not present."
        )
    elif "pandas" in q or "where and why do you use" in q:
        topic_answers = (
            "I use Pandas to load and manipulate tabular data, including the customer churn dataset used by ModelLife.",
            "Pandas helps ModelLife work with the Telco Customer Churn dataset. It can load the CSV into a DataFrame, inspect columns, handle data types, and prepare data for preprocessing and training. For example, TotalCharges is converted to a numeric type before the model pipeline uses the features. I would point to the data-loading and preprocessing code for the exact operations.",
            "Pandas is useful because the churn dataset is tabular and contains named columns. ModelLife uses DataFrames to inspect and prepare this data before model training. The preparation includes converting TotalCharges to numeric, handling invalid or missing values as implemented, separating the target from input features, and preparing the data for the Scikit-learn preprocessing pipeline. Pandas makes these transformations easier to read and test than manually processing rows. I would verify the exact cleaning steps in the current data-loader and preprocessing modules before describing every step as part of the implementation."
        )
    elif "scipy" in q or "kolmogorov" in q or "drift test" in q:
        topic_answers = (
            "SciPy provides statistical functions used in ModelLife's data-drift checks, including the KS test where implemented.",
            "ModelLife uses statistical testing to compare incoming data with reference data. SciPy provides statistical functions, including the Kolmogorov–Smirnov test used for numeric feature comparisons. A small p-value can indicate that the compared distributions differ, based on the selected significance threshold. Drift signals a data-distribution change; it does not automatically prove that model accuracy has fallen.",
            "SciPy supplies statistical tools for ModelLife's drift-detection workflow. For selected numeric features, the Kolmogorov–Smirnov test compares two samples and calculates a statistic and p-value. The implementation can use the p-value with a chosen significance threshold to flag a potential distribution shift. This is a monitoring signal rather than direct proof of performance degradation, because data drift and concept drift are different. The test's assumptions, selected features, reference batch, and threshold should be explained from the drift-detection source code."
        )
    elif "joblib" in q or "model persistence" in q or "save or load model" in q:
        topic_answers = (
            "Joblib is useful for saving and loading trained Python model artifacts so they can be reused without retraining every time.",
            "ModelLife uses model artifacts to manage trained models across the lifecycle. Joblib can serialize a fitted model or pipeline to a file and load it later for prediction or version management. This makes the saved artifact reusable between runs. I would check the registry and service code to identify the exact files and versions persisted by this project.",
            "Joblib supports persistence of Python objects commonly used in machine learning, including fitted Scikit-learn models and preprocessing pipelines. Saving an artifact allows the application to reload a trained model instead of retraining it whenever the program starts. In ModelLife, artifact paths and model-version metadata should stay consistent so that the registry refers to the correct saved model. Serialized files should only be loaded from trusted sources because deserialization can be unsafe. The exact save/load flow should be demonstrated using the model-registry and prediction-service implementation."
        )
    elif "matplotlib" in q or "seaborn" in q or "visualization" in q or "heatmap" in q:
        topic_answers = (
            "Matplotlib and Seaborn are Python visualization libraries that can help explore data and present statistical patterns where the project uses them.",
            "Matplotlib provides low-level plotting tools, while Seaborn offers higher-level statistical plots built on Matplotlib. They can help visualize distributions, relationships, or correlations during exploratory data analysis. The project analyzer detecting a library does not prove it is used in the core workflow, so I would point to the specific plotting code before naming a chart as an implemented feature.",
            "Matplotlib is a flexible plotting library for creating figures and customizing axes, labels, legends, and titles. Seaborn builds on Matplotlib and provides convenient statistical visualizations such as distribution plots and heatmaps. In a machine-learning project, these tools can make data quality, feature relationships, or evaluation results easier to inspect. However, a dependency scan alone only shows that a library is present in scanned files; the actual source code should be checked to confirm which plots are implemented and how they support ModelLife."
        )
    elif "streamlit" in q or "dashboard" in q:
        topic_answers = (
            "Streamlit provides the interactive web dashboard for viewing ModelLife information and using its interview and monitoring pages.",
            "The ModelLife dashboard uses Streamlit to create a Python-based user interface. Streamlit widgets allow users to choose pages or filters, generate interview questions, inspect evidence, and view available monitoring information. The dashboard runs Python code in response to user interactions. I would demonstrate the actual pages and controls in dashboard.py and avoid claiming any feature that is not connected yet.",
            "Streamlit lets a Python application provide an interactive browser-based interface without requiring a separate frontend framework. In ModelLife, dashboard.py defines the navigation, widgets, question-generation workflow, and views for lifecycle or prediction information. User selections and button clicks cause Streamlit to rerun the relevant script, while session state can preserve values during the session. Some advanced capabilities, such as a live LLM assistant or spoken mock interviews, should be described as unavailable unless they have been implemented and tested."
        )
    elif "organized your project" in q or "project structure" in q or "modular design" in q:
        topic_answers = (
            "ModelLife is organized into modules so that data loading, preprocessing, training, drift detection, evaluation, approval, and model registry responsibilities are separated.",
            "The project uses a src directory for reusable modules and keeps the dashboard and main entry point at the project root. Separate modules handle tasks such as data loading, preprocessing, training, drift detection, candidate evaluation, approval, and model registry operations. This separation makes the workflow easier to test and maintain. I would explain the actual call flow by tracing the lifecycle entry point into the relevant modules.",
            "ModelLife follows a modular Python structure. The data-loading and preprocessing modules prepare input data; training and evaluation modules build and assess models; drift detection identifies distribution changes; retraining prepares a candidate; approval controls the decision; and the registry tracks model artifacts and versions. Supporting modules handle prediction service, logging, monitoring, and project-aware interview generation. These modules are easier to test independently, while the lifecycle entry point coordinates the relevant stages. I would use the current repository and tests to confirm the exact dependencies and sequence."
        )
    elif "multiple evaluation metrics" in q or "evaluation metrics" in q or "precision" in q or "recall" in q:
        topic_answers = (
            "ModelLife compares accuracy, precision, recall, F1-score, and ROC-AUC because each metric describes a different aspect of classification performance.",
            "Accuracy measures overall correctness, precision measures how many predicted positive cases are truly positive, and recall measures how many actual positive cases are found. F1-score balances precision and recall, while ROC-AUC measures how well the model ranks positive cases above negative cases when computed from scores. Looking at several metrics helps assess trade-offs, especially for churn data where the classes may be imbalanced.",
            "ModelLife uses multiple classification metrics because a single number can hide important weaknesses. Accuracy is the fraction of correct predictions. Precision focuses on the reliability of positive predictions, while recall focuses on finding actual positive cases. F1-score combines precision and recall, and ROC-AUC evaluates ranking ability across thresholds when calculated from prediction scores or probabilities. In churn prediction, higher recall may identify more customers likely to leave, but can also increase false positives. The best trade-off depends on the business objective, and the project's actual candidate acceptance rules must be verified in model_evaluation.py."
        )
    elif "rest api" in q or "api for modellife predictions" in q:
        topic_answers = (
            "I would expose a POST /predict endpoint that validates input, calls the prediction service, and returns a prediction and model version. This is a proposed API unless implemented.",
            "A possible REST API would accept feature data through POST /predict, validate required fields and types, call the prediction service, and return a structured prediction response. Invalid requests should receive an appropriate 4xx response, and unexpected failures should be handled safely. Authentication and request logging would be important for deployment. I would label this as a proposed extension unless the endpoint exists in the codebase.",
            "I would design a POST /predict endpoint with request-schema validation, feature checks, and clear response models. The endpoint would call the existing prediction service rather than duplicate model logic, and its response could include the prediction and model version. Invalid input should return a client error, while unexpected server errors should not expose sensitive implementation details. Tests should cover valid inputs, missing fields, incorrect types, unavailable artifacts, and service failures. This is a proposed design and must not be described as implemented unless the repository contains that endpoint."
        )
    elif "rag" in q or "retrieval-augmented" in q:
        topic_answers = (
            "RAG retrieves relevant project content before a language model answers, helping ground responses in source files and documentation.",
            "A RAG extension could parse project files into meaningful chunks, index them for search, retrieve the most relevant passages for a question, and pass those passages to a language model. The answer could cite file paths so the developer can verify it. This is a proposed capability unless a retrieval and generation pipeline is present in the code.",
            "A RAG system would first parse source files and documentation, split them into chunks such as functions or sections, and index them using keyword search or embeddings. For each question, it would retrieve the most relevant chunks and provide them as context to a language model. The prompt should require evidence-grounded answers, distinguish facts from assumptions, and say when evidence is missing. File and line citations would help users verify claims. ModelLife's project scanner can supply source content, but RAG should be described as a future extension unless the retrieval and model-generation components are implemented."
        )
    elif "ai agent" in q or "agent help" in q or "tool use" in q:
        topic_answers = (
            "A project-aware AI agent could inspect relevant files, retrieve context, and use approved tools to help explain a codebase. A full agent is a proposed extension unless implemented.",
            "An agent could combine project scanning, retrieval, and a language model to answer questions about code. It might use tools such as file search or approved test execution, while restricting destructive actions and arbitrary code execution. ModelLife currently has project-analysis and question-generation components; those alone do not prove that a full autonomous agent is implemented.",
            "A project-aware agent could receive a question, retrieve relevant source files, decide which approved tools are needed, and formulate an evidence-grounded answer. Tool permissions should be narrow, operations should be logged, and arbitrary code execution should be blocked or isolated. The current ModelLife project analyzer scans files and the interview generator creates template-based questions; a full LLM agent with planning and tool execution would be an extension unless corresponding components are present."
        )
    elif "security risks" in q or "source code" in q and "upload" in q:
        topic_answers = (
            "Source-code analysis risks include exposed secrets, oversized files, unsafe paths, and accidental code execution. The analyzer should validate inputs and remain read-only by default.",
            "A source scanner may encounter secrets, malformed files, oversized inputs, or path traversal attempts. File-type and size limits, safe path handling, secret redaction, and read-only processing reduce risk. Pattern-based redaction is imperfect and should not be treated as a complete security boundary.",
            "Security controls for source-code analysis include strict file and size limits, path validation within the project root, avoiding unsafe symbolic links, and never executing uploaded code by default. Likely secrets can be redacted, but regex-based detection can miss credentials and produce false positives. Access controls and isolated processing are important if the tool is exposed to other users. ModelLife has basic exclusions and pattern-based redaction; these are useful safeguards but do not make the scanner fully secure on their own."
        )
    elif "saved model" in q or "loading a saved model" in q or "model artifact" in q:
        topic_answers = (
            "I would check the artifact path, file existence, registry metadata, serialization compatibility, and preprocessing requirements, then reproduce the failure with a small test.",
            "I would first read the traceback and check whether the model file exists at the expected path. Then I would inspect permissions, registry metadata, Python and library compatibility, and whether the required preprocessing pipeline is available. Finally, I would reproduce the problem with a controlled test and verify predictions from the loaded artifact.",
            "I would identify whether the failure occurs during path resolution, file reading, deserialization, or prediction. I would verify the artifact exists, check registry metadata and permissions, and confirm that the environment's library versions are compatible. I would also check that the expected feature columns and preprocessing pipeline match the saved model. A small regression test should cover the failure. Rollback should only be claimed if the current implementation actually supports it and a fallback artifact has been validated."
        )

    if topic_answers:
        short = short or topic_answers[0]
        medium = medium or topic_answers[1]
        detailed = detailed or topic_answers[2]
    elif not short or not medium or not detailed:
        if "tell me about your project" in q:
            short = short or (f"My project is {project_name}. It uses {tech_list} and focuses on organizing the machine-learning lifecycle, including model training, drift checks, and candidate evaluation.")
            medium = medium or (f"My project is {project_name}. It supports data preparation, model training and evaluation, drift detection, candidate-model review, and version management. The repository scan detected {tech_list}. I would demonstrate each implemented feature using its source files and describe my own contribution accurately.")
            detailed = detailed or (f"My project is {project_name}, an adaptive ML lifecycle project. It brings together data preparation, baseline model training, evaluation, statistical drift checks, candidate-model assessment, human approval, and model-version management. The repository scan detected {tech_list}. I would walk through the modules and tests to explain how they connect, distinguish implemented features from planned work, and describe my individual contribution based on what I actually built.")
        else:
            topic = question.rstrip("?").strip()
            base = guidance or "I would verify the exact implementation in the cited source files."
            short = short or f"For {project_name}, I would explain the purpose of {topic.lower()} and how it relates to the project."
            medium = medium or f"For {topic}, I would explain the relevant component, its inputs and outputs, and how it fits into {project_name}. The current guidance is: {base} I would verify implementation details in the cited files."
            detailed = detailed or f"To explain {topic}, I would describe the problem, the component responsible, its inputs, processing steps, outputs, and how it interacts with the rest of {project_name}. The current guidance is: {base} Because the generator uses templates and does not deeply analyze runtime behavior, I would verify claims against the cited source files and tests before presenting them as facts."

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
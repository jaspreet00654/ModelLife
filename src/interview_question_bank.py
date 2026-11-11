
QUESTION_BANK = [
    {
        "category": "Project Logic",
        "difficulty": "Intermediate",
        "question": "Explain the complete ModelLife workflow.",
        "why_asked": "To check whether you understand your project's end-to-end architecture.",
        "short_answer": (
            "ModelLife loads and preprocesses churn data, trains a Logistic "
            "Regression model, and evaluates its performance. It monitors "
            "incoming data for drift and can train a candidate model when "
            "suitable labelled data is available. The candidate is evaluated "
            "and requires human approval before promotion."
        ),
        "medium_answer": (
            "ModelLife manages several stages of the machine learning "
            "lifecycle. First, it loads and preprocesses the Telco Customer "
            "Churn dataset. Then it trains a Logistic Regression model and "
            "evaluates it using accuracy, precision, recall, F1-score, and "
            "ROC-AUC. The system uses statistical tests to detect data drift. "
            "If drift is detected and labelled data is available, it can train "
            "a candidate model. The candidate is compared with the existing "
            "model, and a human approval step controls promotion or rejection. "
            "ModelLife also includes model version management, prediction "
            "logging, and monitoring components."
        ),
        "detailed_answer": (
            "ModelLife stands for Adaptive ML Model Lifecycle and Continuous "
            "Learning System. Its goal is to make machine learning model "
            "maintenance more organized after initial training.\n\n"
            "The workflow starts by loading the Telco Customer Churn dataset. "
            "During preprocessing, the project handles data types, prepares "
            "numerical and categorical features, and encodes the target "
            "variable. Numerical features are scaled and categorical features "
            "are encoded through the preprocessing pipeline.\n\n"
            "Next, the system trains a Logistic Regression baseline model. "
            "It evaluates model performance using accuracy, precision, recall, "
            "F1-score, and ROC-AUC rather than relying on accuracy alone.\n\n"
            "The drift detection component checks whether incoming data has "
            "changed compared with reference data using statistical tests. "
            "Drift indicates a change in the data distribution; it does not "
            "automatically prove that model accuracy has decreased.\n\n"
            "When drift is detected, the system can train a candidate model "
            "if suitable labelled data is available. The candidate is "
            "evaluated against the existing model using the evaluation "
            "process. A human approval step determines whether the candidate "
            "should be promoted or rejected. The project also contains model "
            "registry, prediction logging, and monitoring components.\n\n"
            "This workflow helps organize model updates while keeping model "
            "evaluation and human approval as important safeguards."
        ),
        "answer": (
            "Explain data loading, preprocessing, baseline training, drift "
            "detection, candidate evaluation, human approval, and versioning."
        ),
        "evidence": [
            "src/lifecycle.py",
            "src/data_loader.py",
            "src/drift_detection.py",
            "src/retraining.py",
            "src/model_evaluation.py",
            "src/approval.py",
            "src/model_registry.py",
        ],
        "follow_ups": [
            "What happens when no drift is detected?",
            "What happens if drift is detected but labels are unavailable?",
            "How does the system recover from a bad model promotion?",
        ],
        "source_type": "project",
    },
    {
        "category": "Backend",
        "difficulty": "Intermediate",
        "question": "How would you design a REST API for ModelLife predictions?",
        "why_asked": "To assess backend design and API fundamentals.",
        "short_answer": (
            "I would design a POST /predict endpoint that validates input, "
            "calls the prediction service, and returns the prediction and "
            "model version. This is a proposed design, not necessarily an "
            "implemented endpoint."
        ),
        "medium_answer": (
            "A REST API could expose a POST /predict endpoint. It would "
            "validate the incoming request, pass valid features to the "
            "prediction service, and return the predicted churn result and "
            "model version. Invalid input should produce a clear client "
            "error, while unexpected server failures should be handled "
            "without exposing sensitive details. Authentication and request "
            "logging would also be important for a deployed service. This "
            "describes a possible extension to ModelLife."
        ),
        "detailed_answer": (
            "I would design the prediction API around a POST /predict "
            "endpoint because prediction requests contain input data. The "
            "endpoint would validate the request schema, check required "
            "features and data types, and pass the validated data to the "
            "existing prediction service.\n\n"
            "The service would load the appropriate model and preprocessing "
            "pipeline, generate the prediction, and return a structured "
            "response containing the prediction and model version. Invalid "
            "requests should return an appropriate 4xx status code, while "
            "unexpected server errors should return a safe 5xx response.\n\n"
            "For deployment, I would add authentication, request-size "
            "limits, structured logging, and tests for valid and invalid "
            "inputs. This is a proposed API design; I would only describe "
            "it as implemented if the endpoint exists in the codebase."
        ),
        "answer": "Explain request validation, prediction service integration, responses, and error handling.",
        "evidence": ["src/prediction_service.py"],
        "follow_ups": [
            "How would you validate the request body?",
            "Which HTTP status codes would you return for invalid input?",
            "How would you protect the endpoint from unauthorized access?",
        ],
        "source_type": "general_and_proposed",
    },
    {
        "category": "Machine Learning",
        "difficulty": "Intermediate",
        "question": "Why does ModelLife compare multiple evaluation metrics?",
        "why_asked": "To test your understanding of model evaluation and trade-offs.",
        "short_answer": (
            "Accuracy alone can hide poor performance on the minority class. "
            "Precision, recall, F1-score, and ROC-AUC provide different views "
            "of model performance and help make a more informed comparison."
        ),
        "medium_answer": (
            "ModelLife evaluates models using accuracy, precision, recall, "
            "F1-score, and ROC-AUC because each metric measures a different "
            "aspect of performance. Accuracy measures overall correctness, "
            "precision measures how many predicted positive cases are "
            "actually positive, and recall measures how many actual positive "
            "cases are found. F1-score balances precision and recall, while "
            "ROC-AUC measures the model's ability to rank positive examples "
            "above negative ones. These metrics are useful when evaluating "
            "a candidate model against an existing model."
        ),
        "detailed_answer": (
            "Using multiple evaluation metrics gives a more complete picture "
            "of model quality. Accuracy is the proportion of all predictions "
            "that are correct, but it can be misleading when classes are "
            "imbalanced.\n\n"
            "Precision measures the proportion of predicted positive cases "
            "that are truly positive. Recall measures the proportion of actual "
            "positive cases that the model identifies. F1-score is the "
            "harmonic mean of precision and recall, making it useful when "
            "both matter. ROC-AUC measures ranking performance across "
            "classification thresholds when calculated from prediction "
            "scores or probabilities.\n\n"
            "In a churn prediction problem, recall can matter when missing "
            "customers who are likely to leave is costly. Precision also "
            "matters because contacting customers unnecessarily has a cost. "
            "ModelLife compares several metrics to help assess these trade-offs "
            "when evaluating a candidate model. The exact promotion criteria "
            "should be explained from the current evaluation code rather than "
            "assumed from the metric names alone."
        ),
        "answer": "Explain what each metric measures and why one metric alone may be insufficient.",
        "evidence": [
            "src/model_evaluation.py",
            "src/train.py",
            "src/performance_monitor.py",
        ],
        "follow_ups": [
            "When would recall matter more than precision?",
            "What is the limitation of evaluating ROC-AUC using hard predictions?",
            "What would you do if the candidate improves recall but lowers precision?",
        ],
        "source_type": "project",
    },
    {
        "category": "AI Agents",
        "difficulty": "Intermediate",
        "question": "How could an AI agent help a developer understand a project?",
        "why_asked": "To assess your understanding of agent architecture and tool use.",
        "short_answer": (
            "An AI agent could inspect relevant project files, retrieve "
            "useful context, and use tools to answer developer questions. "
            "This is a proposed capability, not a claim that ModelLife "
            "already contains a full AI agent."
        ),
        "medium_answer": (
            "A project-aware AI agent could help developers understand an "
            "unfamiliar codebase by identifying important files, functions, "
            "and dependencies. It could retrieve relevant source code and "
            "use that evidence to answer questions. Tool calling could let "
            "it perform specific permitted tasks, such as searching files "
            "or running approved tests. The current project analyzer scans "
            "project files, but that alone does not make it a full AI agent."
        ),
        "detailed_answer": (
            "A project-aware agent could combine code analysis, retrieval, "
            "and a language model. First, a read-only scanner would build "
            "an inventory of supported project files and technologies. "
            "When a developer asks a question, a retrieval component could "
            "select relevant source files or function definitions.\n\n"
            "The language model could then formulate an answer grounded in "
            "the retrieved evidence and cite file paths and line numbers. "
            "If tools are available, the agent could choose permitted actions "
            "such as searching for a function or running a test. Destructive "
            "actions and arbitrary code execution should not be allowed by "
            "default.\n\n"
            "ModelLife currently has project-scanning and question-generation "
            "components. A full language-model agent with autonomous planning "
            "and tool execution would be an extension, not something to claim "
            "as already implemented without corresponding code."
        ),
        "answer": "Explain retrieval, project context, tool use, and the difference between the current scanner and a full agent.",
        "evidence": ["src/project_analyzer.py"],
        "follow_ups": [
            "How is an agent different from a basic chatbot?",
            "What is tool calling?",
            "How would you prevent unsafe code execution?",
        ],
        "source_type": "general_and_proposed",
    },
    {
        "category": "RAG and Knowledge Retrieval",
        "difficulty": "Advanced",
        "question": "How could RAG make ModelLife's interview answers project-specific?",
        "why_asked": "To check your understanding of retrieval-augmented generation.",
        "short_answer": (
            "RAG retrieves relevant project content before a language model "
            "generates an answer. This can make answers more specific and "
            "help link explanations to source files."
        ),
        "medium_answer": (
            "A RAG pipeline could split project documentation and source code "
            "into searchable chunks, create embeddings, and retrieve chunks "
            "relevant to an interview question. Those chunks would be passed "
            "to a language model with instructions to answer using the "
            "retrieved evidence. The answer could include file references "
            "so the developer can verify it. RAG is a proposed extension "
            "unless a retrieval and generation pipeline is implemented."
        ),
        "detailed_answer": (
            "Retrieval-Augmented Generation combines information retrieval "
            "with language-model generation. First, project files would be "
            "parsed and split into meaningful chunks, such as functions, "
            "classes, or documentation sections. An embedding model could "
            "represent these chunks as vectors and store them in a searchable "
            "index.\n\n"
            "When the user asks a question, the system would retrieve the "
            "most relevant chunks and pass them as context to a language "
            "model. The prompt would require the model to distinguish facts "
            "supported by the code from assumptions and proposed designs. "
            "The answer could cite file paths and line ranges.\n\n"
            "To improve reliability, the system should handle missing "
            "evidence explicitly, test retrieval quality, and avoid claiming "
            "that retrieved code proves behavior that has not been tested. "
            "ModelLife's existing project analyzer is a possible source of "
            "project content, but a full RAG pipeline should be described "
            "as planned unless it is actually implemented."
        ),
        "answer": "Explain chunking, embeddings, retrieval, grounded generation, and source citations.",
        "evidence": ["src/project_analyzer.py"],
        "follow_ups": [
            "What are embeddings?",
            "How would you choose chunk sizes?",
            "How can you reduce hallucinations and cite source files?",
        ],
        "source_type": "general_and_proposed",
    },
    {
        "category": "System Design",
        "difficulty": "Advanced",
        "question": "How would you design ModelLife to support multiple users and projects?",
        "why_asked": "To evaluate scalability, isolation, and architecture decisions.",
        "short_answer": (
            "I would separate user accounts, project metadata, source files, "
            "model artifacts, and background jobs. Authentication and "
            "per-user access controls would isolate each user's resources."
        ),
        "medium_answer": (
            "To support multiple users, I would separate authentication, "
            "project metadata, source files, model artifacts, and lifecycle "
            "jobs into clear components. Each project would have an owner "
            "and access permissions. Large model artifacts could use object "
            "storage, while a database could store metadata and version "
            "records. Long-running training tasks could run as background "
            "jobs. The current local application would need additional "
            "infrastructure to support this design."
        ),
        "detailed_answer": (
            "I would start by defining separate components for authentication, "
            "project metadata, source-file indexing, model artifact storage, "
            "and background lifecycle jobs. Every project would have a stable "
            "identifier and an owner or access-control list.\n\n"
            "A database could store users, project settings, model versions, "
            "and job status. Object storage could hold large model artifacts "
            "and reports. A job queue would allow retraining and analysis to "
            "run asynchronously without blocking the user interface.\n\n"
            "Security would require authentication, authorization checks on "
            "every project resource, isolated processing, and careful handling "
            "of uploaded files. Monitoring, retries, audit logs, and backups "
            "would help operational reliability. This is a proposed scalable "
            "architecture; the current local project should not be described "
            "as already providing all these services."
        ),
        "answer": "Cover authentication, resource isolation, artifact storage, background jobs, and reliability.",
        "evidence": ["src/project_analyzer.py", "src/model_registry.py"],
        "follow_ups": [
            "How would you isolate one user's project from another?",
            "Where would you store large model artifacts?",
            "How would you schedule long-running retraining jobs?",
        ],
        "source_type": "general_and_proposed",
    },
    {
        "category": "Security",
        "difficulty": "Intermediate",
        "question": "What security risks arise when users upload source code?",
        "why_asked": "To assess secure handling of user-provided files.",
        "short_answer": (
            "Risks include exposed secrets, malicious files, oversized "
            "uploads, path traversal, and accidental code execution. "
            "Use strict validation, access controls, and isolated processing."
        ),
        "medium_answer": (
            "A source-code analyzer may encounter API keys, passwords, "
            "malicious files, path traversal attempts, or excessively large "
            "files. It should limit supported file types and file sizes, "
            "validate paths, redact likely secrets, and avoid executing "
            "uploaded code. Access controls and isolated processing provide "
            "additional protection. Regex-based secret detection can help "
            "but cannot guarantee that all secrets will be found."
        ),
        "detailed_answer": (
            "The main risks include accidental disclosure of credentials, "
            "malicious or malformed files, path traversal, denial of service "
            "through oversized inputs, and unsafe code execution. A scanner "
            "should accept only intended file types, enforce file-count and "
            "size limits, resolve paths safely within the permitted project "
            "root, and avoid following unsafe links.\n\n"
            "Likely secrets should be redacted before content is displayed "
            "or sent elsewhere, but regex-based detection is imperfect and "
            "cannot guarantee complete protection. The analyzer should be "
            "read-only by default and should never execute uploaded source "
            "files merely to understand them. If execution is necessary, it "
            "should happen in a properly isolated environment with restricted "
            "permissions and resources.\n\n"
            "The project analyzer has basic file exclusions and pattern-based "
            "secret redaction. Those safeguards reduce some risks but should "
            "not be presented as a complete security boundary."
        ),
        "answer": "Discuss secret exposure, path validation, limits, read-only analysis, and the limits of regex redaction.",
        "evidence": ["src/project_analyzer.py"],
        "follow_ups": [
            "Why is regex-based secret redaction not sufficient?",
            "How would you prevent path traversal?",
            "Why should project analysis be read-only by default?",
        ],
        "source_type": "general_and_project",
    },
    {
        "category": "Debugging",
        "difficulty": "Intermediate",
        "question": "How would you investigate a failure when loading a saved model?",
        "why_asked": "To assess systematic debugging and artifact management.",
        "short_answer": (
            "I would check the artifact path, file existence, permissions, "
            "registry metadata, and library compatibility. Then I would "
            "reproduce the error with a small controlled test."
        ),
        "medium_answer": (
            "First, I would read the full error message and verify that the "
            "model file exists at the expected path. I would check file "
            "permissions, registry metadata, serialization compatibility, "
            "and whether the required preprocessing objects are available. "
            "Then I would reproduce the issue in a controlled test and "
            "verify that the loaded artifact produces the expected output. "
            "I would also test rollback if the active model is unavailable."
        ),
        "detailed_answer": (
            "I would investigate the failure systematically. First, I would "
            "capture the full traceback and identify whether the failure "
            "occurs while resolving the path, reading the file, deserializing "
            "the model, or making a prediction.\n\n"
            "Next, I would verify that the artifact exists, the process has "
            "permission to read it, and the model registry points to the "
            "correct version. I would check whether the Python and library "
            "versions are compatible with the serialized artifact. I would "
            "also confirm that the expected feature columns and preprocessing "
            "pipeline match the model's requirements.\n\n"
            "I would reproduce the issue using a small test, verify the "
            "loaded model on known input, and add a regression test for the "
            "failure. If the active model cannot be loaded, a safe fallback "
            "or rollback should only be used if the implementation supports "
            "it and the fallback artifact has been validated."
        ),
        "answer": "Check paths, metadata, deserialization, preprocessing compatibility, and controlled tests.",
        "evidence": ["src/model_registry.py", "src/prediction_service.py"],
        "follow_ups": [
            "What if the registry points to a missing file?",
            "Why should preprocessing be saved with the model?",
            "How would you test rollback?",
        ],
        "source_type": "general_and_project",
    },
    {
        "category": "Behavioral and Project Defense",
        "difficulty": "Beginner",
        "question": "What was a challenging problem you solved while building ModelLife?",
        "why_asked": "To understand your ownership, troubleshooting, and learning process.",
        "short_answer": (
            "One challenge I faced was [describe a real issue]. I investigated "
            "the cause, tried a solution, and tested whether it fixed the "
            "problem. I learned [your actual lesson]."
        ),
        "medium_answer": (
            "One challenging part of building ModelLife was [describe a real "
            "problem you personally faced]. I first noticed [symptom]. I "
            "investigated it by [actual debugging steps], then changed "
            "[actual code or configuration]. I verified the fix by "
            "[actual test or result]. This taught me [lesson]."
        ),
        "detailed_answer": (
            "Choose a real challenge from your own development experience "
            "and explain it using the STAR structure.\n\n"
            "Situation: Describe the project stage and the problem you faced.\n\n"
            "Task: Explain what you needed to achieve and why it mattered.\n\n"
            "Action: Describe the debugging steps you personally performed, "
            "the alternatives you considered, and the change you made.\n\n"
            "Result: State the actual test or observation that showed whether "
            "the fix worked, followed by what you learned and what you would "
            "improve next time.\n\n"
            "Replace these prompts with your real experience. Do not invent "
            "an issue, result, or contribution just to make the answer sound "
            "impressive."
        ),
        "answer": "Use a real example: problem, investigation, action, verification, and lesson learned.",
        "evidence": [],
        "follow_ups": [
            "What alternatives did you consider?",
            "How did you verify the fix?",
            "What would you improve if you revisited it?",
        ],
        "source_type": "personal_response",
    },
]


def get_question_bank():
    """Return the interview question bank."""
    return QUESTION_BANK


def get_answer_versions(item):
    """Return short, medium, and detailed answers for a question."""
    return {
        "Short": item.get("short_answer") or item.get("answer", ""),
        "Medium": item.get("medium_answer") or item.get("answer", ""),
        "Long / Detailed": item.get("detailed_answer") or item.get("answer", ""),
    }


if __name__ == "__main__":
    questions = get_question_bank()
    print(f"Total bank questions: {len(questions)}")

    for index, item in enumerate(questions, start=1):
        print(f"\nQ{index} [{item['category']} | {item['difficulty']}]")
        print(item["question"])
        print("Source type:", item["source_type"])
        print("Answer versions:", ", ".join(get_answer_versions(item)))

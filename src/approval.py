def request_approval(candidate_version, validation_result):
    """Request human approval for a validated candidate."""

    if not validation_result["approved"]:
        print("Candidate failed validation.")
        print("Reason:", validation_result["reason"])
        return False

    print(f"Candidate model: {candidate_version}")
    print("Validation passed.")

    decision = input("Approve this candidate? (yes/no): ")
    return decision.strip().lower() in {"yes", "y"}
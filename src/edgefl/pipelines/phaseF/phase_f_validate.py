"""Scoped F acceptance requires clean execution and its actual search profile."""
from edgefl.optimization.storage import stage, load_stage, checked, reference
from edgefl.optimization.review import software_evidence, accepted_review

def execute(config, prepared_path, run_review, profile_review, test_report):
    software_evidence(test_report)
    memo = {}
    prepared = load_stage(config, prepared_path, "prepare-phase-f", memo=memo)
    run = accepted_review(config, run_review, "run-evolution-federated", memo)
    profile = accepted_review(config, profile_review, "profile-search", memo)
    run_review_stage = load_stage(config, run_review, "review-phase-f", memo=memo)
    if run["inputs"]["prepared_f"] != reference(config.workspace, prepared_path) or run["summary"]["condition"] != "clean":
        raise ValueError("Phase F acceptance requires matched clean execution")
    if profile["inputs"]["run"] != run_review_stage["inputs"]["subject"]:
        raise ValueError("Profile belongs to a different run")
    inputs = {"prepared_f": prepared_path, "run_review": run_review, "profile_review": profile_review}
    with stage(config, "validate-phase-f", prepared["identity"], checked(config.workspace, prepared["gate"]), inputs) as attempt:
        return attempt.finish({"status": "PASS", "scope": prepared["identity"], "seed": prepared["summary"]["seed"],
            "software_evidence": reference(config.workspace, test_report), "source_verified_gate": "PASS",
            "actual_search_profile": "PASS", "research_claim": "scoped_execution_only",
            "other_tasks_seeds_and_workloads": "not_established"})

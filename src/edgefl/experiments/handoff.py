"""Checkpoint eligibility export only; final-test access is not implemented here."""
from edgefl.data.storage import read_json, write_json
from edgefl.experiments.storage import load_stage, artifact, stage, checked, reference, software_evidence
from edgefl.experiments.review import accepted_subject


def validate(config, matrix_path, summary_path, reviews_path, test_report):
    software_evidence(test_report)
    compiled = load_stage(config, matrix_path, "compile-experiments")
    summary = load_stage(config, summary_path, "summarize-experiments")
    if summary["inputs"]["matrix"] != reference(config.workspace, matrix_path):
        raise ValueError("Summary belongs to a different matrix")
    matrix = read_json(artifact(config, compiled, "cases.json"))
    report = read_json(artifact(config, summary, "development.json"))
    paths = read_json(reviews_path)
    if not isinstance(paths, list):
        raise ValueError("Acceptance requires an explicit review list")
    inputs = {"matrix": matrix_path, "summary": summary_path, "reviews": reviews_path, "test_report": test_report}
    accepted = {}
    from edgefl.config import workspace_path
    for i, raw in enumerate(paths):
        path = workspace_path(config.workspace, raw)
        run = accepted_subject(config, path, "run-experiment")
        key = run["summary"]["case_id"]
        if key in accepted or key not in report["results"]:
            raise ValueError("Duplicate or unrelated experiment review")
        review = load_stage(config, path)
        if review["inputs"]["subject"] != report["results"][key]["run"]:
            raise ValueError("Review and reported attempt differ")
        accepted[key] = review["inputs"]["subject"]
        inputs[f"review_{i}"] = path
    missing = sorted(set(matrix["cases"]) - set(accepted))
    unresolved_capture = matrix["capture_folds_unregistered"]
    complete = not missing and not report["failed_cases"] and not unresolved_capture
    with stage(config, "validate-phase-g", inputs, matrix_id=matrix["matrix_id"]) as attempt:
        return attempt.finish({"status": "PASS_SCOPED" if complete else "INCOMPLETE", "software": "PASS",
            "matrix_id": matrix["matrix_id"], "accepted_runs": accepted, "missing_reviews": missing,
            "capture_folds_unregistered": unresolved_capture, "deferred": matrix["deferred"],
            "research_completion": complete and config.values["evidence_kind"] == "research",
            "whole_phase_g_complete": False, "software_evidence": reference(config.workspace, test_report),
            "final_test_evaluated": False})


def export(config, acceptance_path):
    accepted = load_stage(config, acceptance_path, "validate-phase-g")
    software_evidence(checked(config.workspace, accepted["summary"]["software_evidence"]))
    if accepted["summary"]["status"] != "PASS_SCOPED":
        raise ValueError("Incomplete experiment scope cannot publish checkpoint eligibility")
    checkpoints = []
    for key, ref in sorted(accepted["summary"]["accepted_runs"].items()):
        run = load_stage(config, checked(config.workspace, ref), "run-experiment")
        if not run["summary"].get("best"):
            raise ValueError("Missing selected checkpoint")
        checkpoints.append({"case_id": key, "case": run["summary"]["case"], "run": ref,
                            "checkpoint": run["summary"]["best"]["reference"], "selection": run["summary"]["best"],
                            "gate": run["gate"], "prepared": run["inputs"]["prepared"]})
    with stage(config, "export-phase-h-handoff", {"acceptance": acceptance_path}, matrix_id=accepted["summary"]["matrix_id"]) as attempt:
        write_json(attempt.directory / "eligible_checkpoints.json", checkpoints)
        return attempt.finish({"eligible": len(checkpoints), "phase_h_freeze_performed": False,
                               "final_test_evaluated": False, "deferred": accepted["summary"]["deferred"]})

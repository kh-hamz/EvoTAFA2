"""Explicit review of completed F execution or measured profile."""
from dataclasses import asdict
from edgefl.contracts.phase_d import LearningReview
from edgefl.optimization.storage import load_stage, stage, checked

def execute(config, subject_path, disposition, rationale, limitations=""):
    subject = load_stage(config, subject_path)
    if subject["stage"] not in ("run-evolution-federated", "profile-search"):
        raise ValueError("Review requires a run or profile")
    decision = LearningReview(disposition, rationale, limitations)
    if disposition != "unresolved" and (subject["summary"].get("correctness") != "PASS" or subject["summary"].get("attack_coverage", "PASS") != "PASS"):
        raise ValueError("Review cannot override failed execution")
    with stage(config, "review-phase-f", subject["identity"], checked(config.workspace, subject["gate"]), {"subject": subject_path}) as attempt:
        return attempt.finish({"decision": asdict(decision), "status": "UNRESOLVED" if disposition == "unresolved" else "PASS"})

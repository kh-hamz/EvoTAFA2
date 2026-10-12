"""Immutable explicit interpretation of calibration or Phase E run evidence."""

from dataclasses import asdict
from edgefl.contracts.phase_d import LearningReview
from edgefl.trust.storage import stage, load_stage, checked


def execute(config, subject_path, disposition, rationale, limitations=""):
    subject = load_stage(config, subject_path)
    if subject["stage"] not in ("calibrate-risk", "run-trust-federated"):
        raise ValueError("Review requires calibration or comparison run")
    decision = LearningReview(disposition, rationale, limitations)
    if disposition != "unresolved" and (subject["summary"].get("correctness") != "PASS" or subject["summary"].get("attack_coverage", "PASS") != "PASS"):
        raise ValueError("Review cannot override failed Phase E execution/coverage")
    with stage(config, "review-phase-e", subject["identity"], checked(config.workspace, subject["gate"]), {"subject": subject_path}) as attempt:
        return attempt.finish({"decision": asdict(decision), "subject_stage": subject["stage"], "status": "PASS" if disposition != "unresolved" else "UNRESOLVED"})

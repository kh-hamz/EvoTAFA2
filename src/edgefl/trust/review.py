"""Reviewed evidence cannot override failed numerical, source, or coverage gates."""

from edgefl.contracts.phase_d import LearningReview
from edgefl.trust.storage import load_stage, checked


def accepted_review(config, path, subject_path=None, memo=None):
    review = load_stage(config, path, "review-phase-e", memo=memo)
    decision = LearningReview(**review["summary"]["decision"])
    subject = checked(config.workspace, review["inputs"]["subject"])
    if subject_path is not None and subject != subject_path.resolve():
        raise ValueError("Review references a different Phase E subject")
    evidence = load_stage(config, subject, memo=memo)
    if decision.disposition == "unresolved" or evidence["summary"].get("correctness") != "PASS" or evidence["summary"].get("attack_coverage", "PASS") != "PASS":
        raise ValueError("Unresolved/failed Phase E evidence blocks advancement")
    return evidence


def check_acceptance(config, path, memo=None):
    """Recompute the existing E gate from its reviewed evidence and current tests."""
    from edgefl.pipelines.phaseE.phase_e_validate import validate_scope
    memo = {} if memo is None else memo
    acceptance = load_stage(config, path, "validate-phase-e", memo=memo)
    key = ("current_e_acceptance", path.resolve())
    if key not in memo:
        inputs = acceptance["inputs"]
        _, result = validate_scope(config, checked(config.workspace, inputs["metadata"]),
            checked(config.workspace, inputs["calibration"]), checked(config.workspace, inputs["calibration_review"]),
            [checked(config.workspace, ref) for name, ref in inputs.items() if name.startswith("review_")],
            checked(config.workspace, acceptance["summary"]["software_evidence"]), acceptance["options"]["seed"], memo)
        if result["status"] != "PASS" or acceptance["summary"]["status"] != "PASS":
            raise ValueError("Current complete Phase E acceptance required")
        memo[key] = True
    return acceptance

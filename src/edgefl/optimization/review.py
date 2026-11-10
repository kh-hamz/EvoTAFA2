"""Phase F review and current implementation-bound software evidence."""
from edgefl.contracts.phase_d import LearningReview
from edgefl.data.storage import read_json
from edgefl.optimization.storage import load_stage, checked, implementation_hash, regression_hash

def accepted_review(config, path, stage_name=None, memo=None):
    review = load_stage(config, path, "review-phase-f", memo=memo)
    decision = LearningReview(**review["summary"]["decision"])
    subject = load_stage(config, checked(config.workspace, review["inputs"]["subject"]), stage_name, memo=memo)
    if decision.disposition == "unresolved" or subject["summary"].get("correctness") != "PASS" or subject["summary"].get("attack_coverage", "PASS") != "PASS":
        raise ValueError("Unresolved/failed Phase F review")
    return subject

def software_evidence(path):
    evidence = read_json(path)
    if (evidence.get("schema_version") != "phase-f.tests.v1" or evidence.get("successful") is not True
        or any(evidence.get(k) != 0 for k in ("failures", "errors", "skipped"))
        or type(evidence.get("tests")) is not int or evidence["tests"] <= 0
        or evidence.get("implementation_sha256") != implementation_hash()
        or evidence.get("regression_sha256") != regression_hash()):
        raise ValueError("Current complete Phase F software evidence required")
    from edgefl.optimization.diagnostics import environment
    if evidence.get("search_environment") != environment():
        raise ValueError("Software evidence search environment mismatch")
    return evidence

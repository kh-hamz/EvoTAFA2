"""Scoped evidence validation, separate from software-only completion."""

from edgefl.contracts.phase_e import CONDITIONS
from edgefl.trust.storage import stage, load_stage, checked, reference, implementation_hash, regression_hash
from edgefl.trust.review import accepted_review
from edgefl.data.storage import read_json


def validate_scope(config, metadata_path, calibration_path, calibration_review_path, reviews, test_report, seed, memo=None):
    evidence = read_json(test_report)
    if (evidence.get("schema_version") != "phase-e.tests.v1" or evidence.get("successful") is not True
            or evidence.get("failures") != 0 or evidence.get("errors") != 0 or evidence.get("skipped") != 0
            or type(evidence.get("tests")) is not int or evidence["tests"] <= 0
            or evidence.get("implementation_sha256") != implementation_hash()
            or evidence.get("regression_sha256") != regression_hash()):
        raise ValueError("Current passing full Phase E software evidence required")
    memo = {} if memo is None else memo
    metadata = load_stage(config, metadata_path, "prepare-phase-e", memo=memo)
    calibration = load_stage(config, calibration_path, "calibrate-risk", memo=memo)
    accepted_review(config, calibration_review_path, calibration_path, memo)
    if calibration["inputs"]["metadata"] != reference(config.workspace, metadata_path):
        raise ValueError("Acceptance calibration metadata mismatch")
    seen, initial = set(), None
    for path in reviews:
        run = accepted_review(config, path, memo=memo)
        if run["stage"] != "run-trust-federated" or run["summary"]["seed"] != seed:
            raise ValueError("Acceptance requires matched comparison runs")
        for name, expected in (("metadata", metadata_path), ("calibration", calibration_path)):
            if run["inputs"][name] != reference(config.workspace, expected):
                raise ValueError("Acceptance mixes data/calibration scopes")
        if initial is not None and initial != run["inputs"]["initialization"]:
            raise ValueError("Acceptance mixes initializations")
        initial = run["inputs"]["initialization"]
        key = (run["summary"]["condition"], run["summary"]["method"])
        if key in seen:
            raise ValueError("Duplicate condition/method review")
        seen.add(key)
    required = {(condition, method) for condition in CONDITIONS for method in ("fedavg", "adaptive", "frozen")}
    missing = sorted(required - seen)
    return metadata, {"status": "FAIL" if missing else "PASS", "scope": metadata["identity"], "seed": seed,
            "missing_conditions": missing, "software_evidence": reference(config.workspace, test_report),
            "source_verified_gate": "PASS", "calibration_review": "complete", "research_claim": "scoped_execution_only",
            "other_tasks_and_seeds": "not_established_by_this_acceptance"}


def execute(config, metadata_path, calibration_path, calibration_review_path, reviews, test_report, seed):
    metadata, summary = validate_scope(config, metadata_path, calibration_path, calibration_review_path, reviews, test_report, seed)
    inputs = {"metadata": metadata_path, "calibration": calibration_path, "calibration_review": calibration_review_path,
              **{f"review_{i}": p for i, p in enumerate(reviews)}}
    with stage(config, "validate-phase-e", metadata["identity"], checked(config.workspace, metadata["gate"]), inputs, {"seed": seed}) as attempt:
        return attempt.finish(summary)

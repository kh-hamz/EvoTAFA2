"""Shared Phase C orchestration checks; domain algorithms live in client_data."""

from pathlib import Path

from edgefl.client_data.storage import artifact, load_phase_b_completion, read_json


def accepted_phase_b(config, validation_path: Path, split_path: Path,
                     dataset: str, fold: str, context=None) -> tuple[dict, dict]:
    from edgefl.data.integrity import VerificationContext
    context = context if context is not None else VerificationContext()
    validation = load_phase_b_completion(config, validation_path, "validate-phase-b", dataset, context)
    report = read_json(artifact(config, validation, "summary"))
    if report["status"] not in ("PASS", "PASS_WITH_LIMITATIONS"):
        raise ValueError(f"Phase B validation status {report['status']} blocks Phase C")
    split = load_phase_b_completion(config, split_path, "global-split", dataset, context)
    protocol = split["options"].get("protocol", "").lower()
    expected_key = "splits_" + protocol
    expected = validation["inputs"].get(expected_key)
    actual = validation["inputs"].get(expected_key, {})
    from edgefl.client_data.storage import artifact_ref
    supplied = artifact_ref(config.workspace, split_path, "completion", "phase-b.v2")
    if expected is None or actual != supplied:
        raise ValueError("Split was not accepted by the supplied Phase B validation")
    summary = read_json(artifact(config, split, "summary"))
    if fold not in summary["folds"]:
        raise ValueError(f"Unknown split fold: {fold}")
    if not summary["folds"][fold]["feasible"]:
        raise ValueError(f"Phase B split fold is infeasible: {fold}")
    return validation, split

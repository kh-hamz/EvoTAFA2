"""Fit normalization from designated pilots and independently audit seed 103."""

from edgefl.trust.storage import stage, load_stage, checked, artifact, prepared_for, calibration_scope
from edgefl.data.storage import read_json, write_json


def samples(config, pilot):
    records = []
    index = read_json(artifact(config, pilot, "round_index.json"))["rounds"]
    for current in range(1, pilot["summary"]["completed_steps"] + 1):
        report = read_json(checked(config.workspace, index[str(current)]["assessment"]))
        if report["invalid"]:
            raise ValueError("Clean calibration cannot include rejected submissions")
        records.extend({"round": current, "client": client, "raw": value["raw_risk"]}
                       for client, value in report["observation"]["clients"].items())
    return records


def execute(config, metadata_path, fit_paths, audit_path):
    if len(fit_paths) != 2:
        raise ValueError("Exactly two fitting pilots are required")
    memo = {}
    metadata = load_stage(config, metadata_path, "prepare-phase-e", memo=memo)
    prepared = prepared_for(config, metadata, memo)
    scope = calibration_scope(config, metadata, prepared)
    pilots, paths = {}, {}
    for path in [*fit_paths, audit_path]:
        pilot = load_stage(config, path, "run-clean-pilot", memo=memo)
        seed = pilot["summary"]["seed"]
        if seed in pilots or pilot["summary"]["correctness"] != "PASS" or pilot["summary"]["condition"] != "clean" or pilot["summary"]["completed_steps"] != config.learning.values["rounds"]:
            raise ValueError("Invalid/incomplete clean calibration pilot")
        if read_json(artifact(config, pilot, "calibration_scope.json")) != scope:
            raise ValueError("Calibration pilot scope mismatch")
        pilots[seed], paths[seed] = pilot, path
    if set(pilots) != {101, 102, 103} or paths[103] != audit_path or any(paths[s] not in fit_paths for s in (101, 102)):
        raise ValueError("Calibration fitting/audit seed assignment mismatch")
    from edgefl.trust.calibration import fit, audit
    from edgefl.trust.metadata import Metadata
    fitted = fit({s: samples(config, pilots[s]) for s in (101, 102)}, scope)
    meta = Metadata(config.workspace, metadata)
    report = audit(fitted, samples(config, pilots[103]), meta.spec["profiles"], meta.spec["declared_specialist_scenario"])
    with stage(config, "calibrate-risk", metadata["identity"], checked(config.workspace, metadata["gate"]),
               {"metadata": metadata_path, "fit_101": paths[101], "fit_102": paths[102], "audit_103": paths[103]}) as attempt:
        write_json(attempt.directory / "calibration.json", fitted)
        write_json(attempt.directory / "audit.json", report)
        return attempt.finish({"correctness": "PASS", "review_required": True, "fit_client_rounds": fitted["fit_client_rounds"],
                               "audit_client_rounds": report["cohorts"]["all_honest"]["client_rounds"]})

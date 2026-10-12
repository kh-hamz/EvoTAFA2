"""Reproduce a recorded assessment without mutating model or reputation state."""

from edgefl.contracts.records import ArtifactRef, Compatibility, ClientUpdate, PartitionRef, Partition
from edgefl.contracts.interfaces import ScoringRequest
from edgefl.trust.storage import stage, load_stage, checked, artifact, prepared_for
from edgefl.data.storage import read_json, write_json


def execute(config, run_path, round_id):
    memo = {}
    source = load_stage(config, run_path, "run-trust-federated", memo=memo)
    if type(round_id) is not int or not 1 <= round_id <= source["summary"]["completed_steps"]:
        raise ValueError("Reassessment round outside completed trajectory")
    index = read_json(artifact(config, source, "round_index.json"))["rounds"]
    original = read_json(checked(config.workspace, index[str(round_id)]["assessment"]))
    metadata = load_stage(config, checked(config.workspace, source["inputs"]["metadata"]), "prepare-phase-e", memo=memo)
    prepared = prepared_for(config, metadata, memo)
    calibration_stage = load_stage(config, checked(config.workspace, source["inputs"]["calibration"]), "calibrate-risk", memo=memo)
    calibration = read_json(artifact(config, calibration_stage, "calibration.json"))
    from edgefl.learning.data import LearningData
    from edgefl.learning.models import configure
    from edgefl.trust.metadata import Metadata
    from edgefl.trust.scoring import TrustedClientScorer
    from dataclasses import asdict
    if configure(config.learning.values) != read_json(artifact(config, source, "runtime.json")):
        raise ValueError("Reassessment environment mismatch")
    data, rows = LearningData(config.learning, prepared), Metadata(config.workspace, metadata)
    trusted = PartitionRef(ArtifactRef(**original["trusted"]["manifest"]), Partition(original["trusted"]["role"]))
    comp = Compatibility(**original["compatibility"])
    updates = tuple(ClientUpdate(**{**u, "parent_model": ArtifactRef(**u["parent_model"]), "delta": ArtifactRef(**u["delta"]),
                                    "compatibility": Compatibility(**u["compatibility"])}) for u in original["updates"])
    originals = [r["original"] for r in rows.rows("trusted")]
    vocabulary = sorted(set(originals) | {name for p in rows.spec["profiles"].values() for name in p["support"]})
    scorer = TrustedClientScorer(config.workspace, data.view("trusted"), originals, trusted, comp, data.spec["labels"],
                                data.spec["identity"]["task"], config.learning.values, vocabulary,
                                config.values["expertise_support_scale"], config.values["minimum_valid_clients"])
    observed = scorer.observe(ScoringRequest(trusted, ArtifactRef(**original["parent"]), updates, comp))
    assessments = [asdict(a) for a in scorer.assessments(observed, calibration, original["prior_reputation"])]
    # JSON roundtrip makes tuple/list representation immaterial to comparison.
    from edgefl.config import canonical_json
    matches = canonical_json(assessments) == canonical_json(original["assessments"])
    with stage(config, "assess-round", source["identity"], checked(config.workspace, source["gate"]), {"run": run_path}, {"round": round_id}) as attempt:
        write_json(attempt.directory / "assessment.json", {"observation": observed, "assessments": assessments})
        return attempt.finish({"status": "PASS" if matches else "FAIL", "matches_recorded_assessments": matches, "round": round_id, "state_advanced": False})

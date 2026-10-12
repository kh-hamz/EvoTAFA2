"""One controlled Phase E comparison; no experiment-matrix or final-test bypass."""

from edgefl.config import workspace_path
from edgefl.contracts.phase_e import ArtifactPolicy, METHODS
from edgefl.trust.storage import (stage, load_stage, checked, artifact, prepared_for, check_d_acceptance,
                                 implementation_hash, reference, calibration_scope, publish_round_index)
from edgefl.trust.review import accepted_review
from edgefl.learning import storage as d
from edgefl.data.storage import read_json


def execute(config, metadata_path, initialization_path, acceptance_path, plan_path, calibration_path,
            calibration_review_path, method, seed, resume=None):
    if method not in METHODS or seed not in config.learning.values["seeds"]:
        raise ValueError("Unregistered Phase E method/seed")
    memo = {}
    metadata = load_stage(config, metadata_path, "prepare-phase-e", memo=memo)
    prepared = prepared_for(config, metadata, memo)
    check_d_acceptance(config, acceptance_path, prepared, seed, memo)
    initialization = d.load_stage(config.learning, initialization_path, "initialize-model", memo=memo)
    if initialization["options"]["seed"] != seed or initialization["inputs"]["prepared"] != metadata["inputs"]["prepared"]:
        raise ValueError("Comparison initialization mismatch")
    attack = load_stage(config, plan_path, "plan-attacks", memo=memo)
    calibration_stage = load_stage(config, calibration_path, "calibrate-risk", memo=memo)
    accepted_review(config, calibration_review_path, calibration_path, memo)
    if any(value["inputs"]["metadata"] != reference(config.workspace, metadata_path) for value in (attack, calibration_stage)):
        raise ValueError("Attack/calibration metadata mismatch")
    plan = read_json(artifact(config, attack, "attack_plan.json"))
    if plan["seed"] != seed or plan["coverage_status"] != "PASS":
        raise ValueError("Attack plan seed or coverage unsuccessful")
    if plan["condition"] != "clean" and config.learning.values["rounds"] < plan["activation_round"]:
        raise ValueError("Attack never activates within configured rounds")
    calibration = read_json(artifact(config, calibration_stage, "calibration.json"))
    if calibration["scope"] != calibration_scope(config, metadata, prepared):
        raise ValueError("Frozen calibration scope mismatch")
    from edgefl.learning import checkpoints
    from edgefl.learning.data import LearningData
    from edgefl.learning.engine import run
    from edgefl.trust.metadata import Metadata
    from edgefl.trust.session import TrustSession
    inputs = {"metadata": metadata_path, "initialization": initialization_path, "phase_d_acceptance": acceptance_path,
              "attack_plan": plan_path, "calibration": calibration_path, "calibration_review": calibration_review_path}
    bindings = {name: reference(config.workspace, path) for name, path in inputs.items()}
    resumed = None
    options = {"seed": seed, "method": method, "condition": plan["condition"]}
    if resume:
        marker = read_json(resume)
        if (marker.get("schema_version") != "phase-e.resume.v1" or marker.get("configuration_sha256") != config.sha256
                or marker.get("implementation_sha256") != implementation_hash() or marker.get("identity") != metadata["identity"]
                or marker.get("gate") != metadata["gate"] or marker.get("inputs") != bindings):
            raise ValueError("Invalid/stale Phase E resume marker or dependencies")
        resumed = (checkpoints.load(checked(config.workspace, marker["checkpoint"])), marker)
        options["resume"] = reference(config.workspace, resume)
    rows = Metadata(config.workspace, metadata)
    def session(*args):
        return TrustSession(*args, settings=config.values, metadata=rows, plan=plan, calibration=calibration, bindings=bindings)
    with stage(config, "run-trust-federated", metadata["identity"], checked(config.workspace, metadata["gate"]), inputs, options) as attempt:
        summary = run(config.learning, LearningData(config.learning, prepared), prepared, initialization, attempt, method, seed, resumed,
                      session_factory=session, artifact_policy=ArtifactPolicy("phase-e", config.sha256, implementation_hash()))
        final_state = publish_round_index(config, attempt.directory, summary["final_checkpoint"])
        effective = final_state.get("effective_attacking_client_rounds", 0)
        return attempt.finish({**summary, "condition": plan["condition"], "attack_coverage": "PASS" if plan["condition"] == "clean" or effective else "UNSUCCESSFUL",
                               "effective_attacking_client_rounds": effective})

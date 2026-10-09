"""Separate clean development trajectories with recorded deterministic initialization."""

from edgefl.trust.storage import stage, load_stage, checked, prepared_for, check_d_acceptance, implementation_hash, reference, calibration_scope, publish_round_index
from edgefl.trust.config import training_configuration
from edgefl.contracts.phase_e import ArtifactPolicy
from edgefl.data.storage import write_json, read_json


def execute(config, metadata_path, acceptance_path, seed, resume=None):
    if seed not in (*config.values["fit_seeds"], config.values["audit_seed"]):
        raise ValueError("Clean pilot seed must be 101, 102 or 103")
    memo = {}
    metadata = load_stage(config, metadata_path, "prepare-phase-e", memo=memo)
    prepared = prepared_for(config, metadata, memo)
    check_d_acceptance(config, acceptance_path, prepared, memo=memo)
    from edgefl.trust.metadata import Metadata
    from edgefl.attacks.plan import build_plan
    from edgefl.trust.session import TrustSession
    from edgefl.learning.models import initialized
    from edgefl.learning import checkpoints
    from edgefl.learning.data import LearningData
    from edgefl.learning.engine import run
    data = LearningData(config.learning, prepared)
    rows = Metadata(config.workspace, metadata)
    plan = build_plan(rows, config.values, seed, "clean")
    inputs = {"metadata": metadata_path, "phase_d_acceptance": acceptance_path}
    options = {"seed": seed, "condition": "clean", "method": "fedavg"}
    resumed = None
    if resume:
        marker = read_json(resume)
        if (marker.get("schema_version") != "phase-e.resume.v1" or marker.get("configuration_sha256") != config.sha256
                or marker.get("implementation_sha256") != implementation_hash() or marker.get("identity") != metadata["identity"]
                or marker.get("gate") != metadata["gate"] or marker.get("inputs") != {k: reference(config.workspace, p) for k, p in inputs.items()}):
            raise ValueError("Invalid/stale clean pilot resume marker")
        resumed = (checkpoints.load(checked(config.workspace, marker["checkpoint"])), marker)
        options["resume"] = reference(config.workspace, resume)
    with stage(config, "run-clean-pilot", metadata["identity"], checked(config.workspace, metadata["gate"]),
               inputs, options) as attempt:
        if resumed:
            initial_ref = resumed[0]["initialization"]
            checked(config.workspace, initial_ref)
        else:
            model = initialized(data.spec["dimension"], len(data.spec["labels"]), seed)
            path = attempt.directory / "initialization.pt"
            checkpoints.save(path, {"model": model.state_dict()})
            initial_ref = reference(config.workspace, path)
        initialization = {"identity": metadata["identity"], "options": {"seed": seed}, "artifacts": {"mlp.pt": initial_ref}}
        write_json(attempt.directory / "initialization.json", initialization)
        write_json(attempt.directory / "training_configuration.json", training_configuration(config, seed).values)
        scope = calibration_scope(config, metadata, prepared)
        write_json(attempt.directory / "calibration_scope.json", scope)
        def session(*args):
            return TrustSession(*args, settings=config.values, metadata=rows, plan=plan, calibration=None, bindings={"scope": scope, "seed": seed})
        summary = run(training_configuration(config, seed), data, prepared, initialization, attempt, "fedavg", seed, resumed,
                      session_factory=session, artifact_policy=ArtifactPolicy("phase-e", config.sha256, implementation_hash()))
        publish_round_index(config, attempt.directory, summary["final_checkpoint"])
        return attempt.finish({**summary, "condition": "clean", "purpose": "calibration_fit" if seed in config.values["fit_seeds"] else "calibration_audit"})

"""One scoped NSGA-II run; all data and calibration gates remain mandatory."""
from edgefl.contracts.phase_e import ArtifactPolicy
from edgefl.optimization.storage import (stage, load_stage, checked, reference, artifact, bundle,
                                        implementation_hash, publish_round_index)
from edgefl.trust import storage as e
from edgefl.data.storage import read_json

def execute(config, prepared_path, plan_path, resume=None):
    memo = {}
    prepared_f = load_stage(config, prepared_path, "prepare-phase-f", memo=memo)
    context = bundle(config, prepared_f, memo)
    attack = e.load_stage(config.trust, plan_path, "plan-attacks", memo=memo)
    if attack["inputs"]["metadata"] != reference(config.workspace, context["paths"]["metadata"]):
        raise ValueError("Attack metadata mismatch")
    plan = read_json(artifact(config, attack, "attack_plan.json"))
    if plan["seed"] != context["seed"] or plan["coverage_status"] != "PASS":
        raise ValueError("Attack seed/coverage mismatch")
    if plan["condition"] != "clean" and config.learning.values["rounds"] < plan["activation_round"]:
        raise ValueError("Attack never activates")
    from edgefl.learning import checkpoints
    from edgefl.learning.engine import run
    from edgefl.learning.data import LearningData
    from edgefl.trust.metadata import Metadata
    from edgefl.optimization.session import EvolutionSession
    from edgefl.optimization.diagnostics import environment
    inputs = {"prepared_f": prepared_path, "attack_plan": plan_path}
    bindings = {k: reference(config.workspace, p) for k, p in inputs.items()}
    resumed = None
    options = {"seed": context["seed"], "method": "nsga2", "condition": plan["condition"]}
    if resume:
        marker = read_json(resume)
        if (marker.get("schema_version") != "phase-f.resume.v1" or marker.get("configuration_sha256") != config.sha256
                or marker.get("implementation_sha256") != implementation_hash() or marker.get("identity") != prepared_f["identity"]
                or marker.get("gate") != prepared_f["gate"] or marker.get("inputs") != bindings):
            raise ValueError("Invalid/stale Phase F resume dependencies")
        resumed = (checkpoints.load(checked(config.workspace, marker["checkpoint"])), marker)
        options["resume"] = reference(config.workspace, resume)
    metadata = Metadata(config.workspace, context["metadata"])
    def session(*args):
        return EvolutionSession(*args, settings=config.trust.values, metadata=metadata, plan=plan,
                                calibration=context["calibration"], bindings=bindings, search_settings=config.values)
    with stage(config, "run-evolution-federated", prepared_f["identity"], checked(config.workspace, prepared_f["gate"]), inputs, options) as attempt:
        from edgefl.data.storage import write_json
        write_json(attempt.directory / "search_runtime.json", environment())
        summary = run(config.learning, LearningData(config.learning, context["prepared"]), context["prepared"],
                      context["initialization"], attempt, "nsga2", context["seed"], resumed, session_factory=session,
                      artifact_policy=ArtifactPolicy("phase-f", config.sha256, implementation_hash()))
        state = publish_round_index(config, attempt.directory, summary["final_checkpoint"])
        effective = state.get("trust", {}).get("effective_attacking_client_rounds", 0)
        return attempt.finish({**summary, "condition": plan["condition"],
            "attack_coverage": "PASS" if plan["condition"] == "clean" or effective else "UNSUCCESSFUL",
            "effective_attacking_client_rounds": effective})

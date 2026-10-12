"""Experiment session composition over the existing shared training engine."""
from dataclasses import asdict
from edgefl.data.storage import read_json, write_json
from edgefl.experiments.storage import stage, load_stage, checked, artifact, reference, implementation_hash
from edgefl.experiments.preparation import verify_binding, schedule
from edgefl.experiments.panels import ScopedLearningData, ScopedMetadata
from edgefl.experiments.policies import resolve
from edgefl.contracts.phase_g import ExperimentCase


def runtime_context(config, prepared):
    from edgefl.learning.data import LearningData
    base = verify_binding(config, prepared)
    panel = read_json(artifact(config, prepared, "panel.json"))
    data = ScopedLearningData(LearningData(base["f"].learning, base["context"]["prepared"]), panel)
    metadata = ScopedMetadata(base["metadata"], panel)
    return base, data, metadata


def execute(config, prepared_path, resume=None, pilot_seed=None):
    if config.values["evidence_kind"] == "research":
        from edgefl.experiments.storage import software_evidence
        report = software_evidence(config.workspace / "reports/generated/phase_g_tests.json")
        if "fixture" in str(report.get("evidence_kind", "")):
            raise ValueError("Fixture software declarations cannot authorize research execution")
    prepared = load_stage(config, prepared_path, "prepare-experiment")
    if pilot_seed is None and prepared["summary"]["status"] != "READY":
        raise ValueError("Experiment blocked by missing policy evidence or clean review")
    if pilot_seed is not None and (pilot_seed not in (101, 102, 103) or not prepared["summary"]["policy_pilots_permitted"]):
        raise ValueError("Policy pilot requires a registered seed and reviewed clipping threshold")
    base, data, metadata = runtime_context(config, prepared)
    f, context = base["f"], base["context"]
    case = ExperimentCase(**prepared["summary"]["case"])
    policies = read_json(artifact(config, prepared, "policies.json"))
    search_policy, trust_policy = resolve(case, policies["trust"]["clipping_threshold"])
    calibration = read_json(artifact(config, prepared, "calibration.json")) if "calibration.json" in prepared["artifacts"] else None
    if pilot_seed is not None:
        calibration = None
    from edgefl.learning import checkpoints
    from edgefl.learning.engine import run
    from edgefl.learning.models import initialized
    from edgefl.trust.config import training_configuration
    from edgefl.trust.session import TrustSession
    from edgefl.optimization.session import EvolutionSession
    from edgefl.optimization.diagnostics import environment
    from edgefl.contracts.phase_e import ArtifactPolicy
    from edgefl.contracts.records import ArtifactRef, Partition, PartitionRef
    from edgefl.attacks.plan import build_plan
    seed = pilot_seed or case.seed
    method = "fedavg" if pilot_seed else case.method
    is_search = method in ("nsga2", "random")
    training = training_configuration(f.trust, seed) if pilot_seed else f.learning
    plan = build_plan(metadata, f.trust.values, seed, "clean") if pilot_seed else read_json(artifact(config, prepared, "attack_plan.json"))
    participation = schedule(plan["clients"], seed, training.values["rounds"], case.participation)
    if not pilot_seed and participation != read_json(artifact(config, prepared, "schedule.json")):
        raise ValueError("Frozen participation schedule changed")
    inputs = {"prepared": prepared_path}
    bindings = {"prepared": reference(config.workspace, prepared_path), "pilot_seed": pilot_seed,
                "search_policy": asdict(search_policy), "trust_policy": asdict(trust_policy)}
    resumed, options = None, {"case_id": case.case_id, "seed": seed, "method": method, "pilot_seed": pilot_seed}
    if resume:
        marker = read_json(resume)
        if (marker.get("schema_version") != "phase-g.resume.v1" or marker.get("configuration_sha256") != config.sha256
                or marker.get("implementation_sha256") != implementation_hash() or marker.get("identity") != base["identity"]
                or marker.get("gate") != base["gate"] or marker.get("inputs") != {"prepared": bindings["prepared"]}):
            raise ValueError("Stale/incompatible experiment resume marker")
        resumed = (checkpoints.load(checked(config.workspace, marker["checkpoint"])), marker)
        options["resume"] = reference(config.workspace, resume)
    stage_name = "run-experiment-pilot" if pilot_seed else "run-experiment"
    with stage(config, stage_name, inputs, options, base["identity"], base["gate"], prepared["summary"]["matrix_id"], case.case_id) as attempt:
        initial = context["initialization"]
        if pilot_seed:
            if resumed:
                initial_ref = resumed[0]["initialization"]
                checked(config.workspace, initial_ref)
            else:
                model = initialized(data.spec["dimension"], len(data.spec["labels"]), seed)
                path = attempt.directory / "initialization.pt"
                checkpoints.save(path, {"model": model.state_dict()})
                initial_ref = reference(config.workspace, path)
            initial = {"identity": base["identity"], "options": {"seed": seed}, "artifacts": {"mlp.pt": initial_ref}}
        search_settings = {**f.values, "population_size": case.population, "offspring_generations": case.generations}
        trusted_ref = PartitionRef(ArtifactRef(**reference(config.workspace, artifact(config, prepared, "panel.json"))), Partition.TRUSTED)
        def factory(*args):
            args = list(args)
            if method == "adaptive_capped":
                args[5] = "adaptive"
            kwargs = dict(settings=f.trust.values, metadata=metadata, plan=plan, calibration=calibration,
                          bindings=bindings, policy=trust_policy, participation_schedule=participation, trusted_binding=trusted_ref)
            if is_search:
                return EvolutionSession(*args, **kwargs, search_settings=search_settings, search_policy=search_policy)
            return TrustSession(*args, **kwargs)
        write_json(attempt.directory / "search_runtime.json", environment())
        write_json(attempt.directory / "execution_policy.json", bindings)
        write_json(attempt.directory / "calibration_scope.json", read_json(artifact(config, prepared, "calibration_scope.json")))
        summary = run(training, data, context["prepared"], initial, attempt, method, seed, resumed, session_factory=factory,
                      artifact_policy=ArtifactPolicy("phase-g", config.sha256, implementation_hash()))
        state = checkpoints.load(checked(config.workspace, summary["final_checkpoint"])).get("phase_state", {})
        trust_state = state.get("trust", state)
        rounds = trust_state.get("round_artifacts", {})
        if is_search:
            searches = state.get("search_artifacts", {})
            if set(rounds) != set(searches):
                raise ValueError("Search evidence incomplete")
            rounds = {r: {**refs, "search": searches[r]} for r, refs in rounds.items()}
        write_json(attempt.directory / "round_index.json", {"schema_version": "phase-g.round-index.v1", "rounds": rounds})
        effective = trust_state.get("effective_attacking_client_rounds", 0)
        return attempt.finish({**summary, "case": asdict(case), "case_id": case.case_id,
            "matrix_id": prepared["summary"]["matrix_id"], "condition": plan["condition"],
            "attack_coverage": "PASS" if plan["condition"] == "clean" or effective else "UNSUCCESSFUL",
            "effective_attacking_client_rounds": effective, "pilot_seed": pilot_seed,
            "scope": read_json(artifact(config, prepared, "calibration_scope.json")),
            "final_test_evaluated": False})


def fit_clipping(config, prepared_path):
    import numpy as np
    import math
    from edgefl.trust.storage import load_stage as load_e, artifact as e_artifact
    from edgefl.learning import checkpoints
    from edgefl.learning.metrics import norm
    prepared = load_stage(config, prepared_path, "prepare-experiment")
    base = verify_binding(config, prepared)
    calibration_stage = load_e(base["f"].trust, base["context"]["paths"]["calibration"], "calibrate-risk")
    inputs, norms = {"prepared": prepared_path}, {}
    for seed, name in ((101, "fit_101"), (102, "fit_102"), (103, "audit_103")):
        path = checked(config.workspace, calibration_stage["inputs"][name])
        pilot = load_e(base["f"].trust, path, "run-clean-pilot")
        inputs[name] = path
        samples = []
        for refs in read_json(e_artifact(base["f"].trust, pilot, "round_index.json"))["rounds"].values():
            record = read_json(checked(config.workspace, refs["assessment"]))
            if record["invalid"]:
                raise ValueError("Rejected clean submissions cannot fit clipping")
            samples.extend(norm(checkpoints.load(checked(config.workspace, u["delta"]))["delta"]) for u in record["updates"])
        if not samples or any(not math.isfinite(x) for x in samples):
            raise ValueError("Invalid clean norm samples")
        norms[seed] = samples
    threshold = float(np.quantile(norms[101] + norms[102], .95, method="linear"))
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("UNSUCCESSFUL: nonpositive/nonfinite clipping threshold")
    with stage(config, "fit-experiment-clipping", inputs, identity=base["identity"], gate=base["gate"],
               matrix_id=prepared["summary"]["matrix_id"], case_id=prepared["summary"]["case_id"]) as attempt:
        result = {"correctness": "PASS", "threshold": threshold, "quantile": .95, "quantile_method": "linear",
            "fit_seeds": [101, 102], "audit_seed": 103, "fit_observations": len(norms[101]) + len(norms[102]),
            "audit_observations": len(norms[103]), "audit_exceedance_rate": sum(x > threshold for x in norms[103]) / len(norms[103]),
            "upstream_configuration": base["f"].sha256, "refitted_on_audit": False}
        write_json(attempt.directory / "threshold.json", result)
        return attempt.finish(result)


def calibrate(config, prepared_path, fit_paths, audit_path):
    import numpy as np
    from edgefl.trust.calibration import fit, audit
    from edgefl.trust.risk import calibrated_risk
    prepared = load_stage(config, prepared_path, "prepare-experiment")
    base = verify_binding(config, prepared)
    scope = read_json(artifact(config, prepared, "calibration_scope.json"))
    records, inputs = {}, {"prepared": prepared_path}
    if len(fit_paths) != 2:
        raise ValueError("Exactly two clean fitting pilots required")
    for path in [*fit_paths, audit_path]:
        pilot = load_stage(config, path, "run-experiment-pilot")
        seed = pilot["summary"]["pilot_seed"]
        if (seed in records or pilot["summary"]["correctness"] != "PASS" or pilot["summary"]["condition"] != "clean"
                or pilot["summary"]["scope"] != scope or pilot["summary"]["completed_steps"] != base["f"].learning.values["rounds"]):
            raise ValueError("Invalid/incomplete or mismatched clean policy pilot")
        if (seed == 103) != (path.resolve() == audit_path.resolve()):
            raise ValueError("Audit data cannot fit calibration")
        samples = []
        for refs in read_json(artifact(config, pilot, "round_index.json"))["rounds"].values():
            observation = read_json(checked(config.workspace, refs["assessment"]))
            if observation["invalid"]:
                raise ValueError("Calibration includes invalid clean submissions")
            samples.extend({"client": c, "round": observation["observation"]["round_id"], "raw": r["raw_risk"]}
                           for c, r in observation["observation"]["clients"].items())
        records[seed] = samples
        inputs[str(seed)] = path
    if set(records) != {101, 102, 103}:
        raise ValueError("Calibration requires independent seeds 101/102/103")
    fitted = fit({s: records[s] for s in (101, 102)}, scope)
    signals = tuple(scope["signals"])
    scores = [calibrated_risk(r["raw"], fitted, signals)[1] for s in (101, 102) for r in records[s]]
    fitted["flag_threshold"] = float(np.quantile(scores, .99, method="linear"))
    fitted["active_signals"] = list(signals)
    report = audit(fitted, records[103], base["metadata"].spec["profiles"], base["metadata"].spec["declared_specialist_scenario"], signals=signals)
    with stage(config, "calibrate-experiment", inputs, identity=base["identity"], gate=base["gate"],
               matrix_id=prepared["summary"]["matrix_id"], case_id=prepared["summary"]["case_id"]) as attempt:
        write_json(attempt.directory / "calibration.json", fitted)
        write_json(attempt.directory / "audit.json", report)
        return attempt.finish({"correctness": "PASS", "scope": scope, "audit": report,
                               "fit_observations": len(scores), "review_required": True})

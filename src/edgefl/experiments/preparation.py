"""Bind cases to current A-F acceptance and explicit derived resource evidence."""
from dataclasses import asdict
from pathlib import Path
from edgefl.config import workspace_path
from edgefl.data.storage import read_json, write_json
from edgefl.contracts.phase_g import ExperimentCase, digest
from edgefl.experiments.storage import stage, load_stage, artifact, checked, reference
from edgefl.experiments.panels import select_panel


def base_context(config, binding, case, memo=None):
    from edgefl.optimization.config import load
    from edgefl.optimization.storage import load_stage as load_f, bundle
    from edgefl.learning.config import phase_c
    from edgefl.client_data.config import phase_b
    from edgefl.trust.metadata import Metadata
    f = load(workspace_path(config.workspace, binding["phase_f_config"]), config.workspace)
    memo = {} if memo is None else memo.setdefault(("upstream_configuration", f.sha256), {})
    accepted = load_f(f, workspace_path(config.workspace, binding["phase_f_acceptance"]), "validate-phase-f", memo=memo)
    if accepted["summary"].get("status") != "PASS":
        raise ValueError("Applicable Phase F scoped acceptance required")
    prepared_f = load_f(f, checked(config.workspace, accepted["inputs"]["prepared_f"]), "prepare-phase-f", memo=memo)
    context = bundle(f, prepared_f, memo)
    identity = prepared_f["identity"]
    if any(identity[k] != getattr(case, k) for k in ("task", "scenario", "variant")) or context["seed"] != case.seed:
        raise ValueError("Case does not match upstream task, scenario, variant or seed")
    if identity["dataset"] != config.values["dataset"]:
        raise ValueError("Case dataset mismatch")
    c, b = phase_c(f.learning), phase_b(phase_c(f.learning))
    metadata = Metadata(config.workspace, context["metadata"])
    if len(metadata.spec["profiles"]) != case.clients or c.values["clients"] != case.clients:
        raise ValueError("Client count differs from accepted ownership")
    for name in ("participation", "local_epochs"):
        if f.learning.values[name] != getattr(case, name):
            raise ValueError("Case training settings differ from accepted upstream scope")
    for name in ("compromised_fraction", "label_fraction", "update_scale"):
        expected = getattr(case, name)
        if name == "compromised_fraction" and not expected:
            expected = .2
        if f.trust.values[name] != expected:
            raise ValueError("Case attack settings differ from accepted upstream scope")
    if f.trust.values["reputation_retention"] != case.retention:
        raise ValueError("Case reputation setting differs from upstream calibration scope")
    if config.values["evidence_kind"] == "research":
        if c.values["seed"] != case.seed or b.values["seed"] != 42:
            raise ValueError("Research requires matched partition seed and fixed global seed 42")
        if identity["fold"] != ("protocol_a" if case.fold == "A" else case.fold):
            raise ValueError("Case capture fold mismatch")
        expected = {"rounds": 100, "centralized_epochs": 100, "batch_size": 256, "learning_rate": .001,
                    "hidden_layers": [128, 64], "fedprox_mu": .01, "trim_fraction": .2, "root_epochs": 1}
        if any(f.learning.values[k] != v for k, v in expected.items()) or f.trust.values["activation_round"] != case.activation_round:
            raise ValueError("Research budget or onset differs from frozen specification")
        evidence = read_json(checked(config.workspace, accepted["summary"]["software_evidence"]))
        if "fixture" in str(evidence.get("evidence_kind", "")):
            raise ValueError("Fixture evidence cannot authorize research")
    return {"f": f, "accepted": accepted, "prepared_f": prepared_f, "context": context,
            "metadata": metadata, "identity": identity, "gate": prepared_f["gate"]}


def calibration_scope(base, panel, signals, threshold):
    return {"configuration": base["f"].sha256, "metadata": base["context"]["metadata"]["artifacts"]["metadata.sqlite"],
            "panel_id": panel["panel_id"], "signals": list(signals), "clipping_threshold": threshold}


def schedule(clients, seed, rounds, participation):
    import math
    import torch
    from edgefl.reproducibility import derive_seed
    ids = sorted(clients)
    result = {"1": ids}
    for current in range(2, rounds + 1):
        generator = torch.Generator().manual_seed(derive_seed(seed, "participation", round_id=current))
        n = max(1, math.ceil(len(ids) * participation))
        result[str(current)] = sorted(ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()[:n])
    if any(len(row) < 3 for row in result.values()):
        raise ValueError("UNSUCCESSFUL: participation leaves fewer than three scored clients")
    return result


def matching_identity(base, panel, participation, plan):
    context = base["context"]
    return {"data": context["prepared"]["artifacts"]["data.json"]["sha256"],
            "membership": context["prepared"]["artifacts"]["learning_rows.csv"]["sha256"],
            "initialization": context["initialization"]["artifacts"]["mlp.pt"]["sha256"],
            "panel": digest(panel), "schedule": digest(participation), "attack": digest(plan)}


def require_clean_inputs(config, subject, matching, memo=None):
    """A clean review must cover the actual data and initialization being reused."""
    clean = load_stage(config, checked(config.workspace, subject["inputs"]["prepared"]), "prepare-experiment", memo=memo)
    if any(clean["summary"]["matching"][key] != value for key, value in matching.items() if key != "attack"):
        raise ValueError("Clean learning evidence has unmatched data, initialization, panel or participation")


def prepare(config, matrix_path, case_id, binding_path):
    matrix_stage = load_stage(config, matrix_path, "compile-experiments")
    matrix = read_json(artifact(config, matrix_stage, "cases.json"))
    if case_id not in matrix["cases"]:
        raise ValueError("Case is outside compiled matrix")
    case = ExperimentCase(**matrix["cases"][case_id])
    binding = read_json(binding_path)
    required = {"phase_f_config", "phase_f_acceptance"}
    optional = {"clipping_review", "calibration_review", "clean_review"}
    if not required <= set(binding) or set(binding) - required - optional:
        raise ValueError("Invalid case evidence binding")
    base = base_context(config, binding, case)
    panel = select_panel(base["metadata"].rows("trusted"), base["metadata"].spec["profiles"], case.panel_fraction, case.missing_class)
    inputs = {"matrix": matrix_path, "binding": binding_path,
              **{k: workspace_path(config.workspace, v) for k, v in binding.items()}}
    threshold = None
    if binding.get("clipping_review"):
        from edgefl.experiments.review import accepted_subject
        subject = accepted_subject(config, inputs["clipping_review"], "fit-experiment-clipping")
        if subject["summary"]["upstream_configuration"] != base["f"].sha256:
            raise ValueError("Clipping threshold scope mismatch")
        threshold = subject["summary"]["threshold"]
    from edgefl.experiments.policies import resolve
    from dataclasses import replace
    search, trust = resolve(replace(case, intervention="full") if case.intervention == "clipping" and threshold is None else case, threshold)
    expected_scope = calibration_scope(base, panel, trust.signals, threshold)
    calibrated = base["context"]["calibration"]
    needs_calibration = case.panel_fraction != 1 or case.missing_class or trust.signals != ("magnitude", "direction", "loss") or case.intervention == "clipping"
    missing = []
    if case.intervention == "clipping" and threshold is None:
        missing.append("clipping_review")
    if needs_calibration:
        if not binding.get("calibration_review"):
            calibrated = None
            missing.append("calibration_review")
        else:
            from edgefl.experiments.review import accepted_subject
            subject = accepted_subject(config, inputs["calibration_review"], "calibrate-experiment")
            if subject["summary"]["scope"] != expected_scope:
                raise ValueError("Policy calibration scope mismatch")
            calibrated = read_json(artifact(config, subject, "calibration.json"))
    if case.condition != "clean" and not binding.get("clean_review"):
        missing.append("clean_review")
    if binding.get("clean_review"):
        from edgefl.experiments.review import accepted_subject
        subject = accepted_subject(config, inputs["clean_review"], "run-experiment")
        previous = ExperimentCase(**subject["summary"]["case"])
        from edgefl.experiments.matrix import normalized
        expected = normalized(replace(case, condition="clean"))
        if normalized(previous) != expected or previous.condition != "clean":
            raise ValueError("Clean review is not applicable to this method and policy")
    from edgefl.attacks.plan import build_plan
    settings = base["f"].trust.values
    plan = build_plan(base["metadata"], settings, case.seed, case.condition)
    if plan["coverage_status"] != "PASS":
        raise ValueError("UNSUCCESSFUL: attack plan has no effective coverage")
    participation = schedule(plan["clients"], case.seed, base["f"].learning.values["rounds"], case.participation)
    if binding.get("clean_review"):
        require_clean_inputs(config, subject, matching_identity(base, panel, participation, plan))
    with stage(config, "prepare-experiment", inputs, {"case_id": case_id}, base["identity"], base["gate"], matrix["matrix_id"], case_id) as attempt:
        for name, value in (("case.json", asdict(case)), ("panel.json", panel), ("attack_plan.json", plan),
                            ("schedule.json", participation), ("policies.json", {"search": asdict(search), "trust": asdict(trust), "clipping_pending": "clipping_review" in missing}),
                            ("calibration_scope.json", expected_scope)):
            write_json(attempt.directory / name, value)
        if calibrated is not None:
            write_json(attempt.directory / "calibration.json", calibrated)
        return attempt.finish({"status": "BLOCKED" if missing else "READY", "missing": missing, "case_id": case_id,
            "case": asdict(case), "matrix_id": matrix["matrix_id"], "upstream_configuration": base["f"].sha256,
            "matching": matching_identity(base, panel, participation, plan),
            "source_verified_gate": "PASS", "policy_pilots_permitted": "clipping_review" not in missing})


def verify_binding(config, completion, memo=None):
    binding = read_json(checked(config.workspace, completion["inputs"]["binding"]))
    for name, path in binding.items():
        if reference(config.workspace, workspace_path(config.workspace, path)) != completion["inputs"].get(name):
            raise ValueError("Changed experiment binding")
    case = ExperimentCase(**completion["summary"]["case"])
    base = base_context(config, binding, case, memo)
    if base["identity"] != completion["identity"] or base["gate"] != completion["gate"]:
        raise ValueError("Prepared experiment source scope mismatch")
    actual = read_json(artifact(config, completion, "panel.json"))
    expected = select_panel(base["metadata"].rows("trusted"), base["metadata"].spec["profiles"], case.panel_fraction, case.missing_class)
    if actual != expected:
        raise ValueError("Trusted subset differs from the declared policy")
    from edgefl.attacks.plan import build_plan
    plan = build_plan(base["metadata"], base["f"].trust.values, case.seed, case.condition)
    participation = schedule(plan["clients"], case.seed, base["f"].learning.values["rounds"], case.participation)
    if (read_json(artifact(config, completion, "attack_plan.json")) != plan
            or read_json(artifact(config, completion, "schedule.json")) != participation
            or completion["summary"]["matching"] != matching_identity(base, expected, participation, plan)):
        raise ValueError("Declared matched experiment inputs were altered")
    matrix_record = load_stage(config, checked(config.workspace, completion["inputs"]["matrix"]), "compile-experiments", memo=memo)
    matrix = read_json(artifact(config, matrix_record, "cases.json"))
    if (matrix["cases"].get(case.case_id) != asdict(case) or read_json(artifact(config, completion, "case.json")) != asdict(case)
            or completion["summary"]["case_id"] != case.case_id):
        raise ValueError("Case differs from its compiled specification")
    from edgefl.experiments.review import accepted_subject
    from edgefl.experiments.policies import resolve
    from edgefl.config import canonical_json
    from dataclasses import replace
    threshold, missing = None, []
    if binding.get("clipping_review"):
        subject = accepted_subject(config, checked(config.workspace, completion["inputs"]["clipping_review"]), "fit-experiment-clipping", memo)
        if subject["summary"]["upstream_configuration"] != base["f"].sha256:
            raise ValueError("Clipping scope changed")
        threshold = subject["summary"]["threshold"]
    if case.intervention == "clipping" and threshold is None:
        missing.append("clipping_review")
    search, trust = resolve(replace(case, intervention="full") if "clipping_review" in missing else case, threshold)
    policies = {"search": asdict(search), "trust": asdict(trust), "clipping_pending": "clipping_review" in missing}
    if canonical_json(read_json(artifact(config, completion, "policies.json"))) != canonical_json(policies):
        raise ValueError("Applied policy differs from the named intervention")
    scope = calibration_scope(base, expected, trust.signals, threshold)
    if read_json(artifact(config, completion, "calibration_scope.json")) != scope:
        raise ValueError("Policy calibration identity changed")
    derived = case.panel_fraction != 1 or case.missing_class or trust.signals != ("magnitude", "direction", "loss") or case.intervention == "clipping"
    calibrated = base["context"]["calibration"]
    if derived:
        if not binding.get("calibration_review"):
            missing.append("calibration_review")
            calibrated = None
        else:
            subject = accepted_subject(config, checked(config.workspace, completion["inputs"]["calibration_review"]), "calibrate-experiment", memo)
            if subject["summary"]["scope"] != scope:
                raise ValueError("Reviewed calibration uses a different trusted resource or policy")
            calibrated = read_json(artifact(config, subject, "calibration.json"))
    if (calibrated is None) != ("calibration.json" not in completion["artifacts"]):
        raise ValueError("Unexpected or missing policy calibration")
    if calibrated is not None and read_json(artifact(config, completion, "calibration.json")) != calibrated:
        raise ValueError("Prepared calibration differs from accepted evidence")
    if case.condition != "clean" and not binding.get("clean_review"):
        missing.append("clean_review")
    if binding.get("clean_review"):
        subject = accepted_subject(config, checked(config.workspace, completion["inputs"]["clean_review"]), "run-experiment", memo)
        from edgefl.experiments.matrix import normalized
        previous = ExperimentCase(**subject["summary"]["case"])
        if previous.condition != "clean" or normalized(previous) != normalized(replace(case, condition="clean")):
            raise ValueError("Inapplicable clean learning review")
        require_clean_inputs(config, subject, completion["summary"]["matching"], memo)
    if completion["summary"]["missing"] != missing or completion["summary"]["status"] != ("BLOCKED" if missing else "READY"):
        raise ValueError("Prepared readiness disagrees with actual evidence")
    if completion["summary"]["policy_pilots_permitted"] != ("clipping_review" not in missing):
        raise ValueError("Policy pilot permission disagrees with required clipping evidence")
    return base

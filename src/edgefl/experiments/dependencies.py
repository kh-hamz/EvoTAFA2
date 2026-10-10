"""Explicit upstream scope inventory; no acceptance prerequisite is silently omitted."""
from dataclasses import asdict
from edgefl.contracts.phase_g import ExperimentCase, digest


def scope(case):
    return {k: asdict(case)[k] for k in ("task", "seed", "scenario", "fold", "variant", "clients",
            "participation", "local_epochs", "compromised_fraction", "label_fraction", "update_scale", "activation_round", "retention")}


def inventory(matrix):
    scopes = {}
    for key, raw in matrix["cases"].items():
        case = ExperimentCase(**raw)
        values = scope(case)
        # Zero means a clean plan, not an unsupported Phase E attack configuration.
        values["compromised_fraction"] = values["compromised_fraction"] or .2
        sid = digest(values)
        entry = scopes.setdefault(sid, {"scope": values, "case_ids": [],
            "requirements": ["source_verified_phase_c", "phase_d_eight_method_acceptance", "clean_pilots_101_102_103",
                "reviewed_calibration", "phase_e_21_run_acceptance", "phase_f_clean_run_and_profile_acceptance"],
            "minimum_training_runs": {"phase_d": 8, "calibration": 3, "phase_e": 21, "phase_f": 1},
            "status": "requires_bound_evidence"})
        entry["case_ids"].append(key)
    return {"schema_version": "phase-g.dependencies.v1", "scopes": scopes,
            "nominal_upstream_training_runs": sum(sum(v["minimum_training_runs"].values()) for v in scopes.values()),
            "count_note": "Before verified artifact reuse; not additional to overlapping comparison results. Policy pilots and retries are separate.",
            "policy_readiness": {key: {"calibration": raw["panel_fraction"] != 1 or raw["missing_class"] or raw["intervention"] in
                ("clipping", "no_magnitude", "no_direction", "no_loss"), "clean_review": raw["condition"] != "clean"}
                for key, raw in matrix["cases"].items()}}


def materialize_configurations(config, matrix, directory):
    """Write isolated C-F configs; accepted B scope is consumed without changing its rules."""
    from edgefl.learning.config import phase_c
    from edgefl.data.storage import write_json
    from edgefl.config import workspace_path
    requirements = inventory(matrix)
    bindings = {}
    for sid, requirement in requirements["scopes"].items():
        values = requirement["scope"]
        destination = directory / sid
        destination.mkdir(parents=True, exist_ok=False)
        paths = {name: destination / ("phase_" + name + ".json") for name in ("c", "d", "e", "f")}
        relative = {name: path.relative_to(config.workspace).as_posix() for name, path in paths.items()}
        output = workspace_path(config.workspace, config.values["output"]) / "upstream" / sid
        c = phase_c(config.search.learning).values
        c.update(seed=values["seed"], clients=values["clients"], output=(output / "c").relative_to(config.workspace).as_posix())
        d = config.search.learning.values
        d.update(phase_c_config=relative["c"], variant=values["variant"], participation=values["participation"],
                 local_epochs=values["local_epochs"], output=(output / "d").relative_to(config.workspace).as_posix())
        e = config.search.trust.values
        e.update(phase_d_config=relative["d"], compromised_fraction=values["compromised_fraction"],
                 label_fraction=values["label_fraction"], update_scale=values["update_scale"],
                 activation_round=values["activation_round"], reputation_retention=values["retention"],
                 output=(output / "e").relative_to(config.workspace).as_posix())
        f = config.search.values
        f.update(phase_e_config=relative["e"], output=(output / "f").relative_to(config.workspace).as_posix())
        for name, settings in (("c", c), ("d", d), ("e", e), ("f", f)):
            write_json(paths[name], settings)
        for case_id in requirement["case_ids"]:
            bindings[case_id] = {"scope_id": sid, "configurations": relative,
                                 "required_acceptance": "validate-phase-f", "required_upstream": requirement["requirements"]}
    return bindings

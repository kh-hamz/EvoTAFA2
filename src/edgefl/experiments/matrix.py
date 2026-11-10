"""Deterministic study expansion with explicit memberships and shared references."""
from dataclasses import asdict, replace
from collections import Counter
from edgefl.contracts.phase_g import ExperimentCase, METHODS, SEEDS, SCENARIOS, ATTACKS, digest, VERSION

ABLATIONS = ("no_reputation", "no_direct_risk", "no_expertise", "uniform_random",
             "no_magnitude", "no_direction", "no_loss", "without_classification_error",
             "without_benign_fpr", "without_current_risk", "without_historical_unreliability", "uncapped")
FOCUSED = ("clean", "targeted_label", "update_scaling")
COMPARATORS = ("nsga2", "fedavg", "adaptive")


def normalized(case):
    if case.compromised_fraction == 0:
        case = replace(case, condition="clean")
    if case.condition == "clean":
        case = replace(case, compromised_fraction=0., label_fraction=.3, update_scale=5., activation_round=11)
    elif case.condition not in ("targeted_label", "untargeted_label"):
        case = replace(case, label_fraction=.3)
    if case.condition != "update_scaling":
        case = replace(case, update_scale=5.)
    return case


def compile_matrix(values):
    cases, memberships = {}, []
    def add(case, suite, study, reference=None):
        case = normalized(case)
        cases[case.case_id] = asdict(case)
        ref = normalized(reference).case_id if reference else None
        if reference:
            canonical = normalized(reference)
            cases[canonical.case_id] = asdict(canonical)
        row = {"suite": suite, "study": study, "case_id": case.case_id, "reference_id": ref}
        if row not in memberships:
            memberships.append(row)
    if values.get("fixture_cases"):
        if values.get("evidence_kind") != "fixture":
            raise ValueError("Fixture matrix cannot be research evidence")
        for raw in values["fixture_cases"]:
            add(ExperimentCase(**raw), "fixture", "integration")
        result = {"schema_version": VERSION, "cases": dict(sorted(cases.items())), "memberships": memberships,
                  "counts": {"fixture": len(memberships)}, "unique_cases": len(cases), "global_split_seed": 42,
                  "deferred": ["All real-data experiments"], "capture_folds_unregistered": False}
        result["matrix_id"] = digest(result)
        return result
    if "main" in values["suites"]:
        for task in ("binary", "multiclass"):
            for seed in SEEDS:
                for method in METHODS:
                    for scenario in SCENARIOS:
                        add(ExperimentCase(task, seed, method, scenario=scenario), "main", "clean")
                    for attack in ATTACKS:
                        add(ExperimentCase(task, seed, method, condition=attack), "main", attack)
    for task in ("binary", "multiclass"):
        for seed in SEEDS:
            if "ablations" in values["suites"]:
                for condition in FOCUSED:
                    base = ExperimentCase(task, seed, "nsga2", condition=condition)
                    for variant in ABLATIONS:
                        add(replace(base, intervention=variant), "ablations", variant, base)
                    for method in ("adaptive_capped", "random"):
                        add(replace(base, method=method), "ablations", method, base)
                    for method in METHODS:
                        reference = replace(base, method=method)
                        add(replace(reference, variant="base"), "ablations", "features", reference)
                        add(replace(reference, intervention="clipping"), "ablations", "clipping", reference)
            if "sensitivity" in values["suites"]:
                factors = {
                    "compromised_fraction": ((0., .1, .2, .3, .4), ("targeted_label", "update_scaling")),
                    "label_fraction": ((.1, .3, .5), ("targeted_label",)),
                    "update_scale": ((2., 5., 10.), ("update_scaling",)),
                    "clients": ((10, 20, 50), FOCUSED), "participation": ((.5, 1.), FOCUSED),
                    "local_epochs": ((1, 3, 5), FOCUSED), "retention": ((.5, .9, .99), (*FOCUSED, "intermittent_sign")),
                    "panel_fraction": ((.25, .5, 1.), FOCUSED), "population": ((12, 24, 48), FOCUSED),
                    "generations": ((4, 8, 16), FOCUSED), "scenario": (SCENARIOS, FOCUSED),
                    "activation_round": ((11, 51), ("targeted_label", "update_scaling", "intermittent_sign")),
                    "missing_class": ((True,), FOCUSED)}
                for factor, (settings, conditions) in factors.items():
                    for method in COMPARATORS:
                        if factor in ("population", "generations") and method != "nsga2":
                            continue
                        if factor == "retention" and method == "fedavg":
                            continue
                        for condition in conditions:
                            base = ExperimentCase(task, seed, method, condition=condition)
                            for setting in settings:
                                add(replace(base, **{factor: setting}), "sensitivity", factor, base)
    if "capture" in values["suites"]:
        for fold in values["capture_folds"]:
            if fold == "A":
                raise ValueError("Capture folds cannot contain the Protocol A slot")
            for seed in SEEDS:
                for method in METHODS:
                    add(ExperimentCase("binary", seed, method, fold=fold), "capture", "prepare_only")
    memberships.sort(key=lambda x: (x["suite"], x["study"], x["case_id"]))
    payload = {"schema_version": VERSION, "cases": dict(sorted(cases.items())), "memberships": memberships,
               "counts": dict(Counter(row["suite"] for row in memberships)),
               "unique_cases": len(cases), "global_split_seed": 42,
               "deferred": ["CICIoT2023 replication", "locked capture scoring in Phase H"],
               "capture_folds_unregistered": "capture" in values["suites"] and not values["capture_folds"]}
    payload["matrix_id"] = digest(payload)
    return payload

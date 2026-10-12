"""Development-only matched reports with explicit incomplete pairs and undefined metrics."""
from collections import defaultdict
from dataclasses import asdict
import math
import statistics
from edgefl.config import workspace_path
from edgefl.data.storage import read_json, write_json
from edgefl.contracts.phase_g import SEEDS, digest
from edgefl.experiments.storage import load_stage, artifact, checked, stage, reference


def statistics_summary(values):
    values = list(values)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite development observation")
    n = len(values)
    result = {"n": n, "values": values, "mean": statistics.mean(values) if n else None,
              "sample_std": statistics.stdev(values) if n > 1 else None, "ci95": None,
              "interval_reason": "fewer_than_two_pairs" if n < 2 else None}
    if n > 1:
        from scipy.stats import t
        radius = float(t.ppf(.975, n - 1)) * result["sample_std"] / math.sqrt(n)
        result["ci95"] = [result["mean"] - radius, result["mean"] + radius]
    return result


def paired(left, right):
    seeds = sorted(set(left) & set(right))
    return {"seeds": seeds, "missing_seeds": sorted(set(SEEDS) - set(seeds)), "complete_five_seeds": set(seeds) == set(SEEDS),
            **statistics_summary(left[s] - right[s] for s in seeds)}


def round_seconds(row):
    timing = row.get("timing", {})
    if "round_elapsed" in timing:
        return timing["round_elapsed"]
    return (timing.get("client_training_cumulative", 0.) + timing.get("root_training", 0.)
            + timing.get("aggregation_including_aggregate_checkpoint", 0.) + sum(timing.get("evaluation", {}).values())
            + row.get("assessment", {}).get("seconds", 0.))


def run_record(config, path):
    run = load_stage(config, path, "run-experiment")
    summary = run["summary"]
    prepared = load_stage(config, checked(config.workspace, run["inputs"]["prepared"]), "prepare-experiment")
    from edgefl.experiments.preparation import verify_binding
    base = verify_binding(config, prepared)
    context = base["context"]
    history = read_json(artifact(config, run, "history.json"))
    chosen = next((r for r in history if summary.get("best") and r["step"] == summary["best"]["step"]), None)
    successful = summary["correctness"] == "PASS" and summary["attack_coverage"] == "PASS"
    rounds = []
    index = read_json(artifact(config, run, "round_index.json"))["rounds"]
    counts = {c: p["count"] for c, p in base["metadata"].spec["profiles"].items()}
    for row in history:
        if not row["step"]:
            continue
        refs = index[str(row["step"])]
        simulator = read_json(checked(config.workspace, refs["simulator"]))
        weights = row.get("weights")
        truth = simulator["events"]
        malicious = None if weights is None else sum(w for c, w in weights.items() if truth[c]["compromised"])
        honest = None
        if weights is not None:
            total = sum(counts[c] for c in weights)
            honest = {c: {"weight": weights[c], "relative_to_sample_share": weights[c] / (counts[c] / total)}
                      for c in weights if not truth[c]["compromised"]}
        rounds.append({"step": row["step"], "metrics": row["metrics"], "global_update": row.get("global_update"),
            "weight_change": row.get("weight_change"), "malicious_aggregate_weight": malicious,
            "weight_metric_reason": "coordinate_wise_method" if weights is None else None,
            "honest_influence": honest, "risk_diagnostics": simulator["risk_diagnostics"],
            "timing": row.get("timing"), "search": row.get("search"), "memory": row.get("memory"),
            "communication": {"submitted_delta_bytes": sum(r.get("transmitted_bytes", 0) for r in row.get("client_reports", {}).values()),
                              "meaning": "Recorded client submissions, not total network protocol overhead"}, "attack_events": truth})
    return {"case": summary["case"], "case_id": summary["case_id"], "run": reference(config.workspace, path),
            "status": "complete" if successful else "failed", "failure": summary.get("failure"),
            "attack_coverage": summary["attack_coverage"], "effective_attacking_client_rounds": summary["effective_attacking_client_rounds"],
            "initial": history[0]["metrics"], "selected": chosen["metrics"] if successful and chosen else None,
            "selected_step": chosen["step"] if chosen else None, "last": history[-1]["metrics"], "rounds": rounds,
            "runtime_seconds": sum(round_seconds(r) for r in history),
            "matching": {"data": context["prepared"]["artifacts"]["data.json"]["sha256"],
                "membership": context["prepared"]["artifacts"]["learning_rows.csv"]["sha256"],
                "initialization": context["initialization"]["artifacts"]["mlp.pt"]["sha256"],
                "panel": prepared["artifacts"]["panel.json"]["sha256"],
                "schedule": prepared["artifacts"]["schedule.json"]["sha256"],
                "attack": prepared["artifacts"]["attack_plan.json"]["sha256"]}}


def assert_matched(a, b, feature_study=False, panel_study=False, attack_study=False):
    fields = {"membership", "schedule"}
    if not feature_study:
        fields |= {"data", "initialization"}
    if not panel_study:
        fields.add("panel")
    if not attack_study:
        fields.add("attack")
    if any(a["matching"][k] != b["matching"][k] for k in fields):
        raise ValueError("Comparison mixes unmatched data, initialization, panel, participation or attack inputs")


def summarize(config, matrix_path, runs_path):
    compiled = load_stage(config, matrix_path, "compile-experiments")
    matrix = read_json(artifact(config, compiled, "cases.json"))
    registry = read_json(runs_path)
    if not isinstance(registry, dict) or set(registry) - set(matrix["cases"]):
        raise ValueError("Run registry has unknown cases")
    results, inputs = {}, {"matrix": matrix_path, "registry": runs_path}
    for key, raw in registry.items():
        path = workspace_path(config.workspace, raw)
        record = run_record(config, path)
        if record["case_id"] != key or record["case"] != matrix["cases"][key]:
            raise ValueError("Run does not match compiled case")
        results[key] = record
        inputs[key] = path
    comparisons = defaultdict(lambda: {"left": {}, "right": {}})
    for membership in matrix["memberships"]:
        key, ref = membership["case_id"], membership["reference_id"]
        if not ref or key == ref:
            continue
        raw = matrix["cases"][key]
        group = {k: v for k, v in raw.items() if k != "seed"}
        group.update(suite=membership["suite"], study=membership["study"])
        identity = digest(group)
        comparisons[identity]["identity"] = group
        if key in results and ref in results and results[key]["selected"] and results[ref]["selected"]:
            # OFAT partition/local-training changes intentionally create new membership.
            study = membership["study"]
            if study not in ("clients", "scenario", "participation", "local_epochs"):
                assert_matched(results[key], results[ref], feature_study=study == "features",
                               panel_study=study in ("panel_fraction", "missing_class"),
                               attack_study=study in ("compromised_fraction", "label_fraction", "update_scale", "activation_round"))
            comparisons[identity]["left"][raw["seed"]] = results[key]["selected"]["selection_validation"]
            comparisons[identity]["right"][raw["seed"]] = results[ref]["selected"]["selection_validation"]
    main = defaultdict(dict)
    for membership in matrix["memberships"]:
        if membership["suite"] == "main":
            key = membership["case_id"]
            raw = matrix["cases"][key]
            group = digest({k: v for k, v in raw.items() if k not in ("seed", "method")})
            main[group][(raw["method"], raw["seed"])] = key
    for group, mapping in main.items():
        methods = sorted({method for method, seed in mapping} - {"nsga2"})
        for method in methods:
            cid = group + ":nsga2-minus-" + method
            comparisons[cid]["identity"] = {"suite": "main", "group": group, "comparison": "nsga2-minus-" + method}
            for seed in SEEDS:
                a, b = mapping.get(("nsga2", seed)), mapping.get((method, seed))
                if a in results and b in results and results[a]["selected"] and results[b]["selected"]:
                    assert_matched(results[a], results[b])
                    comparisons[cid]["left"][seed] = results[a]["selected"]["selection_validation"]
                    comparisons[cid]["right"][seed] = results[b]["selected"]["selection_validation"]
    statistics_output = {}
    for key, item in comparisons.items():
        statistics_output[key] = {"identity": item["identity"], **{metric: paired(
            {s: v[metric] for s, v in item["left"].items() if v[metric] is not None},
            {s: v[metric] for s, v in item["right"].items() if v[metric] is not None}) for metric in ("macro_f1", "benign_fpr")}}
    groups = defaultdict(dict)
    degradation = {}
    from dataclasses import replace
    from edgefl.contracts.phase_g import ExperimentCase
    from edgefl.experiments.matrix import normalized
    for key, record in results.items():
        group = digest({k: v for k, v in record["case"].items() if k != "seed"})
        if record["selected"]:
            groups[group][record["case"]["seed"]] = record["selected"]["selection_validation"]
        if record["case"]["condition"] != "clean":
            clean_id = normalized(replace(ExperimentCase(**record["case"]), condition="clean")).case_id
            clean = results.get(clean_id)
            if record["selected"] and clean and clean["selected"]:
                assert_matched(record, clean, attack_study=True)
                a, b = clean["selected"]["selection_validation"], record["selected"]["selection_validation"]
                degradation[key] = {"clean_reference": clean_id, "macro_f1_drop": a["macro_f1"] - b["macro_f1"],
                                    "benign_fpr_increase": b["benign_fpr"] - a["benign_fpr"]}
            else:
                degradation[key] = {"clean_reference": clean_id, "reason": "missing_successful_clean_or_poisoned_pair"}
    group_statistics = {key: {"seeds": sorted(observations), "missing_seeds": sorted(set(SEEDS) - set(observations)),
        **{metric: statistics_summary(v[metric] for v in observations.values() if v[metric] is not None)
           for metric in ("macro_f1", "benign_fpr")}} for key, observations in groups.items()}
    missing = sorted(set(matrix["cases"]) - set(results))
    report = {"schema_version": "phase-g.development-report.v1", "matrix_id": matrix["matrix_id"],
              "role": "selection_validation", "checkpoint_rule": "max_macro_f1_then_min_fpr_then_earliest_round",
              "missing_cases": missing, "failed_cases": [k for k, v in results.items() if v["status"] != "complete"],
              "results": results, "paired_comparisons": statistics_output, "method_seed_summaries": group_statistics,
              "clean_to_poisoned_degradation": degradation,
              "interpretation": "Exploratory development intervals include partition/training variation conditional on the fixed global split.",
              "directions": {"macro_f1_difference": "positive favors left", "benign_fpr_difference": "positive favors right"},
              "final_test_evaluated": False}
    with stage(config, "summarize-experiments", inputs, matrix_id=matrix["matrix_id"]) as attempt:
        write_json(attempt.directory / "development.json", report)
        write_json(attempt.directory / "coverage.json", {k: results[k]["status"] if k in results else "missing" for k in matrix["cases"]})
        return attempt.finish({"matrix_id": matrix["matrix_id"], "status": "INCOMPLETE" if missing or report["failed_cases"] else "COMPLETE",
                               "completed": len(results) - len(report["failed_cases"]), "required": len(matrix["cases"]),
                               "final_test_evaluated": False})

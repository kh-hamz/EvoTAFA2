"""Actual search replay and explicit, workload-scoped sequential cost projections."""
import time
from collections import defaultdict
from edgefl.config import workspace_path, canonical_json
from edgefl.data.storage import read_json, write_json
from edgefl.contracts.phase_g import digest, ExperimentCase
from edgefl.experiments.storage import load_stage, artifact, checked, reference, stage
from edgefl.experiments.sessions import runtime_context
from edgefl.experiments.reporting import round_seconds


def workload(config, prepared):
    base, data, metadata = runtime_context(config, prepared)
    case = dict(prepared["summary"]["case"])
    case.pop("seed")
    signature = {"case": case, "features": data.spec["dimension"], "classes": len(data.spec["labels"]),
                 "trusted_observations": len(data.view("trusted")), "device": base["f"].learning.values["device"]}
    return digest(signature), signature, sum(p["count"] for p in metadata.spec["profiles"].values())


def measure(config, run_path, current):
    source = load_stage(config, run_path, "run-experiment")
    if source["summary"]["correctness"] != "PASS" or source["summary"]["attack_coverage"] != "PASS":
        raise ValueError("Profiling requires a successful covered experiment")
    if type(current) is not int or not 1 <= current <= source["summary"]["completed_steps"]:
        raise ValueError("Invalid profiling round")
    prepared_path = checked(config.workspace, source["inputs"]["prepared"])
    prepared = load_stage(config, prepared_path, "prepare-experiment")
    base, data, metadata = runtime_context(config, prepared)
    from edgefl.learning.models import configure
    from edgefl.optimization.diagnostics import environment
    if (configure(base["f"].learning.values) != read_json(artifact(config, source, "runtime.json"))
            or environment() != read_json(artifact(config, source, "search_runtime.json"))):
        raise ValueError("Profiling runtime differs from the recorded execution")
    index = read_json(artifact(config, source, "round_index.json"))["rounds"]
    refs = index[str(current)]
    history = read_json(artifact(config, source, "history.json"))
    row = next(r for r in history if r["step"] == current)
    warm_seconds, warm_counts, measured, elapsed = 0., None, None, None
    signature, shape, training_count = workload(config, prepared)
    with stage(config, "measure-experiment", {"run": run_path}, {"round": current}, source["identity"], source["gate"],
               prepared["summary"]["matrix_id"], prepared["summary"]["case_id"]) as attempt:
        if "search" in refs:
            from edgefl.optimization.replay import decode
            from edgefl.optimization.aggregation import NSGA2Aggregator
            from edgefl.experiments.policies import resolve
            recorded = read_json(checked(config.workspace, refs["search"]))
            case = ExperimentCase(**source["summary"]["case"])
            policies = read_json(artifact(config, prepared, "policies.json"))
            policy, _ = resolve(case, policies["trust"]["clipping_threshold"])
            request = decode(read_json(checked(config.workspace, refs["assessment"])), case.seed)
            aggregator = NSGA2Aggregator(config.workspace, attempt.directory,
                {c: p["count"] for c, p in metadata.spec["profiles"].items()}, data.view("trusted"), request.trusted,
                data.spec["labels"], base["f"].learning.values, recorded["settings"], recorded["previous_weights"],
                base["f"].trust.values["weight_floor"], policy)
            tick = time.perf_counter()
            warm = aggregator.evaluator(request)
            warm.evaluate([1 / len(warm.clients)] * len(warm.clients))
            warm.sync()
            warm_seconds, warm_counts = time.perf_counter() - tick, dict(warm.counters)
            del warm
            tick = time.perf_counter()
            result = aggregator.aggregate(request)
            elapsed = time.perf_counter() - tick
            measured = aggregator.last_report
            fields = ("selected", "selected_objectives", "requests", "generations")
            if (result.failure_reason or canonical_json(recorded["weights"]) != canonical_json(measured["weights"])
                    or any(canonical_json(recorded["search"][n]) != canonical_json(measured["search"][n]) for n in fields)):
                raise ValueError("Actual-model profile does not reproduce recorded search")
        seconds = round_seconds(row)
        if elapsed is not None:
            seconds += elapsed - row["timing"]["aggregation_including_aggregate_checkpoint"]
        report = {"correctness": "PASS", "case_id": source["summary"]["case_id"], "matrix_id": source["summary"]["matrix_id"],
            "signature": signature, "workload": shape, "training_observations": training_count,
            "runtime": environment(), "training_runtime": read_json(artifact(config, source, "runtime.json")),
            "round_seconds": seconds, "client_training_seconds": row["timing"]["client_training_cumulative"],
            "warmup_seconds": warm_seconds, "warmup_counts": warm_counts,
            "search": measured["search"] if measured else None, "search_measured": measured is not None,
            "recorded_training_round": current, "rounds": source["summary"]["completed_steps"],
            "run_artifact_bytes": sum(checked(config.workspace, ref).stat().st_size for ref in source["artifacts"].values()),
            "nested_timings_are_not_additive": True}
        write_json(attempt.directory / "measurement.json", report)
        return attempt.finish(report)


def upstream_costs(config, prepared_paths):
    """Count each already verified upstream training attempt and physical file once."""
    seen, files, runs = set(), set(), []
    pending = []
    for path in prepared_paths:
        value = load_stage(config, path, "prepare-experiment")
        pending.append(checked(config.workspace, value["inputs"]["phase_f_acceptance"]))
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        record = read_json(path)
        if record.get("schema_version") not in ("phase-d.v1", "phase-e.v1", "phase-f.v1"):
            continue
        for ref in record.get("inputs", {}).values():
            parent = checked(config.workspace, ref)
            if parent.name == "completion.json":
                pending.append(parent)
        for ref in record["artifacts"].values():
            files.add(checked(config.workspace, ref))
        if "history.json" in record["artifacts"]:
            history = read_json(checked(config.workspace, record["artifacts"]["history.json"]))
            runs.append({"completion": reference(config.workspace, path), "stage": record["stage"],
                         "seconds": sum(round_seconds(row) for row in history),
                         "initial_evaluation_seconds": sum(history[0].get("evaluation_seconds", {}).values())})
    return {"completed_training_attempts": runs, "recorded_training_seconds": sum(r["seconds"] + r["initial_evaluation_seconds"] for r in runs),
            "unique_artifact_bytes": sum(path.stat().st_size for path in files),
            "preparation_cost": "not timed by this profiler; source audit and materialization costs remain unmeasured",
            "retries": "historical referenced attempts only; future retries excluded"}


def profile(config, matrix_path, bindings_path, measurements_path):
    from edgefl.experiments.runner import preflight
    matrix, readiness = preflight(config, matrix_path, bindings_path)
    entries = read_json(measurements_path)
    if not isinstance(entries, list) or not entries:
        raise ValueError("At least one actual completed experiment measurement required")
    inputs = {"matrix": matrix_path, "bindings": bindings_path, "measurement_manifest": measurements_path}
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        if set(entry) != {"run", "round"}:
            raise ValueError("Measurement requires run and round")
        run_path = workspace_path(config.workspace, entry["run"])
        result_path = measure(config, run_path, entry["round"])
        measurement = load_stage(config, result_path, "measure-experiment")
        if measurement["summary"]["matrix_id"] != matrix["matrix_id"]:
            raise ValueError("Profile mixes experimental matrices")
        groups[measurement["summary"]["signature"]].append(measurement["summary"])
        inputs[f"measurement_{i}"] = result_path
    projected, prepared_refs, paths = {}, {}, []
    for key, info in readiness.items():
        if info["status"] != "ready":
            continue
        path = checked(config.workspace, info["prepared"])
        prepared = load_stage(config, path, "prepare-experiment")
        signature, shape, count = workload(config, prepared)
        samples = groups.get(signature)
        if not samples:
            continue
        predictions = []
        for sample in samples:
            round_cost = sample["round_seconds"] + sample["client_training_seconds"] * (count / sample["training_observations"] - 1)
            predictions.append(round_cost * sample["rounds"] + sample["warmup_seconds"])
        projected[key] = {"seconds": sum(predictions) / len(predictions), "reference_predictions": predictions,
                          "artifact_bytes": sum(s["run_artifact_bytes"] for s in samples) / len(samples), "workload": shape}
        prepared_refs[key] = info["prepared"]
        paths.append(path)
    if not projected:
        raise ValueError("No ready matrix workload is covered by these profiles")
    historical = upstream_costs(config, paths)
    report = {"correctness": "PASS", "matrix_id": matrix["matrix_id"], "covered_cases": sorted(projected),
              "prepared_cases": prepared_refs, "uncovered_cases": sorted(set(matrix["cases"]) - set(projected)),
              "projections": projected, "projected_experiment_seconds": sum(v["seconds"] for v in projected.values()),
              "projected_artifact_bytes": sum(v["artifact_bytes"] for v in projected.values()),
              "verified_existing_upstream_costs": historical,
              "assumptions": ["Sequential same-device execution", "Same recorded model/panel/search workload",
                  "Client training time scales linearly with local observation count", "Seed-dependent search/cache behavior can change actual cost",
                  "Warmup separate and counted once per run", "Preparation time and future retries excluded, not assumed zero"],
              "review_required": True}
    with stage(config, "profile-experiments", inputs, matrix_id=matrix["matrix_id"]) as attempt:
        write_json(attempt.directory / "profile.json", report)
        return attempt.finish(report)

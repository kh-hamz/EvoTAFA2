"""Actual search replay with warm-up separated from the measured budget."""
import time
import torch
from edgefl.config import canonical_json
from edgefl.data.storage import write_json
from edgefl.optimization.storage import stage, checked, load_stage
from edgefl.optimization.replay import context, aggregator
from edgefl.optimization.diagnostics import environment

def execute(config, run_path, round_id, projection_runs, projection_rounds):
    if any(type(v) is not int or v <= 0 for v in (projection_runs, projection_rounds)):
        raise ValueError("Projection run/round counts must be positive integers")
    source = load_stage(config, run_path, "run-evolution-federated")
    if source["summary"]["correctness"] != "PASS":
        raise ValueError("Profiling requires a successful completed run")
    replay = context(config, checked(config.workspace, source["inputs"]["prepared_f"]), run_path, round_id)
    with stage(config, "profile-search", source["identity"], checked(config.workspace, source["gate"]),
               {"run": run_path}, {"round": round_id, "projection_runs": projection_runs,
                                  "projection_rounds": projection_rounds}) as attempt:
        aggregate = aggregator(config, attempt.directory, replay)
        tick = time.perf_counter()
        warm = aggregate.evaluator(replay["request"])
        warm.evaluate([1 / len(warm.clients)] * len(warm.clients))
        warm.sync()
        warm_seconds = time.perf_counter() - tick
        warm_counts = dict(warm.counters)
        del warm
        # aggregate creates a new evaluator/cache and a newly seeded pymoo instance.
        tick = time.perf_counter()
        result = aggregate.aggregate(replay["request"])
        measured_seconds = time.perf_counter() - tick
        measured, recorded = aggregate.last_report, replay["recorded_search"]
        fields = ("selected", "selected_objectives", "generations", "requests")
        matches = (canonical_json(measured["weights"]) == canonical_json(recorded["weights"]) and
                   all(canonical_json(measured["search"][f]) == canonical_json(recorded["search"][f]) for f in fields))
        from edgefl.learning import checkpoints
        payload = checkpoints.load(checked(config.workspace, source["summary"]["final_checkpoint"]))
        prior_round = next(r for r in payload["history"] if r["step"] == round_id)
        # Completed checkpoint history precedes post-save timers; earlier entries include elapsed time.
        elapsed = prior_round["timing"].get("round_elapsed")
        if elapsed is None:
            nonsearch = (prior_round["timing"]["client_training_cumulative"] + prior_round["timing"].get("scoring", prior_round.get("assessment", {}).get("seconds", 0.))
                         + sum(prior_round["timing"]["evaluation"].values()) + prior_round["timing"]["root_training"])
            source_timing_kind = "component_sum_without_checkpoint_io"
        else:
            nonsearch = max(0., elapsed - prior_round["timing"]["aggregation_including_aggregate_checkpoint"])
            source_timing_kind = "completed_round_elapsed_minus_aggregation"
        multiplier = projection_runs * projection_rounds
        profile = {"schema_version": "phase-f.profile.v1", "correctness": "PASS" if matches and not result.failure_reason else "FAIL",
            "matches_recorded_search": matches, "round": round_id, "runtime": environment(),
            "training_runtime": payload["environment"], "workload": measured["workload"],
            "warmup_seconds": warm_seconds, "warmup_model_evaluations": warm_counts["actual_model_evaluations"],
            "warmup_counts": warm_counts,
            "measured_aggregation_wall_seconds": measured_seconds, "counts": measured["search"]["counts"],
            "timing": measured["search"]["timing"], "timing_totals": measured["search"]["timing_totals"],
            "memory": measured["search"]["memory"], "settings": config.values,
            "projection": {"runs": projection_runs, "rounds_per_run": projection_rounds,
                "search_seconds": measured["search"]["timing_totals"]["search"] * multiplier,
                "end_to_end_seconds": (nonsearch + measured_seconds) * multiplier + warm_seconds * projection_runs,
                "source_timing_kind": source_timing_kind,
                "assumptions": "Same workload and hardware; sequential clients; warm-up once per run; no claim for unmeasured tasks.",
                "omitted": ["data preparation", "Phase E calibration pilots", "other comparison methods", "retries"],
                "nested_search_timings_are_not_additive": True}}
        write_json(attempt.directory / "profile.json", profile)
        return attempt.finish(profile)

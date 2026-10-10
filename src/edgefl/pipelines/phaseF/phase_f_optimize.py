"""Replay recorded defense inputs without advancing training or reputation."""
from edgefl.optimization.storage import stage, checked
from edgefl.optimization.replay import context, aggregator

def execute(config, prepared_path, run_path, round_id):
    replay = context(config, prepared_path, run_path, round_id)
    source = replay["source"]
    with stage(config, "optimize-round", source["identity"], checked(config.workspace, source["gate"]),
               {"prepared_f": prepared_path, "run": run_path}, {"round": round_id}) as attempt:
        result = aggregator(config, attempt.directory, replay).aggregate(replay["request"])
        return attempt.finish({"status": "FAIL" if result.failure_reason else "PASS", "failure": result.failure_reason,
                               "round": round_id, "state_advanced": False})

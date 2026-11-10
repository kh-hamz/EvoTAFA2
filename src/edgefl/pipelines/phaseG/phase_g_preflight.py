"""Publish readiness without advancing a training state."""
from edgefl.experiments.runner import preflight
from edgefl.experiments.storage import stage
from edgefl.data.storage import write_json


def execute(config, matrix_path, bindings_path):
    matrix, report = preflight(config, matrix_path, bindings_path)
    with stage(config, "preflight-experiments", {"matrix": matrix_path, "bindings": bindings_path}, matrix_id=matrix["matrix_id"]) as attempt:
        write_json(attempt.directory / "readiness.json", report)
        return attempt.finish({"matrix_id": matrix["matrix_id"], "status": "READY" if all(r["status"] == "ready" for r in report.values()) else "BLOCKED",
                               "ready": sum(r["status"] == "ready" for r in report.values()), "required": len(report)})

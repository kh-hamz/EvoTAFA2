"""Record explicit evidence-based operator decisions without overriding software failures."""
from dataclasses import asdict
from edgefl.contracts.phase_d import LearningReview
from edgefl.learning.storage import stage, load_stage, checked, write_json


def execute(config, run_path, disposition, rationale, limitations=""):
    source = load_stage(config, run_path)
    if source["stage"] not in ("train-centralized", "run-federated"):
        raise ValueError("Only a training run can receive a learning review")
    decision = LearningReview(disposition, rationale, limitations)
    if disposition != "unresolved" and source["summary"]["correctness"] != "PASS":
        raise ValueError("Learning review cannot override failed software")
    with stage(config, "review-learning", source["identity"], checked(config.workspace, source["gate"]),
               {"run": run_path}) as attempt:
        summary = {"decision": asdict(decision), "method": source["summary"]["method"],
                   "seed": source["summary"]["seed"], "evidence": source["artifacts"]}
        write_json(attempt.directory / "review.json", summary)
        return attempt.finish(summary)

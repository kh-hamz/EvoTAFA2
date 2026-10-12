"""Centralized baseline entrypoint; the common executor also serves FL orchestration."""
from edgefl.learning.storage import stage, load_stage, checked, read_json, implementation_hash
from edgefl.learning.review import prerequisites


def execute(config, prepared_path, initialization_path, method, seed, reviews=(), resume=None):
    if method not in ("majority", "logistic", "mlp", "fedavg", "fedprox", "median", "trimmed_mean", "fltrust"):
        raise ValueError("Unknown Phase D method")
    memo = {}
    prepared = load_stage(config, prepared_path, "prepare-learning-data", memo=memo)
    initialization = load_stage(config, initialization_path, "initialize-model", memo=memo)
    if checked(config.workspace, initialization["inputs"]["prepared"]) != prepared_path.resolve():
        raise ValueError("Initialization prepared-data mismatch")
    prerequisites(config, method, prepared, seed, reviews, memo=memo)
    from edgefl.learning.data import LearningData
    from edgefl.learning.engine import run as train
    from edgefl.learning.checkpoints import load
    resumed = None
    if resume:
        marker = read_json(resume)
        if marker.get("schema_version") != "phase-d.resume.v1" or marker.get("configuration_sha256") != config.sha256:
            raise ValueError("Invalid/stale resume marker")
        if marker.get("implementation_sha256") != implementation_hash():
            raise ValueError("Resume implementation mismatch")
        if marker["identity"] != prepared["identity"] or marker["gate"] != prepared["gate"]:
            raise ValueError("Resume experiment mismatch")
        for key, expected in (("prepared", prepared_path), ("initialization", initialization_path)):
            if checked(config.workspace, marker["inputs"][key]) != expected.resolve():
                raise ValueError("Resume dependency mismatch")
        resumed = (load(checked(config.workspace, marker["checkpoint"])), marker)
    name = "train-centralized" if method in ("majority", "logistic", "mlp") else "run-federated"
    inputs = {"prepared": prepared_path, "initialization": initialization_path,
              **{f"review_{i}": p for i, p in enumerate(reviews)}}
    from edgefl.learning.storage import reference
    options = {"method": method, "seed": seed}
    if resume:
        options["resume"] = reference(config.workspace, resume)
    with stage(config, name, prepared["identity"], checked(config.workspace, prepared["gate"]), inputs, options) as attempt:
        return attempt.finish(train(config, LearningData(config, prepared), prepared, initialization,
                                    attempt, method, seed, resumed))

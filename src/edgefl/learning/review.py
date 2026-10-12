"""Evidence review cannot override failed software or data acceptance."""

from edgefl.contracts.phase_d import METHODS, LearningReview
from edgefl.learning.storage import load_stage, checked


def accepted_review(config, path, memo=None):
    review = load_stage(config, path, "review-learning", memo=memo)
    inputs = review["inputs"]
    if set(inputs) != {"run"}:
        raise ValueError("Review requires one exact run")
    run = load_stage(config, checked(config.workspace, inputs["run"]), memo=memo)
    if run["stage"] not in ("train-centralized", "run-federated"):
        raise ValueError("Review does not reference a training run")
    decision = LearningReview(**review["summary"]["decision"])
    if decision.disposition == "unresolved" or run["summary"]["correctness"] != "PASS":
        raise ValueError("Unresolved learning or failed software blocks advancement")
    if review["identity"] != run["identity"] or review["summary"]["method"] != run["summary"]["method"]:
        raise ValueError("Review identity mismatch")
    return run


def prerequisites(config, method, prepared, seed, review_paths, memo=None):
    memo = {} if memo is None else memo
    if method in ("majority", "logistic", "mlp"):
        if review_paths:
            raise ValueError("Centralized sanity runs do not require upstream learning reviews")
        return
    evidence = [accepted_review(config, path, memo=memo) for path in review_paths]
    identity = prepared["identity"]
    def matches(run, same_scenario=True):
        keys = ("dataset", "task", "fold", "variant", "scenario") if same_scenario else ("dataset", "task", "fold", "variant")
        if any(run["identity"][key] != identity[key] for key in keys) or run["summary"]["seed"] != seed:
            return False
        # Different partitions may fit different transformers; the accepted global split must match.
        from edgefl.learning.storage import artifact
        from edgefl.data.storage import read_json
        left = load_stage(config, checked(config.workspace, run["inputs"]["prepared"]), "prepare-learning-data", memo=memo)
        return read_json(artifact(config, left, "data.json"))["experiment"] == read_json(artifact(config, prepared, "data.json"))["experiment"]
    if not any(r["summary"]["method"] == "mlp" and matches(r) and
               load_stage(config, checked(config.workspace, r["inputs"]["prepared"]), memo=memo)["artifacts"]["data.json"] == prepared["artifacts"]["data.json"]
               for r in evidence):
        raise ValueError("Applicable centralized MLP review required")
    if identity["scenario"] != "near_iid":
        if not any(r["summary"]["method"] == "fedavg" and r["identity"]["scenario"] == "near_iid" and matches(r, False) for r in evidence):
            raise ValueError("Applicable near-IID FedAvg review required")
    if method != "fedavg" and not any(r["summary"]["method"] == "fedavg" and matches(r) for r in evidence):
        raise ValueError("Applicable FedAvg review required")


def validate_scope(config, paths, test_report):
    from edgefl.data.storage import read_json
    from edgefl.learning.storage import implementation_hash, reference
    test_evidence = read_json(test_report)
    if (test_evidence.get("schema_version") != "phase-d.tests.v1" or test_evidence.get("successful") is not True
            or type(test_evidence.get("tests")) is not int or test_evidence["tests"] <= 0
            or test_evidence.get("failures") != 0 or test_evidence.get("errors") != 0
            or test_evidence.get("implementation_sha256") != implementation_hash()):
        raise ValueError("Passing current implementation regression evidence required")
    if not paths:
        raise ValueError("Phase D acceptance requires reviewed runs")
    memo = {}
    runs = [accepted_review(config, path, memo=memo) for path in paths]
    first = runs[0]
    if any(r["identity"] != first["identity"] or r["summary"]["seed"] != first["summary"]["seed"] for r in runs):
        raise ValueError("Acceptance scope mixes experiments or seeds")
    if any(r["inputs"]["prepared"] != first["inputs"]["prepared"] or
           r["inputs"]["initialization"] != first["inputs"]["initialization"] for r in runs):
        raise ValueError("Acceptance requires matched data and initialization artifacts")
    methods = [r["summary"]["method"] for r in runs]
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate method in acceptance scope")
    missing = sorted(set(METHODS) - set(methods))
    return {"status": "FAIL" if missing else "PASS", "missing_methods": missing,
            "scope": first["identity"], "seed": first["summary"]["seed"],
            "methods": methods, "real_data_gate": "PASS", "learning_reviews": "complete",
            "software_evidence": reference(config.workspace, test_report)}

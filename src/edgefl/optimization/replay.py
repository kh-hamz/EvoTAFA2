"""Verify and decode committed round inputs for replay and profiling."""
from edgefl.contracts.records import (ArtifactRef, Compatibility, ClientUpdate, PartitionRef, Partition,
                                     ClientAssessment, ClassExpertise, ClassSupport)
from edgefl.contracts.interfaces import AggregationRequest
from edgefl.data.storage import read_json
from edgefl.optimization.storage import load_stage, checked, artifact, reference, bundle
from edgefl.trust import storage as e
from edgefl.reproducibility import derive_seed

def decode(record, seed):
    updates = tuple(ClientUpdate(**{**u, "parent_model": ArtifactRef(**u["parent_model"]),
        "delta": ArtifactRef(**u["delta"]), "compatibility": Compatibility(**u["compatibility"])}) for u in record["updates"])
    assessments = tuple(ClientAssessment(**{**a, "expertise": tuple(ClassExpertise(**v) for v in a["expertise"]),
        "support": tuple(ClassSupport(**v) for v in a["support"]),
        "risk_components": tuple(tuple(v) for v in a["risk_components"])}) for a in record["assessments"])
    trusted = PartitionRef(ArtifactRef(**record["trusted"]["manifest"]), Partition(record["trusted"]["role"]))
    current = record["observation"]["round_id"]
    return AggregationRequest(current, ArtifactRef(**record["parent"]), Compatibility(**record["compatibility"]),
                              updates, assessments, trusted, derive_seed(seed, "evolution", round_id=current))

def context(config, prepared_path, run_path, round_id):
    memo = {}
    prepared_f = load_stage(config, prepared_path, "prepare-phase-f", memo=memo)
    inputs = bundle(config, prepared_f, memo)
    is_f = read_json(run_path).get("schema_version") == "phase-f.v1"
    if is_f:
        source = load_stage(config, run_path, "run-evolution-federated", memo=memo)
        if source["inputs"]["prepared_f"] != reference(config.workspace, prepared_path):
            raise ValueError("Replay uses a different Phase F bundle")
    else:
        source = e.load_stage(config.trust, run_path, "run-trust-federated", memo=memo)
        for name in ("metadata", "calibration", "calibration_review"):
            if source["inputs"][name] != reference(config.workspace, inputs["paths"][name]):
                raise ValueError("Replay E scope mismatch")
        if source["inputs"]["initialization"] != prepared_f["inputs"]["initialization"]:
            raise ValueError("Replay initialization mismatch")
    if source["summary"]["seed"] != inputs["seed"] or type(round_id) is not int or not 1 <= round_id <= source["summary"]["completed_steps"]:
        raise ValueError("Replay seed/round mismatch")
    from edgefl.learning import checkpoints
    from edgefl.learning.models import configure
    from edgefl.learning.data import LearningData
    from edgefl.trust.metadata import Metadata
    from edgefl.optimization.diagnostics import environment
    if configure(config.learning.values) != read_json(artifact(config, source, "runtime.json")):
        raise ValueError("Replay training runtime mismatch")
    if is_f and environment() != read_json(artifact(config, source, "search_runtime.json")):
        raise ValueError("Replay search runtime mismatch")
    index = read_json(artifact(config, source, "round_index.json"))["rounds"]
    record = read_json(checked(config.workspace, index[str(round_id)]["assessment"]))
    previous = None
    recorded_search = None
    if is_f:
        recorded_search = read_json(checked(config.workspace, index[str(round_id)]["search"]))
        previous = recorded_search["previous_weights"]
    elif round_id > 1:
        payload = checkpoints.load(checked(config.workspace, source["summary"]["final_checkpoint"]))
        previous = next(h["weights"] for h in payload["history"] if h["step"] == round_id - 1)
    data = LearningData(config.learning, inputs["prepared"])
    metadata = Metadata(config.workspace, inputs["metadata"])
    return {"source": source, "request": decode(record, inputs["seed"]), "previous": previous,
            "data": data, "counts": {c: p["count"] for c, p in metadata.spec["profiles"].items()},
            "recorded_search": recorded_search}

def aggregator(config, directory, replay):
    from edgefl.optimization.aggregation import NSGA2Aggregator
    from edgefl.learning import checkpoints
    from edgefl.contracts.records import PartitionRef, Partition
    data = replay["data"]
    trusted_ref = PartitionRef(checkpoints.artifact(config.workspace, artifact(config, data.completion, "data.json")), Partition.TRUSTED)
    return NSGA2Aggregator(config.workspace, directory, replay["counts"], data.view("trusted"), trusted_ref,
                           data.spec["labels"], config.learning.values, config.values,
                           replay["previous"], config.trust.values["weight_floor"])

"""Independent numerical aggregation and the existing Aggregator protocol adapter."""

import math
import time
import torch
from edgefl.config import workspace_path
from edgefl.contracts.records import ClientWeight, RoundResult
from edgefl.data.storage import write_json
from edgefl.learning import checkpoints
from edgefl.learning.metrics import norm, magnitude


def reduce_updates(updates, counts, method, trim_fraction=0.2, root=None):
    if not updates:
        raise ValueError("No valid client updates")
    ids = sorted(updates)
    template = updates[ids[0]]
    for state in updates.values():
        checkpoints.validate_state(state, template)
    weights, normalization = None, {}
    if method in ("fedavg", "fedprox"):
        if any(type(counts[k]) is not int or counts[k] <= 0 for k in ids):
            raise ValueError("Invalid registered training count")
        total = sum(counts[k] for k in ids)
        weights = {k: counts[k] / total for k in ids}
    elif method == "fltrust":
        if root is None:
            raise ValueError("FLTrust requires a root update")
        checkpoints.validate_state(root, template)
        root_norm = norm(root)
        if root_norm == 0:
            raise ValueError("Zero root norm")
        scores = {}
        for client in ids:
            client_norm = norm(updates[client])
            dot = sum(float(updates[client][k].double().mul(root[k].double()).sum()) for k in template)
            scores[client] = max(0.0, min(1.0, dot / (client_norm * root_norm))) if client_norm else 0.0
            normalization[client] = root_norm / client_norm if client_norm else 0.0
        total = sum(scores.values())
        if total == 0:
            raise ValueError("No positive FLTrust scores")
        weights = {k: scores[k] / total for k in ids}
    elif method not in ("median", "trimmed_mean"):
        raise ValueError("Unknown federated method")
    result = {}
    for key in template:
        stack = torch.stack([updates[c][key].double() for c in ids])
        if weights is not None:
            coefficients = torch.tensor([weights[c] * normalization.get(c, 1) for c in ids], dtype=torch.float64)
            result[key] = (stack * coefficients.reshape((-1,) + (1,) * (stack.ndim - 1))).sum(0).to(template[key].dtype)
        else:
            ordered = stack.sort(dim=0).values
            if method == "median":
                n = len(ids)
                value = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
            else:
                if not 0 < trim_fraction < 0.5:
                    raise ValueError("Invalid trimming fraction")
                k = math.floor(trim_fraction * len(ids))
                if k < 1 or len(ids) <= 2 * k:
                    raise ValueError("Insufficient clients for declared trimming")
                value = ordered[k:-k].mean(0)
            result[key] = value.to(template[key].dtype)
    checkpoints.validate_state(result, template)
    return result, weights, normalization


class BaselineAggregator:
    def __init__(self, workspace, directory, method, counts, trim_fraction=0.2, root=None):
        self.workspace, self.directory, self.method = workspace, directory, method
        self.counts, self.trim_fraction, self.root = counts, trim_fraction, root

    def aggregate(self, request):
        if request.assessments:
            raise ValueError("Phase D does not consume assessments")
        if self.method == "fltrust" and request.trusted is None:
            raise ValueError("FLTrust requires trusted resource identity")
        started = time.perf_counter()
        parent = checkpoints.load(workspace_path(self.workspace, request.parent_model.path), request.parent_model.sha256)["model"]
        states = {}
        for update in request.updates:
            if update.registered_training_count != self.counts[update.client_id]:
                raise ValueError("Unregistered client sample count")
            states[update.client_id] = checkpoints.load(workspace_path(self.workspace, update.delta.path), update.delta.sha256)["delta"]
            checkpoints.validate_state(states[update.client_id], parent)
        delta, weights, normalization = reduce_updates(states, self.counts, self.method, self.trim_fraction, self.root)
        state = {k: parent[k] + delta[k] for k in parent}
        checkpoints.validate_state(state, parent)
        path = self.directory / f"aggregate-{request.round_id:04d}.pt"
        checkpoints.save(path, {"model": state})
        diagnostics = self.directory / f"aggregation-{request.round_id:04d}.json"
        write_json(diagnostics, {"schema_version": "phase-d.aggregation.v1", "method": self.method,
                                "weights": weights, "normalization": normalization,
                                "global_update": magnitude(delta, parent), "seconds": time.perf_counter() - started})
        entries = tuple(ClientWeight(c, w) for c, w in weights.items()) if weights is not None else None
        return RoundResult(request.round_id, checkpoints.artifact(self.workspace, path), entries,
                           checkpoints.artifact(self.workspace, diagnostics), ())

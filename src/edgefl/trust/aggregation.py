"""Fixed-coefficient adaptive and frozen weighting over compatible submitted deltas."""

import math
import time
from edgefl.config import workspace_path
from edgefl.contracts.records import ClientWeight, RoundResult
from edgefl.learning import checkpoints
from edgefl.learning.metrics import magnitude
from edgefl.data.storage import write_json


def validate_weights(weights):
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1., abs_tol=1e-9, rel_tol=0):
        raise ValueError("Invalid normalized weights")


def adaptive_weights(assessments, counts, floor=1e-6):
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("Invalid weighting floor")
    scores = {}
    for entry in assessments:
        if entry.client_id in scores or entry.client_id not in counts:
            raise ValueError("Duplicate/unknown assessed client")
        n = counts[entry.client_id]
        factors = (entry.quality, entry.current_risk, entry.prior_reputation)
        if type(n) is not int or n <= 0 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in factors):
            raise ValueError("Invalid registered count or assessment factor")
        scores[entry.client_id] = n * max(entry.quality * (1 - entry.current_risk) * entry.prior_reputation, floor)
    if not scores:
        raise ValueError("No assessed updates")
    total = sum(scores.values())
    weights = {c: scores[c] / total for c in sorted(scores)}
    validate_weights(weights)
    return weights


class FixedWeightAggregator:
    def __init__(self, root, directory, method, counts, floor=1e-6, frozen=None):
        if method not in ("adaptive", "frozen"):
            raise ValueError("Unknown assessment-weighted method")
        self.root, self.directory, self.method = root, directory, method
        self.counts, self.floor = counts, floor
        self.frozen = dict(frozen) if frozen is not None else None
        if self.frozen is not None:
            validate_weights(self.frozen)
            if set(self.frozen) != set(counts):
                raise ValueError("Frozen weights omit registered clients")

    def aggregate(self, request):
        started = time.perf_counter()
        ids = {u.client_id for u in request.updates}
        if ids != {a.client_id for a in request.assessments}:
            raise ValueError("Every valid update requires an assessment")
        current = adaptive_weights(request.assessments, self.counts, self.floor)
        frozen = self.frozen
        if self.method == "frozen":
            if frozen is None:
                if request.round_id != 1 or ids != set(self.counts):
                    raise ValueError("Frozen weighting requires full valid first-round coverage")
                frozen = dict(current)
            total = sum(frozen[c] for c in sorted(ids))
            current = {c: frozen[c] / total for c in sorted(ids)}
        validate_weights(current)
        parent = checkpoints.load(workspace_path(self.root, request.parent_model.path), request.parent_model.sha256)["model"]
        accum = {k: v.double().new_zeros(v.shape) for k, v in parent.items()}
        for update in sorted(request.updates, key=lambda u: u.client_id):
            if update.registered_training_count != self.counts[update.client_id]:
                raise ValueError("Unregistered training count")
            delta = checkpoints.load(workspace_path(self.root, update.delta.path), update.delta.sha256)["delta"]
            checkpoints.validate_state(delta, parent)
            for key in accum:
                accum[key] += delta[key].double() * current[update.client_id]
        delta = {k: value.to(parent[k].dtype) for k, value in accum.items()}
        state = {k: parent[k] + delta[k] for k in parent}
        checkpoints.validate_state(state, parent)
        path = self.directory / f"aggregate-{request.round_id:04d}.pt"
        checkpoints.save(path, {"model": state})
        diagnostics = self.directory / f"aggregation-{request.round_id:04d}.json"
        write_json(diagnostics, {"schema_version": "phase-e.aggregation.v1", "method": self.method,
                   "weights": current, "normalization": {}, "weight_floor": self.floor,
                   "frozen_weights": frozen, "global_update": magnitude(delta, parent), "seconds": time.perf_counter() - started})
        self.frozen = frozen
        return RoundResult(request.round_id, checkpoints.artifact(self.root, path),
                           tuple(ClientWeight(c, w) for c, w in current.items()), checkpoints.artifact(self.root, diagnostics), ())

"""Deterministic Euclidean bounded-simplex projection and exact identities."""
import hashlib
import math
import numpy as np
from edgefl.config import canonical_json

def validate(weights, cap=.5):
    x = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Weights must be a nonempty finite vector")
    if not math.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("Invalid simplex cap")
    cap = 1. if len(x) == 1 else cap
    if len(x) * cap < 1:
        raise ValueError("Infeasible simplex cap")
    if np.any(x < 0) or np.any(x > cap) or not math.isclose(math.fsum(x), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("Weights violate the bounded simplex")
    return x

def project(weights, cap=.5):
    x = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Cannot repair empty/nonfinite chromosome")
    if not math.isfinite(cap) or not 0 < cap <= 1 or (len(x) > 1 and len(x) * cap < 1):
        raise ValueError("Invalid/infeasible simplex cap")
    if len(x) == 1 or len(x) * cap == 1:
        return np.full(len(x), 1. / len(x))
    try:
        result = validate(x, cap).copy()
    except ValueError:
        # Translation leaves the projection invariant and prevents huge offsets.
        x = x - np.max(x)
        if not np.isfinite(x).all():
            raise ValueError("Chromosome range exceeds floating point precision")
        lo, hi = float(np.min(x) - cap), float(np.max(x))
        for _ in range(100):
            mid = (lo + hi) / 2
            if float(np.clip(x - mid, 0, cap).sum()) > 1:
                lo = mid
            else:
                hi = mid
        result = np.clip(x - (lo + hi) / 2, 0, cap)
        residual = 1. - math.fsum(result)
        for i in range(len(result)):
            adjustment = min(cap - result[i], residual) if residual > 0 else max(-result[i], residual)
            result[i] += adjustment
            residual -= adjustment
            if residual == 0:
                break
    result[result == 0] = 0.
    return validate(result, cap)

def identity(clients, weights, cap=.5):
    x = validate(weights, cap)
    if len(clients) != len(x) or tuple(clients) != tuple(sorted(set(clients))):
        raise ValueError("Stable unique client mapping required")
    encoded = [list(clients), [float(v if v != 0 else 0.).hex() for v in x]]
    return hashlib.sha256(canonical_json(encoded).encode()).hexdigest()

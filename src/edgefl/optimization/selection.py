"""Fixed-scale compromise over a feasible final Pareto front."""
import math
from dataclasses import asdict

def objectives(candidate, active=None):
    values = asdict(candidate.objectives)
    return tuple(values[n] for n in active) if active else tuple(values.values())

def select(front, active=None):
    from edgefl.optimization.policies import OBJECTIVES
    active = active or OBJECTIVES
    valid = [c for c in front if c.feasible]
    if not valid:
        return None
    def key(c):
        performance = tuple(getattr(c.objectives, n) for n in ("classification_error", "benign_fpr") if n in active)
        return (math.sqrt(sum(v*v for v in objectives(c, active))/len(active)), *performance, c.evaluation_id)
    return min(valid, key=key)

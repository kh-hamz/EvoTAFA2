"""Mixed research seeds and bounded deterministic population refill."""
import math
import numpy as np
from edgefl.optimization.chromosomes import project, identity
from edgefl.trust.aggregation import adaptive_weights
from edgefl.contracts.phase_f import SearchFailure

def expertise_weights(clients, assessments):
    entries = {a.client_id: {e.label: e for e in a.expertise} for a in assessments}
    labels = sorted(set().union(*(set(v) for v in entries.values())))
    contributions, unsupported = [], []
    for label in labels:
        column = [entries[c].get(label) for c in clients]
        if any(e is None for e in column) or len({e.support for e in column}) != 1:
            raise ValueError("Inconsistent trusted expertise support")
        if column[0].support == 0:
            if any(e.score is not None for e in column):
                raise ValueError("Unsupported expertise must be null")
            unsupported.append(label)
            continue
        if any(e.score is None or not math.isfinite(e.score) or not 0 <= e.score <= 1 for e in column):
            raise ValueError("Invalid supported expertise")
        scores = np.array([e.score for e in column])
        contributions.append(scores / scores.sum() if scores.sum() else np.full(len(clients), 1 / len(clients)))
    return (np.mean(contributions, axis=0) if contributions else np.full(len(clients), 1 / len(clients))), unsupported

def refill(clients, existing, size, rng, multiplier=100, excluded=(), cap=.5):
    result = [project(x, cap) for x in existing]
    seen = {identity(clients, x, cap) for x in (*excluded, *result)}
    attempts = 0
    while len(result) < size and attempts < multiplier * size:
        attempts += 1
        x = project(rng.dirichlet(np.ones(len(clients))), cap)
        key = identity(clients, x, cap)
        if key not in seen:
            result.append(x)
            seen.add(key)
    if len(result) != size:
        raise SearchFailure(f"Unsuccessful unique-population refill: {len(result)}/{size} after {attempts} attempts; not proof of infeasibility")
    return np.asarray(result), attempts

def initialize(clients, counts, assessments, previous, size, rng, floor=1e-6, multiplier=100, policy=None):
    from edgefl.optimization.policies import SearchPolicy
    policy = policy or SearchPolicy()
    counts_array = np.array([counts[c] for c in clients], dtype=float)
    if np.any(counts_array <= 0):
        raise ValueError("Positive registered counts required")
    sample = counts_array / counts_array.sum()
    adaptive = adaptive_weights(assessments, counts, floor, policy.direct_risk)
    expert, unsupported = expertise_weights(clients, assessments)
    seeds = [("fedavg", sample), ("uniform", np.full(len(clients), 1 / len(clients))),
             ("adaptive", [adaptive[c] for c in clients]), ("expertise", expert)]
    if policy.initialization == "no_expertise":
        seeds = [(n, v) for n, v in seeds if n != "expertise"]
    if policy.initialization == "uniform_random":
        seeds = [(n, v) for n, v in seeds if n == "uniform"]
    if previous is not None and policy.initialization != "uniform_random":
        if any(not math.isfinite(v) or v < 0 for v in previous.values()):
            raise ValueError("Invalid previous weights")
        warm = np.array([previous.get(c, sample[i]) for i, c in enumerate(clients)])
        seeds.append(("previous", warm / warm.sum() if warm.sum() else sample))
    unique, origins = {}, {}
    for name, seed in seeds:
        x = project(seed, policy.cap)
        key = identity(clients, x, policy.cap)
        unique.setdefault(key, x)
        origins.setdefault(key, []).append(name)
    population, attempts = refill(clients, list(unique.values()), size, rng, multiplier, cap=policy.cap)
    for x in population:
        origins.setdefault(identity(clients, x, policy.cap), ["random"])
    return population, {"origins": origins, "unsupported_classes": unsupported, "refill_attempts": attempts,
                        "duplicate_seed_count": len(seeds) - len(unique), "previous_available": previous is not None}

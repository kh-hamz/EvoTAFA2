"""Budget-matched actual-model random search using pymoo front extraction."""
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from edgefl.contracts.phase_f import SearchResult
from edgefl.optimization.initialization import initialize, refill
from edgefl.optimization.selection import objectives, select


def search(evaluator, settings, counts, previous=None, floor=1e-6):
    policy = evaluator.policy
    size, generations = settings["population_size"], settings["offspring_generations"]
    rng = np.random.default_rng(evaluator.request.evolution_seed)
    population, origins = initialize(evaluator.clients, counts, evaluator.request.assessments,
        previous, size, rng, floor, settings["refill_multiplier"], policy)
    trace = []
    for generation in range(generations + 1):
        if generation:
            population, attempts = refill(evaluator.clients, [], size, rng,
                                          settings["refill_multiplier"], cap=policy.cap)
            origins["refill_attempts"] += attempts
        batch = [evaluator.evaluate(x) for x in population]
        trace.append({"generation": generation, "population": [c.evaluation_id for c in batch]})
    if evaluator.counters["candidate_requests"] != size * (generations + 1):
        raise ValueError("Random-search request budget mismatch")
    valid = [c for c in evaluator.cache.values() if c.feasible]
    front = []
    if valid:
        indices = NonDominatedSorting().do(np.asarray([objectives(c, policy.active_objectives) for c in valid]),
                                           only_non_dominated_front=True)
        front = [valid[int(i)] for i in indices]
    chosen = select(front, policy.active_objectives)
    return SearchResult(chosen, tuple(evaluator.cache.values()), tuple(c.evaluation_id for c in front),
                        tuple(trace), origins, None if chosen else "No finite feasible candidate")

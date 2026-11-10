"""Thin adapters around pinned pymoo NSGA-II; no parallel algorithm implementation."""
import math
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from edgefl.contracts.phase_f import SearchResult, SearchFailure
from edgefl.optimization.chromosomes import project, identity
from edgefl.optimization.initialization import initialize, refill
from edgefl.optimization.selection import objectives, select
from edgefl.optimization.policies import SearchPolicy

class BoundedSimplexRepair(Repair):
    def __init__(self, cap=.5):
        super().__init__()
        self.cap = cap

    def _do(self, problem, X, **kwargs):
        return np.array([project(x, self.cap) for x in X])

class ExactDuplicates(DuplicateElimination):
    def __init__(self, clients, cap=.5):
        super().__init__()
        self.clients, self.rejections = clients, 0
        self.cap = cap

    def _do(self, pop, other, is_duplicate):
        seen = set() if other is None else {identity(self.clients, p.X, self.cap) for p in other}
        for i, p in enumerate(pop):
            key = identity(self.clients, p.X, self.cap)
            is_duplicate[i] = key in seen
            if other is None:
                seen.add(key)
        self.rejections += int(is_duplicate.sum())
        return is_duplicate

class RefillingMating(Mating):
    def __init__(self, clients, multiplier, *args, cap=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.clients, self.multiplier, self.refill_attempts = clients, multiplier, 0
        self.cap = cap

    def do(self, problem, pop, n_offsprings, random_state=None, **kwargs):
        off = super().do(problem, pop, n_offsprings, random_state=random_state, **kwargs)
        if len(off) < n_offsprings:
            values, attempts = refill(self.clients, [p.X for p in off], n_offsprings, random_state,
                                      self.multiplier, excluded=[p.X for p in pop], cap=self.cap)
            self.refill_attempts += attempts
            off = Population.new("X", values)
        return off

class FitnessProblem(ElementwiseProblem):
    def __init__(self, evaluator):
        self.policy = getattr(evaluator, "policy", SearchPolicy())
        super().__init__(n_var=len(evaluator.clients), n_obj=len(self.policy.active_objectives), n_ieq_constr=1, xl=0., xu=self.policy.cap)
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        result = self.evaluator.evaluate(x)
        out["F"] = objectives(result, self.policy.active_objectives) if result.feasible else (1.,) * self.n_obj
        out["G"] = (-1.,) if result.feasible else (1.,)

def search(evaluator, settings, counts, previous=None, floor=1e-6):
    policy = getattr(evaluator, "policy", SearchPolicy())
    size, generations = settings["population_size"], settings["offspring_generations"]
    seed = evaluator.request.evolution_seed
    initial, origins = initialize(evaluator.clients, counts, evaluator.request.assessments, previous,
                                  size, np.random.default_rng(seed), floor, settings["refill_multiplier"], policy)
    repair, duplicates = BoundedSimplexRepair(policy.cap), ExactDuplicates(evaluator.clients, policy.cap)
    selection = TournamentSelection(func_comp=binary_tournament)
    crossover = SBX(prob=.9, prob_var=.5, eta=20, prob_exch=1., prob_bin=.5)
    mutation = PM(prob=1., prob_var=1 / len(evaluator.clients), eta=20, at_least_once=False)
    mating = RefillingMating(evaluator.clients, settings["refill_multiplier"], selection, crossover, mutation,
                            repair=repair, eliminate_duplicates=duplicates, n_max_iterations=100, cap=policy.cap)
    algorithm = NSGA2(pop_size=size, n_offsprings=size, sampling=initial, selection=selection,
                      crossover=crossover, mutation=mutation, survival=RankAndCrowding(crowding_func="cd"),
                      repair=repair, eliminate_duplicates=duplicates, mating=mating)
    algorithm.tournament_type = "comp_by_rank_and_crowding"
    problem = FitnessProblem(evaluator)
    algorithm.setup(problem, seed=seed, termination=("n_gen", generations + 1), verbose=False)
    trace = []
    for generation in range(generations + 1):
        if not algorithm.has_next():
            raise SearchFailure("Premature pymoo termination")
        infills = algorithm.ask()
        if infills is None or len(infills) != size:
            raise SearchFailure("Unsuccessful population cardinality")
        algorithm.evaluator.eval(problem, infills, algorithm=algorithm)
        algorithm.tell(infills=infills)
        if len(algorithm.pop) != size:
            raise ValueError("Survival changed population size")
        members = []
        for p in algorithm.pop:
            crowding = p.get("crowding")
            members.append({"candidate": identity(evaluator.clients, p.X, policy.cap),
                            "rank": int(p.get("rank")) if p.get("rank") is not None else None,
                            "crowding": float(crowding) if crowding is not None and math.isfinite(crowding) else None,
                            "boundary": bool(crowding is not None and math.isinf(crowding)),
                            "feasible": bool(p.FEAS[0])})
        trace.append({"generation": generation, "population": members,
                      "non_dominated_fraction": sum(p["feasible"] and p["rank"] == 0 for p in members) / size})
    if algorithm.has_next() or evaluator.counters["candidate_requests"] != size * (generations + 1):
        raise ValueError("Search budget accounting mismatch")
    front = [evaluator.cache[p["candidate"]] for p in trace[-1]["population"] if p["feasible"] and p["rank"] == 0]
    chosen = select(front, policy.active_objectives)
    origins.update(duplicate_rejections=duplicates.rejections, offspring_refill_attempts=mating.refill_attempts)
    return SearchResult(chosen, tuple(evaluator.cache.values()), tuple(c.evaluation_id for c in front),
                        tuple(trace), origins, None if chosen else "No finite feasible candidate")

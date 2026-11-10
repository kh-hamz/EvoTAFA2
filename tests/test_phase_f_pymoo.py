import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from edgefl.optimization.pymoo_adapter import search
from edgefl.optimization.fitness import NumericalCandidateError
from edgefl.optimization.diagnostics import correlations
from phase_f_fixtures import numerical

class PymooTests(unittest.TestCase):
    def test_known_fronts_constants_duplicates_and_boundaries(self):
        F = np.array([[0,1,0,0], [1,0,0,0], [1,1,0,0], [.5,.5,0,0]])
        fronts = NonDominatedSorting().do(F)
        self.assertEqual(set(fronts[0]), {0,1,3})
        self.assertEqual(set(fronts[1]), {2})
        cd = get_crowding_function("cd")
        np.testing.assert_array_equal(cd.do(np.ones((4,4))), np.zeros(4))
        values = cd.do(np.array([[0,1,0,0], [.5,.5,0,0], [1,0,0,0]]))
        self.assertTrue(np.isinf(values[[0,2]]).all())
        self.assertAlmostEqual(values[1], .5)

    def test_rank_precedes_crowding_in_tournament(self):
        pop = Population.new("F", np.array([[.1]*4, [.2]*4]), "G", np.array([[-1.],[-1.]]))
        pop.set("rank", [0, 1])
        pop.set("crowding", [0., float("inf")])
        algorithm = SimpleNamespace(tournament_type="comp_by_rank_and_crowding", random_state=np.random.default_rng(5))
        self.assertEqual(int(binary_tournament(pop, np.array([[0,1]]), algorithm)[0,0]), 0)

    def test_default_budget_and_reproducible_search(self):
        f = numerical(self.addCleanup)
        f.settings.update(population_size=24, offspring_generations=8)
        first = search(f.evaluator, f.settings, f.counts)
        self.assertIsNotNone(first.selected)
        self.assertEqual(f.evaluator.counters["candidate_requests"], 216)
        self.assertEqual(len(first.generations), 9)
        self.assertTrue(all(len(g["population"]) == 24 for g in first.generations))
        f.evaluator.reset()
        second = search(f.evaluator, f.settings, f.counts)
        self.assertEqual(first.selected, second.selected)
        self.assertEqual(first.generations, second.generations)

    def test_all_invalid_and_constant_correlations(self):
        f = numerical(self.addCleanup)
        with patch.object(f.evaluator, "predict", side_effect=NumericalCandidateError("nonfinite")):
            result = search(f.evaluator, f.settings, f.counts)
        self.assertIsNone(result.selected)
        self.assertEqual(result.failure_reason, "No finite feasible candidate")
        self.assertIsNone(correlations(result.candidates)[0][0]["value"])

    def test_elitist_survival_and_private_rng(self):
        F = np.array([[0.,0.,0.,0.], [.8,.8,.8,.8], [.2,.2,.2,.2], [.4,.4,.4,.4]])
        pop = Population.new("F", F)
        kept = RankAndCrowding(crowding_func="cd").do(
            Problem(n_var=3, n_obj=4), pop, n_survive=2, random_state=np.random.default_rng(5))
        np.testing.assert_array_equal(kept.get("F"), F[[0,2]])
        f = numerical(self.addCleanup)
        state = np.random.get_state()
        search(f.evaluator, f.settings, f.counts)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

import unittest
from dataclasses import replace
import numpy as np
import torch
from edgefl.contracts.phase_g import ExperimentCase
from edgefl.contracts.records import CandidateEvaluation, ClientWeight, FitnessVector
from edgefl.experiments.policies import resolve
from edgefl.optimization.policies import SearchPolicy
from edgefl.optimization.chromosomes import project, validate
from edgefl.optimization.fitness import CandidateEvaluator
from edgefl.optimization.selection import select
from edgefl.optimization.random_search import search
from edgefl.trust.risk import calibrated_risk
from phase_f_fixtures import numerical


class PolicyTests(unittest.TestCase):
    def test_uncapped_projection_and_objective_ties(self):
        np.testing.assert_allclose(project([.9, .1], cap=1), [.9, .1])
        np.testing.assert_allclose(project([2, 0, 0], cap=.5), [.5, .25, .25])
        with self.assertRaises(ValueError): validate([.9, .1])
        weights = (ClientWeight("client", 1.),)
        a = CandidateEvaluation("a", weights, FitnessVector(1, 1, .2, .5), True)
        z = CandidateEvaluation("z", weights, FitnessVector(0, 0, .2, .5), True)
        self.assertEqual(select([a, z], ("current_risk", "historical_unreliability")).evaluation_id, "a")
        self.assertEqual(select([a, z]).evaluation_id, "z")

    def test_direct_risk_and_objective_only_ablation_are_distinct(self):
        base = ExperimentCase("binary", 11, "nsga2")
        direct, trust = resolve(replace(base, intervention="no_direct_risk"))
        objective, other = resolve(replace(base, intervention="without_current_risk"))
        self.assertEqual(direct.active_objectives, objective.active_objectives)
        self.assertFalse(direct.direct_risk)
        self.assertTrue(objective.direct_risk)
        self.assertTrue(trust.reputation)
        neutral, trust = resolve(replace(base, intervention="no_reputation"))
        self.assertFalse(trust.reputation)
        self.assertNotIn("historical_unreliability", neutral.active_objectives)
        with self.assertRaises(ValueError): resolve(replace(base, intervention="clipping"))

    def test_component_mask_and_random_actual_model_budget(self):
        calibration = {"schema_version": "phase-e.calibration.v1", "anchors": {n: {"lower": 0., "upper": 1.} for n in ("magnitude", "direction", "loss")}}
        raw = {"magnitude": 1., "direction": .2, "loss": .4}
        self.assertAlmostEqual(calibrated_risk(raw, calibration, ("direction", "loss"))[1], .3)
        f = numerical(self.addCleanup, classes=3)
        policy = SearchPolicy(backend="random", cap=1.)
        def evaluate():
            evaluator = CandidateEvaluator(f.root, f.request, f.data, f.trusted_ref, f.labels, f.values, f.counts, policy)
            result = search(evaluator, f.settings, f.counts)
            self.assertEqual(evaluator.counters["candidate_requests"], 12)
            self.assertGreater(evaluator.counters["actual_model_evaluations"], 0)
            self.assertIsNotNone(result.selected)
            return result
        a, b = evaluate(), evaluate()
        self.assertEqual(a.selected, b.selected)
        self.assertEqual(a.generations, b.generations)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_random_mask_replay(self):
        f = numerical(self.addCleanup, device="cuda")
        policy = SearchPolicy(backend="random", active_objectives=("classification_error", "benign_fpr", "current_risk"))
        results = []
        for _ in range(2):
            evaluator = CandidateEvaluator(f.root, f.request, f.data, f.trusted_ref, f.labels, f.values, f.counts, policy)
            results.append(search(evaluator, f.settings, f.counts))
        self.assertEqual(results[0].selected, results[1].selected)

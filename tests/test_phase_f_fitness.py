import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from edgefl.contracts.records import Partition, PartitionRef
from edgefl.optimization.fitness import CandidateEvaluator, NumericalCandidateError
from phase_f_fixtures import numerical

class FitnessTests(unittest.TestCase):
    def test_actual_model_metrics_differ_from_client_metric_average(self):
        f = numerical(self.addCleanup)
        result = f.evaluator.evaluate([.5, .25, .25])
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.objectives.classification_error, .8)
        self.assertEqual(result.objectives.benign_fpr, 1.)
        weighted_client_f1 = .5 * .2 + .5 * (3/7)
        self.assertNotAlmostEqual(1-result.objectives.classification_error, weighted_client_f1)
        self.assertAlmostEqual(result.objectives.current_risk, .25)

    def test_cache_failures_and_accounting(self):
        f = numerical(self.addCleanup)
        first = f.evaluator.evaluate([.5, .25, .25])
        self.assertIs(f.evaluator.evaluate([.5, .25, .25]), first)
        with patch.object(f.evaluator, "predict", side_effect=NumericalCandidateError("bad prediction")):
            bad = f.evaluator.evaluate([.25, .5, .25])
        self.assertFalse(bad.feasible)
        self.assertIsNone(bad.objectives)
        f.evaluator.evaluate([.25, .5, .25])
        counts = f.evaluator.counters
        self.assertEqual(counts["candidate_requests"], 4)
        self.assertEqual(counts["cache_hits"], 2)
        self.assertEqual(counts["actual_model_evaluations"], 2)
        self.assertEqual(counts["invalid_candidate_requests"], 2)

    def test_pre_inference_failure_and_infrastructure_exception(self):
        f = numerical(self.addCleanup)
        with patch.object(f.evaluator, "construct", side_effect=NumericalCandidateError("bad parameters")):
            self.assertFalse(f.evaluator.evaluate([.5, .25, .25]).feasible)
        self.assertEqual(f.evaluator.counters["pre_inference_failures"], 1)
        with patch.object(f.evaluator, "predict", side_effect=RuntimeError("device failed")):
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                f.evaluator.evaluate([.25, .5, .25])

    def test_roles_counts_and_missing_benign_rejected(self):
        f = numerical(self.addCleanup)
        with self.assertRaises(ValueError):
            replace(f.request, trusted=PartitionRef(f.trusted_ref.manifest, Partition.FINAL_TEST))
        no_benign = torch.utils.data.TensorDataset(torch.zeros(4, 2), torch.ones(4, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "benign"):
            CandidateEvaluator(f.root, f.request, no_benign, f.trusted_ref, f.labels, f.values, f.counts)
        with self.assertRaisesRegex(ValueError, "count"):
            CandidateEvaluator(f.root, f.request, f.data, f.trusted_ref, f.labels, f.values, {"a": 999, "b": 5, "c": 6})

    def test_changed_assessment_gets_fresh_cache_and_unit_bounds(self):
        f = numerical(self.addCleanup)
        before = f.evaluator.evaluate([.5, .25, .25])
        request = replace(f.request, assessments=tuple(replace(a, current_risk=1., prior_reputation=0.)
                                                      for a in f.request.assessments))
        evaluator = CandidateEvaluator(f.root, request, f.data, f.trusted_ref, f.labels, f.values, f.counts)
        after = evaluator.evaluate([.5, .25, .25])
        self.assertEqual(after.evaluation_id, before.evaluation_id)
        self.assertEqual(after.objectives.current_risk, 1.)
        self.assertEqual(after.objectives.historical_unreliability, 1.)
        self.assertEqual(evaluator.counters["cache_hits"], 0)

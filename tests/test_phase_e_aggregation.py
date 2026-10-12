"""Independent weighting expectations and defense boundary validation."""

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
import torch
from edgefl.contracts.records import ClientAssessment, ClientUpdate, Compatibility
from edgefl.contracts.interfaces import AggregationRequest
from edgefl.learning import checkpoints
from edgefl.trust.aggregation import adaptive_weights, validate_weights, FixedWeightAggregator


def assessment(client, q, risk, prior):
    return ClientAssessment(client, 1, q, (), (), risk, prior, ())


class WeightTests(unittest.TestCase):
    def test_product_uses_registered_counts_and_prior(self):
        entries = [assessment("a", .8, .25, .5), assessment("b", .5, .2, .9)]
        weights = adaptive_weights(entries, {"a": 10, "b": 20})
        self.assertAlmostEqual(weights["a"], 3 / 10.2)
        self.assertAlmostEqual(weights["b"], 7.2 / 10.2)

    def test_floor_collapse_is_sample_proportional_and_no_phase_f_cap(self):
        weights = adaptive_weights([assessment("a", 0, 1, 0), assessment("b", 0, 1, 0)], {"a": 1, "b": 9})
        self.assertAlmostEqual(weights["a"], .1)
        self.assertAlmostEqual(weights["b"], .9)

    def test_invalid_scores_duplicates_and_counts_fail(self):
        for entries, counts in (([assessment("a", float("nan"), 0, .5)], {"a": 1}),
                                ([assessment("a", .5, 0, .5)] * 2, {"a": 1}),
                                ([assessment("a", .5, 0, .5)], {"a": 0})):
            with self.assertRaises(ValueError):
                adaptive_weights(entries, counts)
        with self.assertRaises(ValueError):
            validate_weights({"a": .2, "b": .2})

    def test_frozen_partial_participation_and_actual_model_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "parent.pt"
            checkpoints.save(path, {"model": {"w": torch.zeros(1)}})
            parent = checkpoints.artifact(root, path)
            comp = Compatibility("a" * 64, "b" * 64, "c" * 64, "fixture", 1, 2)
            updates, assessments = [], []
            for i in range(1, 5):
                client = str(i)
                path = root / (client + ".pt")
                checkpoints.save(path, {"delta": {"w": torch.tensor([float(i)])}})
                updates.append(ClientUpdate(client, 1, parent, checkpoints.artifact(root, path), comp, 1, 0., 4))
                assessments.append(assessment(client, i / 4, 0., .5))
            counts = dict.fromkeys(("1", "2", "3", "4"), 1)
            rejected = FixedWeightAggregator(root, root / "rejected", "frozen", counts)
            corrupt = (replace(updates[0], registered_training_count=999), *updates[1:])
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                rejected.aggregate(AggregationRequest(1, parent, comp, corrupt, tuple(assessments), None, 1))
            self.assertIsNone(rejected.frozen)
            aggregator = FixedWeightAggregator(root, root, "frozen", counts)
            first = aggregator.aggregate(AggregationRequest(1, parent, comp, tuple(updates), tuple(assessments), None, 1))
            self.assertAlmostEqual(float(checkpoints.load(root / first.checkpoint.path)["model"]["w"][0]), 3.)
            later = [replace(u, round_id=2, parent_model=first.checkpoint) for u in updates[:3]]
            changed = [replace(a, round_id=2, quality=0.) for a in assessments[:3]]
            second = aggregator.aggregate(AggregationRequest(2, first.checkpoint, comp, tuple(later), tuple(changed), None, 1))
            self.assertAlmostEqual(float(checkpoints.load(root / second.checkpoint.path)["model"]["w"][0]), 3 + 14 / 6, places=6)
            self.assertAlmostEqual(second.weights[0].weight, 1 / 6)
            incomplete = FixedWeightAggregator(root, root / "incomplete", "frozen", counts)
            with self.assertRaisesRegex(ValueError, "full valid"):
                incomplete.aggregate(AggregationRequest(1, parent, comp, tuple(updates[:3]), tuple(assessments[:3]), None, 1))

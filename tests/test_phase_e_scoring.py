"""Hand-calculated trusted metrics, support and expertise expectations."""

import math
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from edgefl.trust.scoring import evaluate_trusted, expertise, InvalidPrediction


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.values = {"device": "cpu", "batch_size": 3}

    def test_binary_metrics_and_original_attack_expertise(self):
        probabilities = torch.tensor([[.8, .2], [.4, .6], [.3, .7], [.9, .1]])
        data = TensorDataset(probabilities.log(), torch.tensor([0, 0, 1, 1]))
        report = evaluate_trusted(torch.nn.Identity(), data, ["Normal", "Normal", "A", "B"], {"Normal": 0, "Attack": 1}, "binary", self.values)
        self.assertEqual(report["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(report["macro_f1"], .5)
        self.assertEqual(report["balanced_accuracy"], .5)
        self.assertEqual(report["benign_fpr"], .5)
        self.assertEqual(report["raw_expertise"], {"A": 1., "B": 0.})
        expected = -sum(math.log(v) for v in (.8, .4, .7, .1)) / 4
        self.assertAlmostEqual(report["loss"], expected, places=6)
        self.assertAlmostEqual(report["class_balanced_loss"], expected, places=6)

    def test_multiclass_unsupported_and_shrinkage(self):
        data = TensorDataset(torch.tensor([[3., 0., 0.], [0., 3., 0.]]), torch.tensor([0, 1]))
        report = evaluate_trusted(torch.nn.Identity(), data, ["Normal", "A"], {"Normal": 0, "A": 1, "B": 2}, "multiclass", self.values)
        self.assertAlmostEqual(report["macro_f1"], 2 / 3)
        self.assertEqual(report["balanced_accuracy"], 1.)
        self.assertIsNone(report["raw_expertise"]["B"])
        parent = {**report, "raw_expertise": {"Normal": 0., "A": .5, "B": None}}
        scores = {e.label: e for e in expertise(report, parent, ("Normal", "A", "B"))}
        self.assertAlmostEqual(scores["A"].score, 26 / 51)
        self.assertIsNone(scores["B"].score)
        self.assertEqual(scores["B"].support, 0)

    def test_no_benign_and_nonfinite_predictions(self):
        data = TensorDataset(torch.tensor([[0., 1.]]), torch.tensor([1]))
        report = evaluate_trusted(torch.nn.Identity(), data, ["A"], {"Normal": 0, "Attack": 1}, "binary", self.values)
        self.assertIsNone(report["benign_fpr"])
        broken = TensorDataset(torch.tensor([[float("nan"), 1.]]), torch.tensor([1]))
        with self.assertRaises(InvalidPrediction):
            evaluate_trusted(torch.nn.Identity(), broken, ["A"], {"Normal": 0, "Attack": 1}, "binary", self.values)

    def test_scorer_is_independent_of_truth_and_rejects_wrong_roles_and_rounds(self):
        from dataclasses import replace
        from edgefl.learning import checkpoints
        from edgefl.learning.models import initialized
        from edgefl.contracts.records import Compatibility, ClientUpdate, PartitionRef, Partition
        from edgefl.contracts.interfaces import ScoringRequest
        from edgefl.trust.scoring import TrustedClientScorer
        from edgefl.attacks.diagnostics import summarize as diagnostic_join
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = initialized(1, 2, 11)
            checkpoints.save(root / "parent.pt", {"model": model.state_dict()})
            parent = checkpoints.artifact(root, root / "parent.pt")
            comp = Compatibility("a" * 64, "b" * 64, "c" * 64, "mlp-float32-v1", 1, 2)
            zero = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            checkpoints.save(root / "delta.pt", {"delta": zero})
            delta = checkpoints.artifact(root, root / "delta.pt")
            updates = tuple(ClientUpdate(c, 1, parent, delta, comp, 4, 0., 0) for c in ("a", "b", "c"))
            data = TensorDataset(torch.tensor([[-1.], [1.], [-1.], [1.]]), torch.tensor([0, 1, 0, 1]))
            trusted = PartitionRef(parent, Partition.TRUSTED)
            scorer = TrustedClientScorer(root, data, ["Normal", "A", "Normal", "A"], trusted, comp,
                                        {"Normal": 0, "Attack": 1}, "binary", self.values, ["Normal", "A", "B"])
            request = ScoringRequest(trusted, parent, updates, comp)
            before = scorer.observe(request)
            profiles = {c: {"concentrated": True} for c in ("a", "b", "c")}
            honest = {c: {"compromised": False, "scheduled_active": False} for c in profiles}
            compromised = {c: {"compromised": True, "scheduled_active": True} for c in profiles}
            self.assertNotEqual(diagnostic_join(honest, dict.fromkeys(profiles, .8), .5, profiles),
                                diagnostic_join(compromised, dict.fromkeys(profiles, .8), .5, profiles))
            self.assertEqual(before, scorer.observe(request))
            self.assertTrue(all(next(e for e in r["expertise"] if e["label"] == "B")["score"] is None for r in before["clients"].values()))
            with self.assertRaises(ValueError):
                ScoringRequest(PartitionRef(parent, Partition.FINAL_TEST), parent, updates, comp)
            with self.assertRaisesRegex(ValueError, "mixed-round"):
                scorer.observe(ScoringRequest(trusted, parent, (*updates[:2], replace(updates[2], round_id=2)), comp))

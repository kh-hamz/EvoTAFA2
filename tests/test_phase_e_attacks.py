"""Independent attack semantics, fixed sampling and immutable data tests."""

import unittest
import torch
from torch.utils.data import TensorDataset
from edgefl.attacks.plan import build_plan, active, attack_seed
from edgefl.attacks.strategies import PoisonedTrainingView, transform


class MetadataFixture:
    def __init__(self):
        self.spec = {"labels": {"Normal": 0, "Attack": 1}, "profiles": {
            c: {"count": 20, "support": {"Normal": 10, "A": 7, "B": 3}} for c in ("a", "b", "c", "d", "e")}}

    def rows(self, role, client):
        return [{"row": i, "observation_id": client + str(i), "original": "Normal" if i < 10 else ("A" if i < 17 else "B"),
                 "target": int(i >= 10)} for i in range(20)]


class AttackTests(unittest.TestCase):
    def setUp(self):
        self.values = {"compromised_fraction": .2, "label_fraction": .3, "activation_round": 11,
                       "intermittent_period": 5, "update_scale": 5.}
        self.meta = MetadataFixture()

    def test_fixed_selection_rounding_and_training_only_target(self):
        targeted = build_plan(self.meta, self.values, 11, "targeted_label")
        self.assertEqual(targeted["target"], "A")
        client = targeted["compromised"][0]
        self.assertEqual(len(targeted["changes"][client]), 2)
        self.assertTrue(all(e["replacement"] == 0 and 10 <= e["row"] < 17 for e in targeted["changes"][client]))
        untargeted = build_plan(self.meta, self.values, 11, "untargeted_label")
        self.assertEqual(untargeted["compromised"], targeted["compromised"])
        self.assertEqual(len(untargeted["changes"][client]), 6)
        self.assertTrue(all(e["replacement"] != e["original_target"] for e in untargeted["changes"][client]))
        self.assertEqual(untargeted, build_plan(self.meta, self.values, 11, "untargeted_label"))

    def test_intermittent_boundaries_and_compromised_not_active(self):
        plan = build_plan(self.meta, self.values, 11, "intermittent_sign")
        c = plan["compromised"][0]
        self.assertEqual([active(plan, c, r) for r in (10, 11, 15, 16, 20, 21)], [False, True, True, False, False, True])
        self.assertFalse(active(plan, next(c for c in plan["clients"] if c not in plan["compromised"]), 11))

    def test_pure_delta_operations_noise_norm_and_seed_separation(self):
        original = {"w": torch.tensor([3., 4.])}
        torch.testing.assert_close(transform(original, "sign_flip", 1)["w"], torch.tensor([-3., -4.]))
        torch.testing.assert_close(transform(original, "update_scaling", 1)["w"], torch.tensor([15., 20.]))
        noisy = transform(original, "additive_noise", 1)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(noisy["w"] - original["w"])), 5., places=5)
        torch.testing.assert_close(original["w"], torch.tensor([3., 4.]))
        self.assertEqual(attack_seed(11, "x", "a", 1), attack_seed(11, "x", "a", 1))
        self.assertNotEqual(attack_seed(11, "x", "a", 1), attack_seed(11, "y", "a", 1))
        zero = {"w": torch.zeros(2)}
        torch.testing.assert_close(transform(zero, "additive_noise", 1)["w"], zero["w"])

    def test_overlay_never_mutates_original(self):
        original = TensorDataset(torch.zeros((4, 1)), torch.tensor([0, 1, 1, 0]))
        view = PoisonedTrainingView(original, [{"row": 1, "original_target": 1, "replacement": 0}])
        self.assertEqual(view[1][1], 0)
        self.assertEqual(int(original[1][1]), 1)
        with self.assertRaises(ValueError):
            PoisonedTrainingView(original, [{"row": 1, "original_target": 0, "replacement": 1}])

    def test_zero_compromise_and_missing_target(self):
        with self.assertRaisesRegex(ValueError, "no compromised"):
            build_plan(self.meta, {**self.values, "compromised_fraction": .01}, 11, "sign_flip")
        for p in self.meta.spec["profiles"].values():
            p["support"] = {"Normal": 20}
        with self.assertRaisesRegex(ValueError, "eligible target"):
            build_plan(self.meta, self.values, 11, "targeted_label")
        self.assertEqual(build_plan(self.meta, self.values, 11, "clean")["compromised"], [])

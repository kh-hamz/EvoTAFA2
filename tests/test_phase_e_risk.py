"""Independent vector and quantile checks for clean-excess risk."""

import copy
import math
import unittest
import torch
from edgefl.trust.risk import raw_signals, normalize, calibrated_risk
from edgefl.trust.calibration import fit, audit


class RiskTests(unittest.TestCase):
    def test_direction_loss_and_magnitude_have_distinct_meanings(self):
        updates = {"a": {"w": torch.tensor([1., 0.])}, "b": {"w": torch.tensor([1., 0.])}, "c": {"w": torch.tensor([-1., 0.])}}
        signals = raw_signals(updates, {"a": .5, "b": .4, "c": 1.5}, .5)
        self.assertEqual(signals["c"]["direction"], 1.)
        self.assertEqual(signals["a"]["direction"], 0.)
        self.assertEqual(signals["c"]["magnitude"], 0.)
        self.assertEqual(signals["c"]["loss"], 1.)
        self.assertEqual(signals["b"]["loss"], 0.)

    def test_even_medians_zero_vectors_and_insufficient_peers(self):
        updates = {str(i): {"w": torch.tensor([float(v)])} for i, v in enumerate((0, 2, 4, 6))}
        values = raw_signals(updates, dict.fromkeys(updates, 1.), 1.)
        self.assertEqual(values["0"]["coordinate_median_norm"], 3.)
        self.assertEqual(values["0"]["direction"], 1.)
        self.assertAlmostEqual(values["3"]["magnitude"], math.log(2), places=10)
        zero = {c: {"w": torch.zeros(1)} for c in updates}
        self.assertFalse(raw_signals(zero, dict.fromkeys(zero, 1.), 1.)["0"]["direction_reference_available"])
        with self.assertRaises(ValueError):
            raw_signals({"x": zero["0"]}, {"x": 1.}, 1.)

    def test_degenerate_normalization_clips_and_nonfinite_rejection(self):
        anchors = {"lower": 2., "upper": 4.}
        self.assertEqual([normalize(x, anchors) for x in (0, 2, 3, 4, 8)], [0, 0, .5, 1, 1])
        self.assertEqual(normalize(0, {"lower": 0., "upper": 0.}), 0.)
        self.assertEqual(normalize(.1, {"lower": 0., "upper": 0.}), 1.)
        with self.assertRaises(ValueError):
            normalize(float("nan"), anchors)

    def test_fit_and_independent_audit_do_not_refit(self):
        def row(x):
            return {"client": "a", "round": 1, "raw": dict.fromkeys(("magnitude", "direction", "loss"), x)}
        fitted = fit({101: [row(0.), row(.1)], 102: [row(.2), row(.3)]}, {"fixture": True})
        self.assertAlmostEqual(fitted["anchors"]["loss"]["lower"], .15)
        self.assertAlmostEqual(fitted["anchors"]["loss"]["upper"], .297)
        original = copy.deepcopy(fitted)
        report = audit(fitted, [row(1.)], {"a": {"concentrated": True}})
        self.assertEqual(fitted, original)
        self.assertEqual(report["cohorts"]["concentrated"]["client_rounds"], 1)
        self.assertIsNone(report["cohorts"]["declared_specialist_scenario"]["flag_rate"])
        self.assertEqual(calibrated_risk(row(1.)["raw"], fitted)[1], 1.)

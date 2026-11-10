import unittest
from edgefl.experiments.reporting import paired, statistics_summary, assert_matched


class ReportingTests(unittest.TestCase):
    def test_hand_calculated_paired_interval_and_missing_seeds(self):
        a = dict(zip((11, 22, 33, 44, 55), (1., 2., 3., 4., 5.)))
        b = dict.fromkeys(a, 0.)
        result = paired(a, b)
        self.assertEqual(result["mean"], 3.)
        self.assertAlmostEqual(result["sample_std"], 1.5811388300841898)
        self.assertAlmostEqual(result["ci95"][0], 1.0367568385, places=7)
        self.assertTrue(result["complete_five_seeds"])
        small = paired({11: .2}, {11: .1, 22: .3})
        self.assertIsNone(small["ci95"])
        self.assertEqual(small["missing_seeds"], [22, 33, 44, 55])
        with self.assertRaises(ValueError): statistics_summary([float("nan")])

    def test_matching_exceptions_are_explicit(self):
        a = {"matching": dict.fromkeys(("data", "membership", "initialization", "panel", "schedule", "attack"), "a")}
        b = {"matching": {**a["matching"], "data": "different", "initialization": "different"}}
        with self.assertRaises(ValueError): assert_matched(a, b)
        assert_matched(a, b, feature_study=True)
        b["matching"]["membership"] = "other"
        with self.assertRaises(ValueError): assert_matched(a, b, feature_study=True)

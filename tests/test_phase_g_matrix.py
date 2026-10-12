import unittest
from dataclasses import replace
from edgefl.contracts.phase_g import ExperimentCase
from edgefl.experiments.matrix import compile_matrix, normalized
from edgefl.experiments.dependencies import inventory


class MatrixTests(unittest.TestCase):
    def test_primary_count_and_matched_families(self):
        settings = {"suites": ["main"], "capture_folds": []}
        result = compile_matrix(settings)
        self.assertEqual(result["unique_cases"], 800)
        self.assertEqual(result["counts"], {"main": 800})
        cases = list(result["cases"].values())
        self.assertEqual(sum(r["condition"] == "clean" for r in cases), 320)
        self.assertTrue(all(r["scenario"] == "dirichlet_moderate" for r in cases if r["condition"] != "clean"))
        self.assertEqual(result, compile_matrix(settings))
        requirements = inventory(result)
        self.assertTrue(all("phase_e_21_run_acceptance" in r["requirements"] for r in requirements["scopes"].values()))
        self.assertGreater(requirements["nominal_upstream_training_runs"], 800)

    def test_focused_studies_and_reference_deduplication(self):
        result = compile_matrix({"suites": ["ablations", "sensitivity", "capture"], "capture_folds": ["unseen_capture"]})
        self.assertTrue(all(row["case_id"] in result["cases"] and (row["reference_id"] is None or row["reference_id"] in result["cases"])
                            for row in result["memberships"]))
        self.assertEqual(sum(r["suite"] == "capture" for r in result["memberships"]), 40)
        self.assertTrue(all(result["cases"][r["case_id"]]["task"] == "binary" for r in result["memberships"] if r["suite"] == "capture"))
        base = ExperimentCase("binary", 11, "fedavg")
        self.assertEqual(normalized(base), normalized(replace(base, condition="update_scaling", compromised_fraction=0, update_scale=10)))

    def test_missing_capture_registration_and_fixture_boundary(self):
        self.assertTrue(compile_matrix({"suites": ["capture"], "capture_folds": []})["capture_folds_unregistered"])
        with self.assertRaises(ValueError):
            compile_matrix({"fixture_cases": [{}], "evidence_kind": "research"})
        with self.assertRaises(ValueError):
            ExperimentCase("multiclass", 11, "fedavg", fold="unseen_x")

import tempfile
import unittest
from pathlib import Path
from edgefl.data.storage import read_json, write_json
from edgefl.experiments.config import load
from edgefl.experiments.dependencies import inventory, materialize_configurations
from edgefl.experiments.matrix import compile_matrix

ROOT = Path(__file__).resolve().parents[1]


class DependencyTests(unittest.TestCase):
    def test_declared_config_and_protected_output(self):
        config = load(ROOT / "configs/phase_g.json", ROOT)
        self.assertEqual(config.values["evidence_kind"], "research")
        self.assertEqual(config.values["workers"], 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            path = Path(tmp) / "bad.json"
            value = config.values
            value["output"] = "src"
            write_json(path, value)
            with self.assertRaises(ValueError): load(path, ROOT)
            value["output"] = "runs/phase_g"
            value["fixture_cases"] = [{}]
            write_json(path, value)
            with self.assertRaises(ValueError): load(path, ROOT)

    def test_scope_configs_keep_data_constraints_and_vary_partition_seed(self):
        config = load(ROOT / "configs/phase_g.json", ROOT)
        matrix = compile_matrix({"suites": ["main"], "capture_folds": []})
        key = next(k for k, v in matrix["cases"].items() if v["seed"] == 22 and v["method"] == "nsga2" and v["condition"] == "clean")
        subset = {"cases": {key: matrix["cases"][key]}}
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            result = materialize_configurations(config, subset, Path(tmp))
            c = read_json(ROOT / result[key]["configurations"]["c"])
            self.assertEqual(c["seed"], 22)
            self.assertEqual(c["minimum_client_observations"], 10000)
            self.assertEqual(c["near_iid_limits"], {"maximum_size_deviation": .1, "maximum_total_variation": .05})
            e = read_json(ROOT / result[key]["configurations"]["e"])
            self.assertEqual(e["compromised_fraction"], .2)
            self.assertEqual(inventory(subset)["nominal_upstream_training_runs"], 33)

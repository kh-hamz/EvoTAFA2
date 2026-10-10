"""Foundation tests use tiny temporary metadata only; never touch research datasets."""

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from edgefl.cli import main
from edgefl.config import ConfigurationError, load_config, workspace_path
from edgefl.contracts.interfaces import AggregationRequest, ScoringRequest
from edgefl.contracts.records import (
    ArtifactRef, CandidateEvaluation, ClassSupport, ClientAssessment, ClientManifest,
    ClientUpdate, ClientWeight, Compatibility, FitnessVector, Partition, PartitionRef,
    require_compatible,
)
from edgefl.pipelines.catalog import PIPELINES
from edgefl.reproducibility import STREAMS, derive_seed, seed_manifest
from edgefl.runs import initialize_foundation_run, source_snapshot

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "docs/research_protocol.md").write_text("Test protocol", encoding="utf-8")
        self.path = self.root / "configs/phase_a.json"
        self.values = json.loads((ROOT / "configs/phase_a.json").read_text(encoding="utf-8"))
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.values), encoding="utf-8")

    def load(self):
        return load_config(self.path, self.root)

    def test_default_configuration_and_schema(self):
        schema = json.loads((ROOT / "src/edgefl/schemas/config.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        config = self.load()
        self.assertEqual(config.values["scope"], "phase_a")
        self.assertEqual(config.values["research_defaults"]["clients"], 10)

    def test_snapshot_is_immutable_and_hash_ignores_json_formatting(self):
        before = self.load()
        copy = before.values
        copy["scope"] = "phase_b"
        self.path.write_text(json.dumps(self.values, sort_keys=True, indent=4), encoding="utf-8")
        self.assertEqual(before.sha256, self.load().sha256)
        self.assertEqual(before.values["scope"], "phase_a")

    def test_reject_unknown_keys_and_wrong_phase(self):
        for key, value in (("unexpected", 1), ("scope", "phase_b")):
            with self.subTest(key=key):
                original = dict(self.values)
                self.values[key] = value
                self.write()
                with self.assertRaises(ConfigurationError):
                    self.load()
                self.values = original

    def test_reject_invalid_numeric_settings(self):
        cases = (("clients", True), ("learning_rate", 0), ("malicious_fraction", 1),
                 ("nsga_population", 1), ("attack_start_round", 101))
        for key, value in cases:
            with self.subTest(key=key):
                original = self.values["research_defaults"][key]
                self.values["research_defaults"][key] = value
                self.write()
                with self.assertRaises(ConfigurationError):
                    self.load()
                self.values["research_defaults"][key] = original

    def test_reject_split_sum_and_duplicate_seeds(self):
        self.values["research_defaults"]["global_split"]["client_pool"] = 0.6
        self.write()
        with self.assertRaisesRegex(ConfigurationError, "sum to 1"):
            self.load()
        self.values["research_defaults"]["global_split"]["client_pool"] = 0.7
        self.values["randomness"]["matched_seeds"] = [11, 11]
        self.write()
        with self.assertRaises(ConfigurationError):
            self.load()

    def test_reject_duplicate_json_keys_and_nonfinite_values(self):
        for text in ('{"scope":"phase_a","scope":"phase_b"}', '{"x":NaN}'):
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                self.load()
        self.values["research_defaults"]["learning_rate"] = math.inf
        self.write()
        with self.assertRaises(ConfigurationError):
            self.load()

    def test_reject_unsafe_paths(self):
        for value in ("../archive", "/absolute", "C:/elsewhere", "C:relative", ".", "runs\\x"):
            with self.subTest(path=value), self.assertRaises(ConfigurationError):
                workspace_path(self.root, value)
        self.values["dataset"]["primary"] = "docs/research_protocol.md"
        self.write()
        with self.assertRaisesRegex(ConfigurationError, "immutable archive"):
            self.load()

    def test_reject_output_alias_into_archive(self):
        real = workspace_path
        def alias(root, value):
            return root / "archive/outputs" if value == "runs" else real(root, value)
        with patch("edgefl.config.workspace_path", side_effect=alias):
            with self.assertRaisesRegex(ConfigurationError, "overlap"):
                self.load()

    def test_validation_cli_does_not_write_artifacts(self):
        before = sorted(str(p) for p in self.root.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(["--workspace", str(self.root), "validate-config"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))

    def test_run_records_are_distinct_linked_and_honestly_pending(self):
        config = self.load()
        first = initialize_foundation_run(config, 11)
        second = initialize_foundation_run(config, 11)
        self.assertNotEqual(first, second)
        metadata = json.loads((first / "run.json").read_text())
        self.assertFalse(metadata["eligible_for_training"])
        self.assertEqual(metadata["dataset_registration"]["status"], "pending_phase_b")
        self.assertIsNone(metadata["dataset_registration"]["primary"]["sha256"])
        self.assertEqual(metadata["configuration_sha256"], config.sha256)
        for artifact in metadata["artifacts"].values():
            self.assertEqual(hashlib.sha256((first / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        self.assertEqual((first / "seeds.json").read_bytes(), (second / "seeds.json").read_bytes())
        self.assertEqual(json.loads((first / "events.jsonl").read_text())["pipeline"], "foundation")
        self.assertFalse((self.root / "archive").exists())
        with self.assertRaises(ValueError):
            initialize_foundation_run(config, 999)

    def test_source_fingerprint_excludes_data_and_changes_with_code(self):
        (self.root / "archive").mkdir()
        (self.root / "archive/data.csv").write_text("not research data")
        (self.root / "src").mkdir()
        code = self.root / "src/example.py"
        code.write_text("x = 1")
        first = source_snapshot(self.root)
        (self.root / "archive/data.csv").write_text("changed")
        self.assertEqual(first, source_snapshot(self.root))
        code.write_text("x = 2")
        self.assertNotEqual(first["sha256"], source_snapshot(self.root)["sha256"])


class SeedTests(unittest.TestCase):
    def test_named_streams_and_identity_coordinates_are_independent(self):
        values = {derive_seed(11, stream) for stream in STREAMS}
        self.assertEqual(len(values), len(STREAMS))
        self.assertEqual(derive_seed(11, "evolution"), derive_seed(11, "evolution"))
        self.assertNotEqual(derive_seed(11, "minibatch", client="a", round_id=1),
                            derive_seed(11, "minibatch", client="b", round_id=1))
        self.assertNotEqual(derive_seed(11, "minibatch", client="a", round_id=1),
                            derive_seed(11, "minibatch", client="a", round_id=2))

    def test_global_split_is_fixed_across_matched_seeds(self):
        a, b = seed_manifest(11, 42), seed_manifest(22, 42)
        self.assertEqual(a["global_split"], b["global_split"])
        self.assertNotEqual(a["streams"], b["streams"])

    def test_invalid_seed_namespace_or_coordinates(self):
        for master, stream in ((True, "partition"), (-1, "partition"), (11, "typo")):
            with self.assertRaises(ValueError):
                derive_seed(master, stream)
        with self.assertRaises(ValueError):
            derive_seed(11, "minibatch", round_id=-1)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ref = ArtifactRef("artifacts/model.bin", "a" * 64)
        self.compat = Compatibility("a" * 64, "b" * 64, "c" * 64, "mlp-v1", 40, 2)
        self.update = ClientUpdate("c1", 1, self.ref, self.ref, self.compat, 100, 1.0, 128)
        self.trusted = PartitionRef(self.ref, Partition.TRUSTED)

    def test_reject_test_and_selection_data_in_scoring_and_aggregation(self):
        for role in (Partition.FINAL_TEST, Partition.SELECTION_VALIDATION, Partition.LOCAL_TRAIN):
            forbidden = PartitionRef(self.ref, role)
            with self.subTest(role=role), self.assertRaises(ValueError):
                ScoringRequest(forbidden, self.ref, (self.update,), self.compat)
            with self.subTest(role=role), self.assertRaises(ValueError):
                AggregationRequest(1, self.ref, self.compat, (self.update,), (), forbidden, 11)
        ScoringRequest(self.trusted, self.ref, (self.update,), self.compat)

    def test_client_training_and_validation_roles(self):
        with self.assertRaises(ValueError):
            ClientManifest("c1", self.ref, self.trusted,
                           PartitionRef(self.ref, Partition.LOCAL_VALIDATION), (), 11)
        with self.assertRaises(ValueError):
            ClassSupport("Normal", -1)

    def test_compatibility_checks_feature_order_not_just_dimensions(self):
        require_compatible(self.compat, self.compat)
        with self.assertRaises(ValueError):
            require_compatible(self.compat, replace(self.compat, feature_order_sha256="d" * 64))

    def test_stale_updates_duplicate_clients_and_mixed_parents_rejected(self):
        bad_groups = (
            (replace(self.update, round_id=0),), (self.update, self.update),
            (replace(self.update, parent_model=ArtifactRef("other", "b" * 64)),),
        )
        for updates in bad_groups:
            with self.assertRaises(ValueError):
                AggregationRequest(1, self.ref, self.compat, updates, (), None, 11)

    def test_assessments_cannot_reference_unknown_or_stale_clients(self):
        assessment = ClientAssessment("unknown", 1, 0.5, (), (), 0.2, 0.5, ())
        for item in (assessment, replace(assessment, client_id="c1", round_id=0)):
            with self.assertRaises(ValueError):
                AggregationRequest(1, self.ref, self.compat, (self.update,), (item,), self.trusted, 11)

    def test_candidates_require_valid_weights_and_explicit_failures(self):
        fitness = FitnessVector(0.2, 0.1, 0.3, 0.4)
        CandidateEvaluation("ok", (ClientWeight("c1", 1.0),), fitness, True)
        CandidateEvaluation("failed", (), None, False, "nonfinite_update")
        for weights in ((ClientWeight("c1", -1),), (ClientWeight("c1", 0.5),),
                        (ClientWeight("c1", 0.5), ClientWeight("c1", 0.5))):
            with self.assertRaises(ValueError):
                CandidateEvaluation("bad", weights, fitness, True)
        with self.assertRaises(ValueError):
            FitnessVector(math.nan, 0.1, 0.2, 0.3)
        with self.assertRaises(ValueError):
            CandidateEvaluation("bad", (), None, False)


class ScopeTests(unittest.TestCase):
    def test_future_pipelines_are_metadata_only_and_dependencies_are_ordered(self):
        known = set()
        for stage in PIPELINES:
            self.assertLessEqual(set(stage.requires), known)
            known.add(stage.name)
        self.assertEqual({p.phase for p in PIPELINES if p.status == "available"}, {"A", "B", "C", "D", "E", "F", "G"})
        self.assertTrue(all(p.status == "planned" for p in PIPELINES if p.phase not in ("A", "B", "C", "D", "E", "F", "G")))

    def test_training_command_is_unavailable(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(["train"])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

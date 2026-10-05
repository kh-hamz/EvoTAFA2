"""Phase C client construction, preprocessing, leakage, and gate tests."""

import csv
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from edgefl.client_data.assignment import assign
from edgefl.client_data.config import load
from edgefl.client_data.local_split import split_local
from edgefl.client_data.preprocessing import FrozenPreprocessor, fit
from edgefl.client_data.storage import read_json, rows, sha256, write_json
from edgefl.client_data.validation import validate
from edgefl.client_data.policies import MANDATORY_EXCLUSIONS
from edgefl.contracts.phase_b import MANIFEST_FIELDS as PHASE_B_FIELDS
from edgefl.pipelines.catalog import PIPELINES


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, fields, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        output.writeheader()
        output.writerows(values)


class PhaseCFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ("frame.time", "ip.src_host", "ip.dst_host", "tcp.seq",
                      "tcp.dstport", "dns.qry.name", "tcp.payload",
                      "Attack_label", "Attack_type")
        self.selected = self.root / "selected.csv"
        self.splits = self.root / "splits.csv"
        selected_rows, split_rows = [], []
        record = 0
        for group_index in range(8):
            for member in range(4):
                record += 1
                label = "Normal" if member % 2 == 0 else "Backdoor"
                selected_rows.append((f"2021 10:00:{record:02d}.000", "10.0.0.1",
                    "10.0.0.2", str(record), "80", "alpha" if record % 3 else "beta",
                    f"payload-{record}", "0" if label == "Normal" else "1", label))
                split_rows.append(dict(zip(PHASE_B_FIELDS["splits"], (
                    "fold", f"oid-{record}", label, "capture", f"session-{group_index}",
                    f"group-{group_index}", "client_pool", ""))))
        for role, label in ((role, label) for role in ("trusted", "selection_validation", "final_test")
                            for label in ("Normal", "Backdoor")):
            record += 1
            selected_rows.append((f"2021 11:00:{record:02d}.000", "10.0.0.3",
                "10.0.0.4", str(record), "80", "gamma", f"protected-{record}",
                "0" if label == "Normal" else "1", label))
            split_rows.append(dict(zip(PHASE_B_FIELDS["splits"], (
                "fold", f"oid-{record}", label, "protected", f"protected-{record}",
                f"protected-{record}", role, ""))))
        with self.selected.open("w", newline="", encoding="utf-8") as stream:
            output = csv.writer(stream, lineterminator="\n")
            output.writerow(self.names)
            output.writerows(selected_rows)
        write_csv(self.splits, PHASE_B_FIELDS["splits"], split_rows)
        provenance_rows = []
        for number, row in enumerate(selected_rows, 1):
            label = row[-1]
            provenance_rows.append(dict(zip(PHASE_B_FIELDS["provenance"], (
                f"oid-{number}", number, f"key-{number}", f"exact-{number}",
                f"feature-{number}", f"evidence-{number}", label, "verified", 1,
                "unique_canonical_source"))))
        self.provenance = self.root / "provenance.csv"
        write_csv(self.provenance, PHASE_B_FIELDS["provenance"], provenance_rows)
        self.preprocessing = {
            "numeric_imputation": "mean", "scaling": "standard", "max_categories": 8,
            "identifier_payload_exclusions": sorted(MANDATORY_EXCLUSIONS),
            "strict_shortcut_exclusions": ["tcp.seq"],
        }
        self.config = {
            "seed": 42, "clients": 2,
            "scenarios": {"near_iid": {"kind": "near_iid"}},
            "max_partition_attempts": 4, "minimum_client_observations": 8,
            "minimum_training_observations": 8, "minimum_groups_per_client": 2,
            "local_validation_fraction": .2,
            "closed_set_scope": "supported_classes",
            "support_policy": {"profile": "server_scored", **{role: {"benign": 0, "attack": 0}
                               for role in ("assigned", "training", "validation")}},
            "near_iid_limits": {"maximum_size_deviation": .1, "maximum_total_variation": .05},
            "preprocessing": self.preprocessing,
        }

    def build(self):
        assignment_dir = self.root / "assignment"
        assignment_dir.mkdir()
        assignment = assign(self.splits, "fold", "near_iid", {"kind": "near_iid"},
                            assignment_dir, 42, 2, 4, 8, 2, True)
        local_dir = self.root / "local"
        local_dir.mkdir()
        local = split_local(assignment_dir / "assignments.csv", local_dir, 42, .2, 8, True)
        preprocessing_dir = self.root / "preprocessing"
        preprocessing_dir.mkdir()
        preprocessing = fit(local_dir / "local_splits.csv", self.provenance, self.selected,
                            preprocessing_dir, self.preprocessing)
        return assignment_dir, local_dir, preprocessing_dir, assignment, local, preprocessing


class AssignmentTests(PhaseCFixture):
    def test_all_scenarios_are_deterministic_and_preserve_groups(self):
        specifications = ({"kind": "near_iid"}, {"kind": "dirichlet", "alpha": .5},
                          {"kind": "specialist", "specialists_per_attack": 1})
        for index, specification in enumerate(specifications):
            first = self.root / f"first-{index}"
            second = self.root / f"second-{index}"
            first.mkdir(); second.mkdir()
            one = assign(self.splits, "fold", f"scenario-{index}", specification,
                         first, 42, 2, 4, 8, 2, True)
            two = assign(self.splits, "fold", f"scenario-{index}", specification,
                         second, 42, 2, 4, 8, 2, True)
            self.assertEqual(one, two)
            self.assertEqual(sha256(first / "assignments.csv"),
                             sha256(second / "assignments.csv"))
            ownership = {}
            for row in rows(first / "assignments.csv"):
                ownership.setdefault(row["group_id"], set()).add(row["client_id"])
            self.assertTrue(all(len(clients) == 1 for clients in ownership.values()))

    def test_impossible_minimum_is_rejected(self):
        directory = self.root / "rejected"
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, "observation minimum"):
            assign(self.splits, "fold", "near_iid", {"kind": "near_iid"},
                   directory, 42, 2, 2, 100, 2, True)


class LocalSplitTests(PhaseCFixture):
    def test_validation_is_group_isolated_and_training_counts_are_registered(self):
        assignment_dir, local_dir, _, _, report, _ = self.build()
        groups = {}
        roles = Counter()
        for row in rows(local_dir / "local_splits.csv"):
            groups.setdefault((row["client_id"], row["group_id"]), set()).add(row["partition"])
            roles[row["partition"]] += 1
        self.assertTrue(all(len(value) == 1 for value in groups.values()))
        self.assertEqual(report["registered_training_count"], roles["local_train"])
        self.assertFalse(report["local_validation_is_attackable"])
        self.assertEqual(sum(1 for _ in rows(assignment_dir / "assignments.csv")), sum(roles.values()))


class PreprocessingTests(PhaseCFixture):
    def test_fit_uses_only_local_training_and_freezes_two_feature_variants(self):
        _, local_dir, preprocessing_dir, _, _, report = self.build()
        local = {row["observation_id"]: row for row in rows(local_dir / "local_splits.csv")}
        fitting = list(rows(preprocessing_dir / "fit_rows.csv"))
        self.assertTrue(fitting)
        self.assertTrue(all(local[row["observation_id"]]["partition"] == "local_train"
                            for row in fitting))
        expected = sum(int(row["record"]) for row in fitting) / len(fitting)
        transformer = read_json(preprocessing_dir / "transformer.json")
        self.assertAlmostEqual(transformer["variants"]["base"]["numeric"]["tcp.seq"]["mean"],
                               expected)
        self.assertNotIn("tcp.seq", transformer["variants"]["strict"]["numeric"])
        self.assertIn("tcp.dstport", {item["field"] for item in
                      transformer["variants"]["base"]["removed_constant_fields"]})
        self.assertEqual(report["finite_transform_checks"]["base"], len(local))
        frozen = FrozenPreprocessor.load(preprocessing_dir / "transformer.json")
        with self.selected.open(encoding="utf-8", newline="") as stream:
            source = csv.reader(stream)
            first = next(source)
            values = next(source)
        self.assertTrue(all(math.isfinite(value) for value in
                            frozen.transform(tuple(first), tuple(values))))

    def test_preprocessing_is_byte_deterministic(self):
        _, local_dir, first, _, _, _ = self.build()
        second = self.root / "preprocessing-second"
        second.mkdir()
        fit(local_dir / "local_splits.csv", self.provenance, self.selected,
            second, self.preprocessing)
        for name in ("transformer.json", "label_mappings.json", "fit_rows.csv",
                     "feature_dictionary.csv", "feature_contract.json"):
            self.assertEqual(sha256(first / name), sha256(second / name), name)


class GateTests(PhaseCFixture):
    def test_complete_gate_passes_and_regenerates_every_phase_c_artifact(self):
        assignment_dir, local_dir, preprocessing_dir, _, _, _ = self.build()
        gate_dir = self.root / "gate"
        gate_dir.mkdir()
        report = validate(
            splits=self.splits, assignments=assignment_dir / "assignments.csv",
            local_splits=local_dir / "local_splits.csv",
            client_manifests=local_dir / "client_manifests.json", provenance=self.provenance,
            selected=self.selected, transformer=preprocessing_dir / "transformer.json",
            feature_dictionary=preprocessing_dir / "feature_dictionary.csv",
            feature_contract=preprocessing_dir / "feature_contract.json",
            label_mappings=preprocessing_dir / "label_mappings.json",
            fit_rows=preprocessing_dir / "fit_rows.csv", directory=gate_dir,
            config=self.config, fold="fold", scenario="near_iid")
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["eligible_for_training"])
        self.assertTrue(all(report["deterministic_regeneration"].values()))

    def test_local_validation_in_fitting_manifest_fails_gate(self):
        assignment_dir, local_dir, preprocessing_dir, _, _, _ = self.build()
        bad = self.root / "bad_fit.csv"
        existing = list(rows(preprocessing_dir / "fit_rows.csv"))
        validation = next(row for row in rows(local_dir / "local_splits.csv")
                          if row["partition"] == "local_validation")
        number = int(validation["observation_id"].split("-")[-1])
        existing.append({"client_id": validation["client_id"],
                         "observation_id": validation["observation_id"],
                         "record": number, "label": validation["label"]})
        write_csv(bad, ("client_id", "observation_id", "record", "label"), existing)
        gate_dir = self.root / "bad-gate"
        gate_dir.mkdir()
        report = validate(
            splits=self.splits, assignments=assignment_dir / "assignments.csv",
            local_splits=local_dir / "local_splits.csv",
            client_manifests=local_dir / "client_manifests.json", provenance=self.provenance,
            selected=self.selected, transformer=preprocessing_dir / "transformer.json",
            feature_dictionary=preprocessing_dir / "feature_dictionary.csv",
            feature_contract=preprocessing_dir / "feature_contract.json",
            label_mappings=preprocessing_dir / "label_mappings.json", fit_rows=bad,
            directory=gate_dir, config=self.config, fold="fold", scenario="near_iid")
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("Preprocessing used a non-training observation", report["errors"])


class ConfigurationAndScopeTests(unittest.TestCase):
    def test_repository_phase_c_configuration_and_catalogue(self):
        config = load(ROOT / "configs/phase_c.json", ROOT)
        self.assertEqual(config.values["clients"], 10)
        available = {stage.phase for stage in PIPELINES if stage.status == "available"}
        self.assertEqual(available, {"A", "B", "C"})
        self.assertTrue(all(stage.status == "planned" for stage in PIPELINES
                            if stage.phase not in ("A", "B", "C")))


if __name__ == "__main__":
    unittest.main()

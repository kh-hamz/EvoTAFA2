"""Trusted-role and feature-collision checks independent of orchestration."""

import tempfile
import unittest
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.data.grouping import group
from edgefl.data.schema import canonical
from edgefl.data.storage import rows, writer
from edgefl.data.trusted_panel import panel


def manifest(path, kind, entries):
    stream, out = writer(path, MANIFEST_FIELDS[kind])
    with stream:
        for entry in entries:
            out.writerow(entry)


class PanelRoleTests(unittest.TestCase):
    def test_panel_excludes_test_and_selection_and_reports_missing_benign(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for oid, role, label in (("1", "trusted", "Backdoor"),
                                     ("2", "final_test", "Normal"),
                                     ("3", "selection_validation", "Normal")):
                records.append(dict(fold="f", observation_id=oid, label=label,
                                    capture_id="capture", session_id=oid, group_id=oid,
                                    partition=role, reason=""))
            manifest(root / "splits.csv", "splits", records)
            support = {"folds": {"f": {"class_counts": {"Normal": {}, "Backdoor": {}}}}}
            report = panel(root / "splits.csv", support, root / "panel", 42, 512)
            self.assertFalse(report["folds"]["f"]["usable_for_fpr"])
            self.assertEqual([r["observation_id"] for r in rows(root / "panel/panel.csv")], ["1"])

    def test_panel_is_deterministic_and_respects_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [dict(fold="f", observation_id=str(i), label="Normal", capture_id="c",
                            session_id=str(i), group_id=str(i), partition="trusted", reason="")
                       for i in range(50)]
            manifest(root / "one.csv", "splits", records)
            manifest(root / "two.csv", "splits", reversed(records))
            support = {"folds": {"f": {"class_counts": {"Normal": {}}}}}
            panel(root / "one.csv", support, root / "a", 42, 5)
            panel(root / "two.csv", support, root / "b", 42, 5)
            self.assertEqual((root / "a/panel.csv").read_bytes(), (root / "b/panel.csv").read_bytes())
            self.assertEqual(len(list(rows(root / "a/panel.csv"))), 5)

    def test_confirmed_wireshark_boolean_equivalence_is_field_specific(self):
        self.assertEqual(canonical("tcp.flags.ack", "True"), "1")
        self.assertEqual(canonical("http.response", "False"), "0")
        self.assertEqual(canonical("tcp.payload", "True"), "True")

    def test_feature_collisions_are_not_duplicate_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources, evidence = [], []
            for oid, label in (("1", "Normal"), ("2", "Backdoor")):
                sources.append(dict(observation_id=oid, record=oid, record_key=oid,
                                    exact_key=oid, feature_key="same_features", evidence_key=oid,
                                    label=label, status="verified", candidate_count=1, reason=""))
                evidence.append(dict(observation_id=oid, label=label, status="verified", reason="",
                                     capture_id="capture", packet=oid, epoch=100 + int(oid),
                                     session_id="", session_start=100 + int(oid),
                                     session_end=100 + int(oid), evidence_key=oid))
            manifest(root / "provenance.csv", "provenance", sources)
            manifest(root / "evidence.csv", "evidence", evidence)
            report = group(root / "provenance.csv", root / "evidence.csv", root / "group", 300)
            retained = list(rows(root / "group/groups.csv"))
            self.assertTrue(all(r["status"] == "verified" for r in retained))
            self.assertEqual(retained[0]["group_id"], retained[1]["group_id"])
            self.assertEqual(report["conflicting_label_vectors"], 1)
            self.assertEqual(list(rows(root / "group/duplicates.csv")), [])

"""Independent regressions for the six Phase B/C readiness findings."""

import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from edgefl.client_data.assignment import assign, ClientState, GroupProfile, _heterogeneity
from edgefl.client_data.local_split import split_local
from edgefl.client_data.policies import MANDATORY_EXCLUSIONS
from edgefl.client_data.preprocessing import fit
from edgefl.client_data.storage import load_completion, load_training_ready, read_json, write_json, sha256, rows
from edgefl.client_data.experiments import resolve_experiment
from edgefl.contracts.phase_c import SupportPolicy
from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.data.capture_diagnostics import CaptureDiagnostics, clock_offset_exact
from edgefl.data.integrity import VerificationContext
from edgefl.pipelines import phase_c_assign, phase_c_local_split, phase_c_preprocessing, phase_c_validate
from test_phase_c import PhaseCFixture, write_csv
import test_phase_c_pipeline as phase_c_fixture


class IntegrityRegressions(PhaseCFixture):
    signed_phase_b = phase_c_fixture.PhaseCOrchestrationTests.signed_phase_b

    def build_gate(self, task="multiclass"):
        with contextlib.redirect_stdout(io.StringIO()):
            config, validation, split, provenance = self.signed_phase_b()
            assignment = phase_c_assign.execute(config, validation, split, "primary", "fold", "near_iid", task)
            local = phase_c_local_split.execute(config, assignment, "primary")
            preprocessing = phase_c_preprocessing.execute(config, local, provenance, "primary")
            inputs = {"phase_b_validation": validation, "split": split, "provenance": provenance,
                      "assignments": assignment, "local_split": local, "preprocessing": preprocessing,
                      "fold": "fold", "scenario": "near_iid"}
            gate = phase_c_validate.execute(config, inputs, "primary")
        return config, gate, assignment, inputs

    def changed_source(self, filename):
        config, gate, _, _ = self.build_gate()
        self.assertTrue(load_completion(config, gate, "validate-phase-c", "primary")["eligible_for_training"])
        with (self.root / "archive" / filename).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Registered source changed"):
            load_completion(config, gate, "validate-phase-c", "primary")
        with self.assertRaisesRegex(ValueError, "Registered source changed"):
            load_training_ready(config, gate, "primary")

    def test_changed_selected_csv_invalidates_cached_pass(self):
        self.changed_source("selected.csv")

    def test_changed_source_csv_invalidates_cached_pass(self):
        self.changed_source("source.csv")

    def test_changed_pcap_invalidates_cached_pass(self):
        self.changed_source("source.pcap")

    def test_registry_is_hashed_once_in_one_load(self):
        config, gate, _, _ = self.build_gate()
        from edgefl.data.inventory import verify_sources
        with patch("edgefl.data.integrity.verify_sources", wraps=verify_sources) as verifier:
            load_training_ready(config, gate, "primary")
            self.assertEqual(verifier.call_count, 1)

    def test_command_uses_one_verification_context(self):
        config, _, _, inputs = self.build_gate()
        from edgefl.data.inventory import verify_sources
        with patch("edgefl.data.integrity.verify_sources", wraps=verify_sources) as verifier:
            with contextlib.redirect_stdout(io.StringIO()):
                phase_c_validate.execute(config, inputs, "primary", context=VerificationContext())
            self.assertEqual(verifier.call_count, 1)

    def test_training_bundle_checks_requested_task_and_fold(self):
        config, gate, _, _ = self.build_gate()
        bundle = load_training_ready(config, gate, "primary", task="multiclass", fold="fold")
        self.assertEqual(bundle.task, "multiclass")
        self.assertIn("trusted_panel", {ref[0] for ref in bundle.references})
        with self.assertRaisesRegex(ValueError, "task mismatch"):
            load_training_ready(config, gate, "primary", task="binary")
        with self.assertRaisesRegex(ValueError, "fold mismatch"):
            load_training_ready(config, gate, "primary", fold="other")

    def test_forged_training_export_is_rejected_even_with_updated_hash(self):
        config, gate, _, _ = self.build_gate()
        metadata = read_json(gate)
        export_path = self.root / metadata["artifacts"]["training_inputs"]["path"]
        exported = read_json(export_path)
        exported["eligible_for_training"] = False
        write_json(export_path, exported)
        metadata["artifacts"]["training_inputs"]["sha256"] = sha256(export_path)
        write_json(gate, metadata)
        with self.assertRaisesRegex(ValueError, "eligibility disagrees"):
            load_completion(config, gate, "validate-phase-c", "primary")

    def test_intermediate_stage_cannot_enable_training(self):
        config, _, assignment, _ = self.build_gate()
        metadata = read_json(assignment)
        metadata["eligible_for_training"] = True
        write_json(assignment, metadata)
        with self.assertRaisesRegex(ValueError, "Only the pretraining gate"):
            load_completion(config, assignment, "assign-clients", "primary")

    def test_missing_pretraining_report_is_rejected(self):
        config, gate, _, _ = self.build_gate()
        metadata = read_json(gate)
        del metadata["artifacts"]["pretraining_report"]
        write_json(gate, metadata)
        with self.assertRaisesRegex(ValueError, "Missing training-readiness"):
            load_completion(config, gate, "validate-phase-c", "primary")

    def test_wrong_upstream_stage_is_rejected(self):
        config, gate, _, _ = self.build_gate()
        metadata = read_json(gate)
        metadata["inputs"]["assignments"] = metadata["inputs"]["local_split"]
        write_json(gate, metadata)
        with self.assertRaisesRegex(ValueError, "upstream stage mismatch"):
            load_completion(config, gate, "validate-phase-c", "primary")

    def test_dataset_mismatch_is_rejected(self):
        config, gate, _, _ = self.build_gate()
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            load_training_ready(config, gate, "smoke")

    def test_entire_protocol_b_binary_pipeline_can_pass_without_fifteen_labels(self):
        with contextlib.redirect_stdout(io.StringIO()):
            config, validation, _, provenance = self.signed_phase_b()
            b_split = self.root / read_json(validation)["inputs"]["splits_b"]["path"]
            assignment = phase_c_assign.execute(config, validation, b_split, "primary", "fold", "near_iid", "binary")
            local = phase_c_local_split.execute(config, assignment, "primary")
            preprocessing = phase_c_preprocessing.execute(config, local, provenance, "primary")
            gate = phase_c_validate.execute(config, {"phase_b_validation": validation, "split": b_split,
                "provenance": provenance, "assignments": assignment, "local_split": local,
                "preprocessing": preprocessing, "fold": "fold", "scenario": "near_iid"}, "primary")
        self.assertEqual(load_training_ready(config, gate, "primary").task, "binary")

    def test_stale_contract_version_requires_regeneration(self):
        config, gate, _, _ = self.build_gate()
        metadata = read_json(gate)
        metadata["schema_version"] = "phase-c.v1"
        write_json(gate, metadata)
        with self.assertRaisesRegex(ValueError, "completion contract"):
            load_completion(config, gate, "validate-phase-c", "primary")

    def test_full_15_scope_rejects_supported_subset(self):
        with contextlib.redirect_stdout(io.StringIO()):
            config, validation, split, _ = self.signed_phase_b()
        values = dict(config.values)
        values["closed_set_scope"] = "full_15"
        summary = read_json(self.root / read_json(split)["artifacts"]["summary"]["path"])
        with self.assertRaisesRegex(ValueError, "Full-15"):
            resolve_experiment(values, "primary", read_json(split), summary, "fold", "multiclass")

    def test_protocol_b_binary_does_not_require_heldout_attack_label(self):
        with contextlib.redirect_stdout(io.StringIO()):
            config, validation, _, _ = self.signed_phase_b()
        metadata = read_json(validation)
        split = self.root / metadata["inputs"]["splits_b"]["path"]
        value = read_json(split)
        summary = read_json(self.root / value["artifacts"]["summary"]["path"])
        summary["folds"]["fold"]["held_out_captures"] = ["heldout_xss"]
        contract = resolve_experiment(config.values, "primary", value, summary, "fold", "binary")
        self.assertEqual(contract["labels"], ["Normal", "Attack"])
        self.assertNotIn("XSS", contract["labels"])
        with self.assertRaisesRegex(ValueError, "binary evaluation only"):
            resolve_experiment(config.values, "primary", value, summary, "fold", "multiclass")


class AllocationRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, groups):
        path = self.root / "splits.csv"
        def records():
            number = 0
            for index, (size, label) in enumerate(groups):
                for _ in range(size):
                    number += 1
                    yield dict(zip(MANIFEST_FIELDS["splits"], ("fold", str(number), label, "capture",
                               f"session-{index}", f"group-{index}", "client_pool", "")))
        write_csv(path, MANIFEST_FIELDS["splits"], records())
        return path

    def test_pure_label_fixture_has_exact_balanced_assignment(self):
        path = self.manifest([(100, "Normal" if i % 2 else "Backdoor") for i in range(200)])
        report = assign(path, "fold", "near_iid", {"kind": "near_iid"}, self.root / "out",
                        42, 10, 4, 1000, 2, False)
        self.assertEqual([value["observations"] for value in report["clients"].values()], [2000] * 10)
        for value in report["clients"].values():
            self.assertEqual(value["class_counts"]["Normal"], 1000)
            self.assertEqual(value["class_counts"]["Backdoor"], 1000)
        self.assertEqual(report["heterogeneity"]["maximum_total_variation"], 0)
        self.assertEqual(report["heterogeneity"]["maximum_size_deviation"], 0)

    def test_distance_reference_and_size_formula(self):
        a, b = ClientState(), ClientState()
        a.add(GroupProfile("a", 12, (("Normal", 9), ("Backdoor", 3))))
        b.add(GroupProfile("b", 8, (("Normal", 1), ("Backdoor", 7))))
        metrics = _heterogeneity({"a": a, "b": b})
        self.assertEqual(metrics["target_size"], 10)
        self.assertAlmostEqual(metrics["maximum_size_deviation"], .2)
        self.assertAlmostEqual(metrics["client_total_variation"]["a"], .25)
        self.assertAlmostEqual(metrics["client_total_variation"]["b"], .375)

    def test_search_exhaustion_retains_only_ineligible_diagnostic(self):
        path = self.manifest([(90, "Normal"), (5, "Normal"), (3, "Normal"), (2, "Backdoor")])
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "infeasibility not proven"):
            assign(path, "fold", "near_iid", {"kind": "near_iid"}, output, 42, 2, 2, 1, 2, False)
        self.assertEqual(read_json(output / "allocation_failure.json")["outcome"], "search_exhausted")
        self.assertFalse(read_json(output / "best_candidate.json")["eligible_for_training"])
        self.assertFalse((output / "assignments.csv").exists())

    def test_sparse_specialist_and_local_split_are_valid(self):
        path = self.manifest([(100, "Backdoor") for _ in range(8)])
        output = self.root / "out"
        report = assign(path, "fold", "specialist", {"kind": "specialist", "specialists_per_attack": 1},
                        output, 42, 2, 4, 200, 2, False, policy=SupportPolicy())
        self.assertTrue(all(value["class_counts"]["Normal"] == 0 for value in report["clients"].values()))
        local = split_local(output / "assignments.csv", self.root / "local", 42, .2, 100, False)
        for value in local["clients"].values():
            self.assertEqual(value["training_class_counts"]["Normal"], 0)
            self.assertFalse(value["metric_support"]["validation"]["binary_roc_auc_pr_auc"])
        profile = SupportPolicy("local_binary_metrics", (1, 1), (1, 1), (1, 1))
        with self.assertRaisesRegex(ValueError, "search exhausted"):
            split_local(output / "assignments.csv", self.root / "strict", 42, .2, 100, False, policy=profile, attempts=2)


class PredictorRegressions(PhaseCFixture):
    def test_varying_arp_addresses_never_become_features(self):
        names = (*self.names[:-2], "arp.src.proto_ipv4", "arp.dst.proto_ipv4", *self.names[-2:])
        import csv
        with self.selected.open(newline="") as stream:
            source = list(csv.reader(stream))[1:]
        records = [(*row[:-2], f"10.0.1.{i+1}", f"10.0.2.{i+1}", *row[-2:]) for i, row in enumerate(source)]
        with self.selected.open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(names); writer.writerows(records)
        _, _, preprocessing, _, _, _ = self.build()
        for variant in read_json(preprocessing / "transformer.json")["variants"].values():
            self.assertFalse({item["source"] for item in variant["output_features"]} & MANDATORY_EXCLUSIONS)

    def test_removing_address_exclusion_is_rejected(self):
        values = dict(self.preprocessing)
        values["identifier_payload_exclusions"] = sorted(MANDATORY_EXCLUSIONS - {"arp.src.proto_ipv4"})
        from edgefl.client_data.policies import validate_feature_policy
        with self.assertRaisesRegex(ValueError, "exclusions missing"):
            validate_feature_policy(values)


class DiagnosticRegressions(unittest.TestCase):
    def test_exact_clock_arithmetic_preserves_nanoseconds_and_midnight(self):
        self.assertEqual(str(clock_offset_exact("2021 01:00:00.000000123", "1609459200.000000023")), "3600.000000100")
        self.assertEqual(clock_offset_exact("2021 00:00:00.000000000", "1609545600"), 0)
        self.assertIsNone(clock_offset_exact("6.0", "1609459200"))

    def test_small_regression_is_measured_without_authorizing_recovery(self):
        diagnostic = CaptureDiagnostics()
        diagnostic.packet(1, "1609459200.000002")
        anomaly = diagnostic.packet(2, "1609459200.000001")
        self.assertEqual(anomaly["backward_seconds"], "0.000001")
        self.assertFalse(diagnostic.summary()["recovery_authorized"])

    def test_diagnostic_mode_cannot_retain_observations(self):
        from test_phase_b import PipelineTests
        fixture = PipelineTests("test_timestamp_reset_blocks_capture")
        fixture.setUp()
        try:
            audit, provenance = fixture.audit_provenance()
            from edgefl.pipelines.phase_b_diagnostics import execute
            from edgefl.data import pcap
            identity = {"path": "fake", "sha256": "0" * 64, "version": "fixture"}
            def summary(arguments, output, timeout=1800):
                output.write_text("fixture capture")
            with patch.object(pcap, "tool_identity", return_value=identity), patch.object(pcap, "run_tool", side_effect=summary), patch.object(pcap, "extract", side_effect=fixture.fake_extract):
                result = execute(fixture.config, audit, provenance, "primary")
            report = fixture.summary(result)
            self.assertEqual(report["statuses"], {"quarantined": 321})
            self.assertFalse(report["recovery_authorized"])
        finally:
            fixture.doCleanups()


class CompleteLineageRegression(unittest.TestCase):
    def test_actual_phase_b_validation_feeds_closed_set_and_unseen_binary_gates(self):
        """Exercise real stage logic; only the external Wireshark adapter is faked."""
        import test_phase_b as b_fixture
        from edgefl.client_data.config import load
        from edgefl.pipelines import phase_b_group, phase_b_split, phase_b_panel, phase_b_validate
        fixture = b_fixture.PipelineTests("test_complete_pipeline_and_source_preservation")
        fixture.setUp()
        try:
            audit, provenance = fixture.audit_provenance()
            evidence = fixture.capture(audit, provenance)
            groups = phase_b_group.execute(fixture.config, provenance, evidence, "primary")
            inputs = {"audit": audit, "provenance": provenance, "evidence": evidence, "groups": groups}
            for protocol in ("a", "b"):
                split = phase_b_split.execute(fixture.config, groups, "primary", protocol.upper())
                inputs["splits_" + protocol] = split
                inputs["panel_" + protocol] = phase_b_panel.execute(fixture.config, split, "primary")
            accepted = phase_b_validate.execute(fixture.config, inputs, "primary")
            self.assertEqual(fixture.summary(accepted)["status"], "PASS_WITH_LIMITATIONS")
            values = json.loads((Path(__file__).resolve().parents[1] / "configs/phase_c.json").read_text())
            values.update(clients=2, minimum_client_observations=10, minimum_training_observations=8,
                          output="out_c", max_partition_attempts=4)
            values["preprocessing"]["strict_shortcut_exclusions"] = ["tcp.seq"]
            write_json(fixture.root / "configs/phase_c.json", values)
            config = load(fixture.root / "configs/phase_c.json", fixture.root)
            for protocol, fold, task in (("a", "protocol_a", "multiclass"), ("b", "unseen_attack2", "binary")):
                assignment = phase_c_assign.execute(config, accepted, inputs["splits_" + protocol],
                                                    "primary", fold, "near_iid", task)
                local = phase_c_local_split.execute(config, assignment, "primary")
                preprocessing = phase_c_preprocessing.execute(config, local, provenance, "primary")
                gate = phase_c_validate.execute(config, {"phase_b_validation": accepted,
                    "split": inputs["splits_" + protocol], "provenance": provenance, "assignments": assignment,
                    "local_split": local, "preprocessing": preprocessing, "fold": fold, "scenario": "near_iid"}, "primary")
                bundle = load_training_ready(config, gate, "primary", task=task, fold=fold)
                self.assertEqual(bundle.task, task)
                if task == "binary":
                    labels = read_json(fixture.root / read_json(preprocessing)["artifacts"]["label_mappings"]["path"])
                    self.assertNotIn("DDoS_TCP", labels["multiclass"])
        finally:
            fixture.doCleanups()

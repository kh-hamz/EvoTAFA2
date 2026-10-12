"""Phase D orchestration over source-verified synthetic B/C fixtures."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_phase_c_pipeline as fixtures
from test_phase_c import write_csv
from edgefl.data.storage import read_json, write_json, rows
from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.pipelines import phase_c_assign, phase_c_local_split, phase_c_preprocessing, phase_c_validate
from edgefl.pipelines import phase_d_prepare, phase_d_initialize, phase_d_centralized, phase_d_federated, phase_d_review, phase_d_validate
from edgefl.learning.config import load
from edgefl.learning.storage import load_stage, artifact
from edgefl.learning.data import LearningData
from edgefl.learning import checkpoints

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setup_data(self, task="binary", device="cpu"):
        fixture = fixtures.PhaseCOrchestrationTests("test_all_stages_publish_verified_lineage_and_only_gate_enables_training")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        with fixture.selected.open(newline="", encoding="utf-8") as stream:
            original = list(csv.reader(stream))
        names, values = original[0], original[1:]
        splits = list(rows(fixture.splits))
        provenance = list(rows(fixture.provenance))
        for repetition in range(4):
            for i, source in enumerate(values[:32]):
                row = list(source)
                number = len(values) + 1
                row[3] = str(number)
                values.append(row)
                label = row[-1]
                splits.append(dict(zip(MANIFEST_FIELDS["splits"], ("fold", f"oid-{number}", label,
                    "capture", f"extra-session-{repetition}-{i//4}", f"extra-group-{repetition}-{i//4}", "client_pool", ""))))
                provenance.append(dict(zip(MANIFEST_FIELDS["provenance"], (f"oid-{number}", number,
                    f"key-{number}", f"exact-{number}", f"feature-{number}", f"evidence-{number}", label,
                    "verified", 1, "unique_canonical_source"))))
        for row in values:
            row[4] = "1" if row[-1] == "Normal" else "9"
        with fixture.selected.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows(values)
        write_csv(fixture.splits, MANIFEST_FIELDS["splits"], splits)
        write_csv(fixture.provenance, MANIFEST_FIELDS["provenance"], provenance)
        fixture.config["clients"] = 5
        with contextlib.redirect_stdout(io.StringIO()):
            c, validation, split, provenance_path = fixture.signed_phase_b()
            assignment = phase_c_assign.execute(c, validation, split, "primary", "fold", "near_iid", task)
            local = phase_c_local_split.execute(c, assignment, "primary")
            preprocessing = phase_c_preprocessing.execute(c, local, provenance_path, "primary")
            self.gate = phase_c_validate.execute(c, {"phase_b_validation": validation, "split": split,
                "provenance": provenance_path, "assignments": assignment, "local_split": local,
                "preprocessing": preprocessing, "fold": "fold", "scenario": "near_iid"}, "primary")
        values = read_json(ROOT / "configs/phase_d.json")
        values.update(output="out_d", device=device, threads=1, rounds=2, centralized_epochs=2, batch_size=16)
        write_json(self.root / "configs/phase_d.json", values)
        self.config = load(self.root / "configs/phase_d.json", self.root)
        self.prepared = phase_d_prepare.execute(self.config, self.gate, "primary", task, "fold", "near_iid")
        self.initialization = phase_d_initialize.execute(self.config, self.prepared, 11)

    def central_review(self):
        run = phase_d_centralized.execute(self.config, self.prepared, self.initialization, "mlp", 11)
        self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
        review = phase_d_review.execute(self.config, run, "acceptable", "Synthetic fixture: finite updates and checked memberships.")
        return run, review

    def test_all_eight_methods_and_scoped_acceptance(self):
        self.setup_data()
        mlp, central = self.central_review()
        reports = [central]
        for method in ("majority", "logistic"):
            result = phase_d_centralized.execute(self.config, self.prepared, self.initialization, method, 11)
            reports.append(phase_d_review.execute(self.config, result, "acceptable", "Synthetic behavior verified."))
        fedavg = phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [central])
        self.assertEqual(read_json(fedavg)["summary"]["correctness"], "PASS", read_json(fedavg)["summary"])
        average_review = phase_d_review.execute(self.config, fedavg, "acceptable", "Synthetic FedAvg checks passed.")
        reports.append(average_review)
        for method in ("fedprox", "median", "trimmed_mean", "fltrust"):
            result = phase_d_federated.execute(self.config, self.prepared, self.initialization, method, 11, [central, average_review])
            self.assertEqual(read_json(result)["summary"]["correctness"], "PASS", read_json(result)["summary"])
            reports.append(phase_d_review.execute(self.config, result, "acceptable", "Synthetic aggregation verified."))
        from edgefl.learning.storage import implementation_hash
        test_report = self.root / "fixture-tests.json"
        write_json(test_report, {"schema_version": "phase-d.tests.v1", "tests": 1, "successful": True,
                                 "failures": 0, "errors": 0, "implementation_sha256": implementation_hash(),
                                 "evidence_kind": "synthetic_validator_fixture"})
        accepted = phase_d_validate.execute(self.config, reports, test_report)
        self.assertEqual(read_json(accepted)["summary"]["status"], "PASS")
        loaded = load_stage(self.config, self.prepared)
        data = LearningData(self.config, loaded)
        self.assertEqual(len(data.view("local_train")) + len(data.view("local_validation")), 160)
        with self.assertRaises(ValueError):
            data.view("final_test")
        members = list(rows(artifact(self.config, loaded, "learning_rows.csv")))
        self.assertNotIn("final_test", {r["role"] for r in members})

    def test_changed_source_and_array_tampering_rejected(self):
        self.setup_data()
        metadata = load_stage(self.config, self.prepared)
        array = next(ref for key, ref in metadata["artifacts"].items() if key.endswith("-x.npy"))
        array_path = self.root / array["path"]
        original = array_path.read_bytes()
        array_path.write_bytes(original + b"tampering")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_stage(self.config, self.prepared)
        array_path.write_bytes(original)
        selected = self.root / "archive/selected.csv"
        with selected.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(ValueError):
            load_stage(self.config, self.prepared)

    def test_review_prerequisites_and_multiclass_path(self):
        self.setup_data("multiclass")
        with self.assertRaisesRegex(ValueError, "centralized"):
            phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11)
        run, good = self.central_review()
        unresolved = phase_d_review.execute(self.config, run, "unresolved", "Investigate weak learning.")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [unresolved])
        result = phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [good])
        self.assertEqual(read_json(result)["summary"]["correctness"], "PASS")

    def test_resume_matches_uninterrupted_cpu(self):
        self.setup_data()
        _, review = self.central_review()
        complete = phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [review])
        marker = complete.parent / "resume-0001.json"
        resumed = phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [review], marker)
        a, b = [checkpoints.load(self.root / read_json(p)["summary"]["final_checkpoint"]["path"])["model"] for p in (complete, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_wrong_dependency_and_checkpoint_marker_rejected(self):
        self.setup_data()
        metadata = read_json(self.initialization)
        original = dict(metadata["inputs"])
        metadata["inputs"] = {}
        write_json(self.initialization, metadata)
        with self.assertRaisesRegex(ValueError, "dependencies"):
            load_stage(self.config, self.initialization)
        metadata["inputs"] = original
        write_json(self.initialization, metadata)
        run, _ = self.central_review()
        marker = run.parent / "resume-0001.json"
        altered = read_json(marker)
        altered["implementation_sha256"] = "0" * 64
        write_json(marker, altered)
        with self.assertRaisesRegex(ValueError, "implementation"):
            phase_d_centralized.execute(self.config, self.prepared, self.initialization, "mlp", 11, resume=marker)

    def test_failed_round_cannot_receive_acceptable_review(self):
        self.setup_data()
        _, central = self.central_review()
        average = phase_d_federated.execute(self.config, self.prepared, self.initialization, "fedavg", 11, [central])
        reviewed = phase_d_review.execute(self.config, average, "acceptable", "Synthetic reference checks.")
        with patch("edgefl.learning.aggregation.reduce_updates", side_effect=ValueError("Insufficient clients")):
            failed = phase_d_federated.execute(self.config, self.prepared, self.initialization, "trimmed_mean", 11, [central, reviewed])
        self.assertEqual(read_json(failed)["summary"]["correctness"], "FAIL")
        self.assertTrue((failed.parent / "failed_round.json").exists())
        with self.assertRaisesRegex(ValueError, "override"):
            phase_d_review.execute(self.config, failed, "acceptable", "Cannot approve a failed run.")


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_execution_and_resume(self):
        self.setup_data(device="cuda")
        complete, _ = self.central_review()
        resumed = phase_d_centralized.execute(self.config, self.prepared, self.initialization, "mlp", 11,
                                              resume=complete.parent / "resume-0001.json")
        a, b = [checkpoints.load(self.root / read_json(p)["summary"]["final_checkpoint"]["path"])["model"] for p in (complete, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


class CompleteLineageTests(unittest.TestCase):
    def test_actual_a_b_c_d_binary_and_multiclass(self):
        import test_phase_b as b_fixture
        from edgefl.client_data.config import load as load_c
        from edgefl.pipelines import phase_b_group, phase_b_split, phase_b_panel, phase_b_validate
        fixture = b_fixture.PipelineTests("test_complete_pipeline_and_source_preservation")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with contextlib.redirect_stdout(io.StringIO()):
            audit, provenance = fixture.audit_provenance()
            evidence = fixture.capture(audit, provenance)
            groups = phase_b_group.execute(fixture.config, provenance, evidence, "primary")
            inputs = {"audit": audit, "provenance": provenance, "evidence": evidence, "groups": groups}
            for protocol in ("a", "b"):
                inputs["splits_" + protocol] = phase_b_split.execute(fixture.config, groups, "primary", protocol.upper())
                inputs["panel_" + protocol] = phase_b_panel.execute(fixture.config, inputs["splits_" + protocol], "primary")
            accepted = phase_b_validate.execute(fixture.config, inputs, "primary")
            cv = read_json(ROOT / "configs/phase_c.json")
            cv.update(clients=2, minimum_client_observations=10, minimum_training_observations=8, output="out_c", max_partition_attempts=4)
            cv["preprocessing"]["strict_shortcut_exclusions"] = ["tcp.seq"]
            write_json(fixture.root / "configs/phase_c.json", cv)
            c = load_c(fixture.root / "configs/phase_c.json", fixture.root)
            dv = read_json(ROOT / "configs/phase_d.json")
            dv.update(output="out_d", device="cpu", threads=1, rounds=1, centralized_epochs=1)
            write_json(fixture.root / "configs/phase_d.json", dv)
            d = load(fixture.root / "configs/phase_d.json", fixture.root)
            for protocol, fold, task in (("a", "protocol_a", "multiclass"), ("b", "unseen_attack2", "binary")):
                assignment = phase_c_assign.execute(c, accepted, inputs["splits_" + protocol], "primary", fold, "near_iid", task)
                local = phase_c_local_split.execute(c, assignment, "primary")
                preprocessing = phase_c_preprocessing.execute(c, local, provenance, "primary")
                gate = phase_c_validate.execute(c, {"phase_b_validation": accepted, "split": inputs["splits_" + protocol],
                    "provenance": provenance, "assignments": assignment, "local_split": local,
                    "preprocessing": preprocessing, "fold": fold, "scenario": "near_iid"}, "primary")
                prepared = phase_d_prepare.execute(d, gate, "primary", task, fold, "near_iid")
                initialization = phase_d_initialize.execute(d, prepared, 11)
                run = phase_d_centralized.execute(d, prepared, initialization, "mlp", 11)
                self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
                review = phase_d_review.execute(d, run, "acceptable", "Synthetic lineage and finite training verified.")
                federated = phase_d_federated.execute(d, prepared, initialization, "fedavg", 11, [review])
                self.assertEqual(read_json(federated)["summary"]["correctness"], "PASS")

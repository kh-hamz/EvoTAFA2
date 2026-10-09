"""Source-verified synthetic E integration, not real-dataset research evidence."""

import contextlib
import io
import sys
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
import test_phase_d_pipeline as d_fixture
from edgefl.data.storage import read_json, write_json
from edgefl.learning import checkpoints
from edgefl.learning.storage import implementation_hash as d_hash
from edgefl.pipelines.phaseD import phase_d_centralized, phase_d_federated, phase_d_review, phase_d_validate
from edgefl.pipelines.phaseE import (phase_e_prepare, phase_e_attack_plan, phase_e_pilot, phase_e_calibrate,
                             phase_e_review, phase_e_federated, phase_e_assess, phase_e_validate)
from edgefl.trust.config import load
from edgefl.trust.storage import load_stage, implementation_hash, regression_hash, artifact
from edgefl.contracts.phase_e import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]


def build_fixture(owner, task="binary", device="cpu"):
    fixture = d_fixture.PipelineTests("test_all_eight_methods_and_scoped_acceptance")
    owner(fixture.doCleanups)
    fixture.setup_data(task, device)
    _, central = fixture.central_review()
    reviews = [central]
    for method in ("majority", "logistic"):
        run = phase_d_centralized.execute(fixture.config, fixture.prepared, fixture.initialization, method, 11)
        reviews.append(phase_d_review.execute(fixture.config, run, "acceptable", "Synthetic numerical fixture."))
    average = phase_d_federated.execute(fixture.config, fixture.prepared, fixture.initialization, "fedavg", 11, [central])
    avg_review = phase_d_review.execute(fixture.config, average, "acceptable", "Synthetic numerical fixture.")
    reviews.append(avg_review)
    for method in ("fedprox", "median", "trimmed_mean", "fltrust"):
        run = phase_d_federated.execute(fixture.config, fixture.prepared, fixture.initialization, method, 11, [central, avg_review])
        reviews.append(phase_d_review.execute(fixture.config, run, "acceptable", "Synthetic numerical fixture."))
    report = fixture.root / "fixture-d-tests.json"
    write_json(report, {"schema_version": "phase-d.tests.v1", "tests": 1, "successful": True, "failures": 0,
                       "errors": 0, "implementation_sha256": d_hash(), "evidence_kind": "synthetic_validator_fixture"})
    fixture.acceptance = phase_d_validate.execute(fixture.config, reviews, report)
    settings = read_json(ROOT / "configs/phase_e.json")
    settings.update(output="out_e", activation_round=2, intermittent_period=1)
    write_json(fixture.root / "configs/phase_e.json", settings)
    fixture.e = load(fixture.root / "configs/phase_e.json", fixture.root)
    fixture.metadata = phase_e_prepare.execute(fixture.e, fixture.prepared, fixture.acceptance)
    pilots = {seed: phase_e_pilot.execute(fixture.e, fixture.metadata, fixture.acceptance, seed) for seed in (101, 102, 103)}
    fixture.calibration = phase_e_calibrate.execute(fixture.e, fixture.metadata, [pilots[101], pilots[102]], pilots[103])
    fixture.pilots = pilots
    fixture.calibration_review = phase_e_review.execute(fixture.e, fixture.calibration, "acceptable", "Synthetic calibration correctness and independent audit checked.")
    return fixture


def execute(f, condition="clean", method="adaptive", resume=None, plan=None):
    plan = plan or phase_e_attack_plan.execute(f.e, f.metadata, 11, condition)
    run = phase_e_federated.execute(f.e, f.metadata, f.initialization, f.acceptance, plan, f.calibration,
                                   f.calibration_review, method, 11, resume)
    return run, plan


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = build_fixture(cls.addClassCleanup)

    def test_all_conditions_and_weighted_methods_receive_scoped_acceptance(self):
        f, reviews = self.f, []
        for condition in CONDITIONS:
            plan = phase_e_attack_plan.execute(f.e, f.metadata, 11, condition)
            for method in ("fedavg", "adaptive", "frozen"):
                with self.subTest(condition=condition, method=method):
                    run, _ = execute(f, condition, method, plan=plan)
                    summary = read_json(run)["summary"]
                    self.assertEqual(summary["correctness"], "PASS", summary)
                    self.assertEqual(summary["attack_coverage"], "PASS", summary)
                    first = read_json(run.parent / "assessment-0001.json")
                    second = read_json(run.parent / "assessment-0002.json")
                    self.assertTrue(all(v == .5 for v in first["prior_reputation"].values()))
                    prior = read_json(run.parent / "reputation-0001.json")["state"]
                    self.assertEqual(second["prior_reputation"], {c: e["value"] for c, e in prior.items()})
                    if method == "frozen":
                        a = read_json(run.parent / "aggregation-0001.json")["weights"]
                        b = read_json(run.parent / "aggregation-0002.json")["weights"]
                        for client in a:
                            self.assertAlmostEqual(a[client], b[client])
                    reviews.append(phase_e_review.execute(f.e, run, "acceptable", "Synthetic controlled condition verified."))
        report = f.root / "fixture-e-tests.json"
        write_json(report, {"schema_version": "phase-e.tests.v1", "tests": 1, "successful": True, "failures": 0,
                           "errors": 0, "skipped": 0, "implementation_sha256": implementation_hash(),
                           "regression_sha256": regression_hash(), "evidence_kind": "synthetic_validator_fixture"})
        accepted = phase_e_validate.execute(f.e, f.metadata, f.calibration, f.calibration_review, reviews, report, 11)
        self.assertEqual(read_json(accepted)["summary"]["status"], "PASS")
        missing = phase_e_validate.execute(f.e, f.metadata, f.calibration, f.calibration_review, reviews[:-1], report, 11)
        self.assertEqual(read_json(missing)["summary"]["status"], "FAIL")
        evidence = read_json(report)
        write_json(report, {**evidence, "regression_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "software evidence"):
            phase_e_validate.execute(f.e, f.metadata, f.calibration, f.calibration_review, reviews, report, 11)

    def test_baseline_adapters_and_exact_resumed_frozen_state(self):
        f = self.f
        for method in ("fedprox", "median", "trimmed_mean", "fltrust"):
            run, _ = execute(f, method=method)
            self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
        original, plan = execute(f, "sign_flip", "frozen")
        resumed, _ = execute(f, "sign_flip", "frozen", original.parent / "resume-0001.json", plan)
        again, _ = execute(f, "sign_flip", "frozen", resumed.parent / "resume-0002.json", plan)
        payloads = [checkpoints.load(f.root / read_json(p)["summary"]["final_checkpoint"]["path"]) for p in (original, resumed, again)]
        for payload in payloads[1:]:
            for name in payloads[0]["model"]:
                torch.testing.assert_close(payloads[0]["model"][name], payload["model"][name], rtol=0, atol=0)
            self.assertEqual({k: v for k, v in payloads[0]["phase_state"].items() if k != "round_artifacts"},
                             {k: v for k, v in payload["phase_state"].items() if k != "round_artifacts"})
        self.assertEqual(read_json(again)["summary"]["attack_coverage"], "PASS")
        reassessed = phase_e_assess.execute(f.e, original, 2)
        self.assertEqual(read_json(reassessed)["summary"]["status"], "PASS")
        earlier = phase_e_assess.execute(f.e, again, 1)
        self.assertEqual(read_json(earlier)["summary"]["status"], "PASS")

    def test_pilot_resume_remains_usable_for_calibration(self):
        f = self.f
        resumed = phase_e_pilot.execute(f.e, f.metadata, f.acceptance, 101, f.pilots[101].parent / "resume-0001.json")
        completed = phase_e_calibrate.execute(f.e, f.metadata, [resumed, f.pilots[102]], f.pilots[103])
        self.assertEqual(read_json(completed)["summary"]["correctness"], "PASS")
        first = read_json(f.calibration.parent / "calibration.json")
        second = read_json(completed.parent / "calibration.json")
        self.assertEqual(first, second)

    def test_invalid_submission_penalty_and_failed_round_boundary(self):
        f = self.f
        from edgefl.attacks.trainer import AttackTrainer
        original = AttackTrainer.train
        def invalid_one(trainer, request):
            update = original(trainer, request)
            if request.round_id == 2 and request.client.client_id == sorted(trainer.originals)[0]:
                return replace(update, registered_training_count=999999)
            return update
        with patch.object(AttackTrainer, "train", invalid_one):
            run, _ = execute(f)
        self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
        transitions = read_json(run.parent / "reputation-0002.json")["transitions"]
        penalized = [t for t in transitions if t["reason"] == "invalid_submission_penalty"]
        self.assertEqual(len(penalized), 1)
        self.assertAlmostEqual(penalized[0]["current"], .9 * penalized[0]["previous"])
        from edgefl.trust.aggregation import FixedWeightAggregator
        aggregate = FixedWeightAggregator.aggregate
        def fail_second(aggregator, request):
            if request.round_id == 2:
                raise ValueError("Injected aggregation failure")
            return aggregate(aggregator, request)
        with patch.object(FixedWeightAggregator, "aggregate", fail_second):
            failed, _ = execute(f)
        result = read_json(failed)["summary"]
        self.assertEqual(result["correctness"], "FAIL")
        self.assertEqual(result["completed_steps"], 1)
        self.assertFalse((failed.parent / "resume-0002.json").exists())
        self.assertEqual(result["final_checkpoint"]["path"], read_json(failed.parent / "resume-0001.json")["checkpoint"]["path"])
        with self.assertRaisesRegex(ValueError, "override"):
            phase_e_review.execute(f.e, failed, "acceptable", "Cannot approve failure.")

    def test_changed_source_calibration_and_unresolved_review_rejected(self):
        f = self.f
        path = f.root / "archive/selected.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                load_stage(f.e, f.metadata)
        finally:
            path.write_bytes(original)
        calibration = f.calibration.parent / "calibration.json"
        original = calibration.read_bytes()
        try:
            calibration.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_stage(f.e, f.calibration)
        finally:
            calibration.write_bytes(original)
        unresolved = phase_e_review.execute(f.e, f.calibration, "unresolved", "Unresolved audit behavior.")
        plan = phase_e_attack_plan.execute(f.e, f.metadata, 11, "clean")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            phase_e_federated.execute(f.e, f.metadata, f.initialization, f.acceptance, plan, f.calibration, unresolved, "adaptive", 11)

    def test_role_and_configuration_boundaries_and_lazy_cli(self):
        from edgefl.trust.metadata import Metadata
        f = self.f
        meta = Metadata(f.root, load_stage(f.e, f.metadata))
        with self.assertRaises(ValueError):
            meta.rows("final_test")
        path = f.root / "configs/invalid_e.json"
        for changes in ({"output": "src"}, {"weight_floor": 0}, {"unexpected": True}):
            write_json(path, {**f.e.values, **changes})
            with self.assertRaises(ValueError):
                load(path, f.root)
        command = "import sys; sys.path.insert(0,'src'); import edgefl.cli; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True, capture_output=True)


class TaskAndDeviceTests(unittest.TestCase):
    def test_multiclass_pipeline(self):
        f = build_fixture(self.addCleanup, task="multiclass")
        run, _ = execute(f, "targeted_label", "adaptive")
        self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
        report = read_json(run.parent / "assessment-0002.json")["observation"]
        self.assertEqual(report["parent"]["expertise_kind"], "multiclass_f1")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_phase_e_exact_resume(self):
        f = build_fixture(self.addCleanup, device="cuda")
        original, plan = execute(f, "additive_noise", "adaptive")
        resumed, _ = execute(f, "additive_noise", "adaptive", original.parent / "resume-0001.json", plan)
        self.assertEqual(read_json(resumed)["summary"]["correctness"], "PASS")
        a, b = [checkpoints.load(f.root / read_json(p)["summary"]["final_checkpoint"]["path"]) for p in (original, resumed)]
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual({k: v for k, v in a["phase_state"].items() if k != "round_artifacts"},
                         {k: v for k, v in b["phase_state"].items() if k != "round_artifacts"})

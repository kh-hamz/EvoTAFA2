"""Source-verified synthetic F workflows; no real-data acceptance claims."""
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
import test_phase_e_pipeline as e_fixture
from edgefl.data.storage import read_json, write_json
from edgefl.learning import checkpoints
from edgefl.trust import storage as e_storage
from edgefl.pipelines.phaseE import phase_e_review, phase_e_validate
from edgefl.optimization.config import load
from edgefl.optimization.storage import load_stage, implementation_hash, regression_hash, artifact
from edgefl.optimization.diagnostics import environment
from edgefl.pipelines import (phase_f_prepare, phase_f_federated, phase_f_optimize,
                             phase_f_profile, phase_f_review, phase_f_validate)
from edgefl.contracts.phase_e import CONDITIONS

ROOT = e_fixture.ROOT

def setup(owner, task="binary", device="cpu"):
    f = e_fixture.build_fixture(owner, task=task, device=device)
    reviews, f.e_runs, f.plans = [], {}, {}
    for condition in CONDITIONS:
        for method in ("fedavg", "adaptive", "frozen"):
            run, plan = e_fixture.execute(f, condition, method, plan=f.plans.get(condition))
            f.plans[condition] = plan
            f.e_runs[(condition, method)] = run
            reviews.append(phase_e_review.execute(f.e, run, "acceptable", "Synthetic E prerequisite execution."))
    e_report = f.root / "fixture-e-tests.json"
    write_json(e_report, {"schema_version": "phase-e.tests.v1", "tests": 1, "successful": True,
        "failures": 0, "errors": 0, "skipped": 0, "implementation_sha256": e_storage.implementation_hash(),
        "regression_sha256": e_storage.regression_hash(), "evidence_kind": "synthetic_validator_fixture"})
    f.e_acceptance = phase_e_validate.execute(f.e, f.metadata, f.calibration, f.calibration_review, reviews, e_report, 11)
    settings = read_json(ROOT / "configs/phase_f.json")
    settings.update(output="out_f", population_size=6, offspring_generations=1)
    path = f.root / "configs/phase_f.json"
    write_json(path, settings)
    f.f = load(path, f.root)
    f.prepared_f = phase_f_prepare.execute(f.f, f.e_acceptance, f.initialization)
    f.clean = phase_f_federated.execute(f.f, f.prepared_f, f.plans["clean"])
    if read_json(f.clean)["summary"]["correctness"] != "PASS":
        raise AssertionError(read_json(f.clean)["summary"])
    return f

def payload(f, path):
    return checkpoints.load(f.root / read_json(path)["summary"]["final_checkpoint"]["path"])

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = setup(cls.addClassCleanup)

    def test_clean_replay_profile_and_scoped_acceptance(self):
        f = self.f
        replay = phase_f_optimize.execute(f.f, f.prepared_f, f.clean, 2)
        self.assertEqual(read_json(replay)["summary"]["status"], "PASS")
        original = read_json(f.clean.parent / "aggregation-0002.json")
        repeated = read_json(replay.parent / "aggregation-0002.json")
        self.assertEqual(original["weights"], repeated["weights"])
        self.assertEqual(original["search"]["generations"], repeated["search"]["generations"])
        e_replay = phase_f_optimize.execute(f.f, f.prepared_f, f.e_runs[("clean", "adaptive")], 2)
        self.assertEqual(read_json(e_replay)["summary"]["status"], "PASS")
        profile = phase_f_profile.execute(f.f, f.clean, 2, 2, 3)
        info = read_json(profile)["summary"]
        self.assertEqual(info["correctness"], "PASS", info)
        self.assertEqual(info["warmup_model_evaluations"], 1)
        self.assertEqual(info["counts"]["candidate_requests"], 12)
        self.assertGreater(info["counts"]["actual_model_evaluations"], 0)
        review = phase_f_review.execute(f.f, f.clean, "acceptable", "Synthetic clean execution.")
        reviewed_profile = phase_f_review.execute(f.f, profile, "acceptable", "Actual model inference fixture.")
        report = f.root / "fixture-f-tests.json"
        write_json(report, {"schema_version": "phase-f.tests.v1", "tests": 1, "successful": True,
            "failures": 0, "errors": 0, "skipped": 0, "implementation_sha256": implementation_hash(),
            "regression_sha256": regression_hash(), "search_environment": environment(),
            "evidence_kind": "synthetic_validator_fixture"})
        accepted = phase_f_validate.execute(f.f, f.prepared_f, review, reviewed_profile, report)
        self.assertEqual(read_json(accepted)["summary"]["status"], "PASS")
        self.assertEqual(load_stage(f.f, accepted)["summary"]["status"], "PASS")
        with patch("edgefl.optimization.review.regression_hash", return_value="0"*64):
            with self.assertRaisesRegex(ValueError, "software evidence"):
                load_stage(f.f, accepted)
        value = read_json(report)
        write_json(report, {**value, "regression_sha256": "0"*64})
        with self.assertRaisesRegex(ValueError, "software evidence"):
            phase_f_validate.execute(f.f, f.prepared_f, review, reviewed_profile, report)

    def test_sign_flip_exact_resume_and_completed_boundary(self):
        f = self.f
        original = phase_f_federated.execute(f.f, f.prepared_f, f.plans["sign_flip"])
        self.assertEqual(read_json(original)["summary"]["correctness"], "PASS")
        resumed = phase_f_federated.execute(f.f, f.prepared_f, f.plans["sign_flip"], original.parent / "resume-0001.json")
        again = phase_f_federated.execute(f.f, f.prepared_f, f.plans["sign_flip"], resumed.parent / "resume-0002.json")
        first = payload(f, original)
        for path in (resumed, again):
            other = payload(f, path)
            for key in first["model"]:
                torch.testing.assert_close(first["model"][key], other["model"][key], rtol=0, atol=0)
            self.assertEqual(first["phase_state"]["previous_selected"], other["phase_state"]["previous_selected"])
            self.assertEqual(first["phase_state"]["trust"]["reputation"], other["phase_state"]["trust"]["reputation"])
            self.assertEqual(set(other["phase_state"]["search_artifacts"]), {"1", "2"})
        a = read_json(original.parent / "aggregation-0002.json")["search"]
        b = read_json(resumed.parent / "aggregation-0002.json")["search"]
        self.assertEqual(a["generations"], b["generations"])
        self.assertEqual(read_json(again)["summary"]["attack_coverage"], "PASS")
        earlier = phase_f_optimize.execute(f.f, f.prepared_f, again, 1)
        self.assertEqual(read_json(earlier)["summary"]["status"], "PASS")

    def test_failed_search_retains_parent_and_reputation(self):
        from edgefl.optimization.aggregation import NSGA2Aggregator
        from edgefl.optimization.fitness import NumericalCandidateError
        f = self.f
        original = NSGA2Aggregator.evaluator
        def invalid_second(aggregate, request):
            evaluator = original(aggregate, request)
            if request.round_id == 2:
                def fail(state):
                    raise NumericalCandidateError("Injected nonfinite candidate")
                evaluator.predict = fail
            return evaluator
        with patch.object(NSGA2Aggregator, "evaluator", invalid_second):
            failed = phase_f_federated.execute(f.f, f.prepared_f, f.plans["clean"])
        summary = read_json(failed)["summary"]
        self.assertEqual(summary["correctness"], "FAIL")
        self.assertEqual(summary["completed_steps"], 1)
        self.assertIn("aggregation_diagnostics", summary["failure"])
        self.assertEqual(payload(f, failed)["phase_state"]["trust"]["reputation"],
                         checkpoints.load(f.clean.parent / "checkpoint-0001.pt")["phase_state"]["trust"]["reputation"])
        self.assertFalse((failed.parent / "resume-0002.json").exists())
        with self.assertRaisesRegex(ValueError, "override"):
            phase_f_review.execute(f.f, failed, "acceptable", "Failure cannot pass.")

    def test_partial_participation_preserves_mapping_and_priors(self):
        from edgefl.optimization.session import EvolutionSession
        from edgefl.learning.federated import BaselineSession
        f = self.f
        def partial(session, current):
            return sorted(session.counts) if current == 1 else sorted(session.counts)[1:4]
        with patch.object(EvolutionSession, "select", partial):
            run = phase_f_federated.execute(f.f, f.prepared_f, f.plans["clean"])
        self.assertEqual(read_json(run)["summary"]["correctness"], "PASS")
        weights = read_json(run.parent / "aggregation-0002.json")["weights"]
        self.assertEqual(len(weights), 3)
        self.assertTrue(all(v <= .5 for v in weights.values()))
        changes = read_json(run.parent / "reputation-0002.json")["transitions"]
        absent = [t for t in changes if t["reason"] == "ordinary_nonparticipation"]
        self.assertEqual(len(absent), 2)
        self.assertTrue(all(t["previous"] == t["current"] for t in absent))

    def test_changed_source_calibration_round_artifact_and_environment_rejected(self):
        f = self.f
        for path in (f.root / "archive/selected.csv", f.calibration.parent / "calibration.json",
                     f.clean.parent / "aggregation-0001.json"):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    load_stage(f.f, f.clean)
            finally:
                path.write_bytes(original)
        marker = read_json(f.clean.parent / "resume-0001.json")
        write_json(f.root / "bad-resume.json", {**marker, "implementation_sha256": "0"*64})
        with self.assertRaisesRegex(ValueError, "resume"):
            phase_f_federated.execute(f.f, f.prepared_f, f.plans["clean"], f.root / "bad-resume.json")

    def test_invalid_config_unresolved_review_and_lazy_cli(self):
        f = self.f
        for edit in ({"weight_cap": .6}, {"population_size": 5}, {"output": "out_e"}, {"unknown": 0}):
            path = f.root / "configs/invalid_f.json"
            write_json(path, {**f.f.values, **edit})
            with self.assertRaises(ValueError):
                load(path, f.root)
        numeric = f.root / "configs/integral_f.json"
        write_json(numeric, {**f.f.values, "population_size": 6.0, "offspring_generations": 1.0, "refill_multiplier": 100.0})
        normalized = load(numeric, f.root)
        self.assertEqual(normalized.sha256, f.f.sha256)
        self.assertTrue(all(type(normalized.values[k]) is int for k in
                            ("population_size", "offspring_generations", "refill_multiplier")))
        unresolved = phase_f_review.execute(f.f, f.clean, "unresolved", "Pending interpretation.")
        from edgefl.optimization.review import accepted_review
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            accepted_review(f.f, unresolved)
        code = "import sys; sys.path.insert(0,'src'); import edgefl.cli; assert not {'torch','numpy','pymoo'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_engine_boundary_resume_with_evolution_session(self):
        # Component integration of the actual shared engine; these outputs are not gate evidence.
        from types import SimpleNamespace
        from edgefl.config import Configuration, canonical_json
        from edgefl.contracts.phase_e import ArtifactPolicy
        from edgefl.learning.engine import run
        from edgefl.learning.data import LearningData
        from edgefl.learning import storage as d
        from edgefl.optimization.session import EvolutionSession
        from edgefl.trust.metadata import Metadata
        f = self.f
        values = f.config.values
        values["device"] = "cuda"
        config = Configuration(f.root, f.config.source, canonical_json(values))
        prepared = d.load_stage(f.config, f.prepared)
        initialization = d.load_stage(f.config, f.initialization)
        data = LearningData(f.config, prepared)
        metadata = Metadata(f.root, e_storage.load_stage(f.e, f.metadata))
        calibration = read_json(f.calibration.parent / "calibration.json")
        plan = read_json(f.plans["clean"].parent / "attack_plan.json")
        def session(*args):
            return EvolutionSession(*args, settings=f.e.values, metadata=metadata, plan=plan,
                calibration=calibration, bindings={"evidence_kind": "numerical_cuda_fixture"},
                search_settings=f.f.values)
        paths, results = [], []
        for i in range(2):
            directory = f.root / ("cuda-engine-" + str(i))
            directory.mkdir()
            attempt = SimpleNamespace(directory=directory, identity=prepared["identity"],
                                      gate=prepared["gate"], inputs={})
            resume = None if i == 0 else (checkpoints.load(paths[0] / "checkpoint-0001.pt"),
                                         read_json(paths[0] / "resume-0001.json"))
            result = run(config, data, prepared, initialization, attempt, "nsga2", 11, resume,
                         session_factory=session, artifact_policy=ArtifactPolicy("phase-f", "numerical-fixture", "numerical-fixture"))
            self.assertEqual(result["correctness"], "PASS", result)
            paths.append(directory)
            results.append(checkpoints.load(f.root / result["final_checkpoint"]["path"]))
        for name in results[0]["model"]:
            torch.testing.assert_close(results[0]["model"][name], results[1]["model"][name], rtol=0, atol=0)
        self.assertEqual(results[0]["phase_state"]["previous_selected"], results[1]["phase_state"]["previous_selected"])
        self.assertEqual(results[0]["phase_state"]["trust"]["reputation"], results[1]["phase_state"]["trust"]["reputation"])

class TaskDeviceTests(unittest.TestCase):
    def test_multiclass_actual_search(self):
        from phase_f_fixtures import numerical
        from edgefl.optimization.pymoo_adapter import search
        f = numerical(self.addCleanup, classes=3)
        result = search(f.evaluator, f.settings, f.counts)
        self.assertTrue(result.selected.feasible)
        metrics = f.evaluator.reports[result.selected.evaluation_id]
        self.assertEqual(len(metrics["class_f1"]), 3)
        self.assertEqual(metrics["support"], [3,1,0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_search_replay(self):
        from phase_f_fixtures import numerical
        from edgefl.optimization.pymoo_adapter import search
        from edgefl.learning.models import configure
        configure({"device": "cuda", "threads": 1})
        f = numerical(self.addCleanup, device="cuda")
        first = search(f.evaluator, f.settings, f.counts)
        f.evaluator.reset()
        second = search(f.evaluator, f.settings, f.counts)
        self.assertEqual(first.selected, second.selected)
        self.assertEqual(first.generations, second.generations)

import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch
import phase_g_fixtures as fixture
from edgefl.data.storage import read_json, write_json
from edgefl.experiments.storage import load_stage, implementation_hash, regression_hash
from edgefl.experiments.sessions import execute
from edgefl.experiments.review import review_batch, review_subject
from edgefl.experiments.runner import preflight, execute_queue, verify_profile_runtime
from edgefl.experiments.reporting import summarize
from edgefl.experiments.handoff import validate, export
from edgefl.experiments.profiling import profile
from edgefl.learning import checkpoints


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = fixture.setup(cls.addClassCleanup)
        fixture.execute_all(cls.f)

    def test_actual_policy_execution_and_exact_resume(self):
        f = self.f
        for name, path in f.g_runs.items():
            with self.subTest(name=name):
                self.assertEqual(read_json(path)["summary"]["correctness"], "PASS")
                self.assertFalse(read_json(path)["summary"]["final_test_evaluated"])
        record = read_json(f.g_runs["no_reputation"].parent / "assessment-0002.json")
        self.assertTrue(all(v == 1 for v in record["prior_reputation"].values()))
        changes = read_json(f.g_runs["no_reputation"].parent / "reputation-0002.json")
        self.assertEqual(changes["transitions"], [])
        clipped = read_json(f.g_runs["clipping"].parent / "assessment-0002.submissions.json")["clipping"]
        self.assertTrue(clipped)
        for entry in clipped.values():
            self.assertLessEqual(entry["applied_norm"], entry["threshold"] * (1 + 1e-6))
            self.assertAlmostEqual(entry["factor"], min(1., entry["threshold"] / entry["submitted_norm"]))
        original = f.g_runs["poison"]
        resumed = execute(f.g, f.g_prepared["poison"], original.parent / "resume-0001.json")
        a = checkpoints.load(f.root / read_json(original)["summary"]["final_checkpoint"]["path"])
        b = checkpoints.load(f.root / read_json(resumed)["summary"]["final_checkpoint"]["path"])
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual(a["phase_state"]["previous_selected"], b["phase_state"]["previous_selected"])
        self.assertEqual(a["phase_state"]["trust"]["reputation"], b["phase_state"]["trust"]["reputation"])

    def test_profiles_batch_reviews_coverage_and_handoff(self):
        f = self.f
        bindings, runs, reviews = f.root / "g-bindings.json", f.root / "g-runs.json", f.root / "g-reviews.json"
        write_json(bindings, {f.g_cases[k].case_id: v.relative_to(f.root).as_posix() for k, v in f.g_prepared.items()})
        write_json(runs, {f.g_cases[k].case_id: v.relative_to(f.root).as_posix() for k, v in f.g_runs.items()})
        write_json(reviews, [v.relative_to(f.root).as_posix() for v in f.g_reviews.values()])
        _, readiness = preflight(f.g, f.g_matrix, bindings)
        self.assertTrue(all(v["status"] == "ready" for v in readiness.values()))
        measurements = f.root / "g-measurements.json"
        write_json(measurements, [{"run": f.g_runs[k].relative_to(f.root).as_posix(), "round": 2} for k in ("clean", "random")])
        cost = profile(f.g, f.g_matrix, bindings, measurements)
        self.assertEqual(read_json(cost)["summary"]["correctness"], "PASS")
        self.assertEqual(len(read_json(cost)["summary"]["covered_cases"]), 2)
        verify_profile_runtime(f.g, read_json(cost))
        with patch("edgefl.optimization.diagnostics.environment", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "Runtime changed|environment mismatch"):
                verify_profile_runtime(f.g, read_json(cost))
        manifest = f.root / "g-review-manifest.json"
        write_json(manifest, [{"subject": cost.relative_to(f.root).as_posix(), "disposition": "acceptable", "rationale": "Fixture cost path only."}])
        self.assertEqual(read_json(review_batch(f.g, manifest))["summary"]["status"], "REVIEWED")
        summary = summarize(f.g, f.g_matrix, runs)
        report = f.root / "fixture-g-tests.json"
        write_json(report, {"schema_version": "phase-g.tests.v1", "tests": 1, "successful": True, "failures": 0,
            "errors": 0, "skipped": 0, "implementation_sha256": implementation_hash(), "regression_sha256": regression_hash(),
            "evidence_kind": "synthetic_validator_fixture"})
        accepted = validate(f.g, f.g_matrix, summary, reviews, report)
        self.assertEqual(read_json(accepted)["summary"]["status"], "PASS_SCOPED")
        self.assertFalse(read_json(accepted)["summary"]["research_completion"])
        handoff = export(f.g, accepted)
        self.assertEqual(read_json(handoff)["summary"]["eligible"], len(f.g_cases))
        self.assertFalse(read_json(handoff)["summary"]["phase_h_freeze_performed"])

    def test_changed_source_policy_and_incomplete_readiness_rejected(self):
        f = self.f
        blocked = fixture.bind(f, "clipping")
        with self.assertRaisesRegex(ValueError, "blocked"):
            execute(f.g, blocked)
        path = f.g_prepared["clean"].parent / "policies.json"
        original = path.read_bytes()
        try:
            write_json(path, {"invalid": True})
            with self.assertRaises(ValueError): load_stage(f.g, f.g_prepared["clean"])
        finally:
            path.write_bytes(original)
        source = f.root / "archive/selected.csv"
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                load_stage(f.g, f.g_prepared["clean"])
        finally:
            source.write_bytes(original)
        from edgefl.experiments.preparation import require_clean_inputs
        matching = dict(read_json(f.g_prepared["poison"])["summary"]["matching"], initialization="changed")
        with self.assertRaisesRegex(ValueError, "unmatched"):
            require_clean_inputs(f.g, read_json(f.g_runs["clean"]), matching)
        review = review_subject(f.g, f.g_runs["clean"], "unresolved", "Investigation remains incomplete.")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            fixture.bind(f, "poison", clean_review=review)

    def test_lazy_cli_and_real_subprocess_queue(self):
        f = self.f
        code = "import sys;sys.path.insert(0,'src');from edgefl.cli import main;main(['pipelines']);assert not any(x in sys.modules for x in ('torch','numpy','pymoo'))"
        lazy = subprocess.run([sys.executable, "-c", code], cwd=fixture.ROOT, capture_output=True, text=True)
        self.assertEqual(lazy.returncode, 0, lazy.stderr)
        # Only the already-reviewed clean prepared case is supplied; other cases stay blocked.
        bindings = f.root / "queue-bindings.json"
        write_json(bindings, {f.g_cases["clean"].case_id: f.g_prepared["clean"].relative_to(f.root).as_posix()})
        result = execute_queue(f.g, f.g_matrix, bindings, suite="fixture")
        self.assertEqual(result["cases"][f.g_cases["clean"].case_id]["status"], "complete")
        self.assertEqual(result["status"], "INCOMPLETE")
        again = execute_queue(f.g, f.g_matrix, bindings, suite="fixture")
        self.assertEqual(again["cases"][f.g_cases["clean"].case_id]["completion"], result["cases"][f.g_cases["clean"].case_id]["completion"])

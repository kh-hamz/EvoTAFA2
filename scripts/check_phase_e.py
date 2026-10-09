"""Run all A-E regressions and publish current implementation-bound software evidence."""

import json
import sys
import unittest
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

if __name__ == "__main__":
    modules = ("test_phase_a", "test_phase_b", "test_phase_b_panels", "test_phase_c", "test_phase_c_pipeline",
               "test_stabilization", "test_phase_d", "test_phase_d_pipeline", "test_phase_e_attacks", "test_phase_e_scoring",
               "test_phase_e_risk", "test_phase_e_reputation", "test_phase_e_aggregation", "test_phase_e_pipeline")
    results = []
    for selected in (modules[:8], modules[8:]):
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in selected)
        results.append(unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite))
    from edgefl.trust.storage import implementation_hash, regression_hash
    from edgefl.learning.storage import implementation_hash as d_hash
    output = ROOT / "reports/generated/phase_e_tests.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    def publish(path, report):
        if path.exists():
            prior = path.read_bytes()
            history = path.parent / "history" / (path.stem + "-" + hashlib.sha256(prior).hexdigest()[:16] + ".json")
            history.parent.mkdir(parents=True, exist_ok=True)
            if not history.exists():
                history.write_bytes(prior)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    base = results[0]
    publish(output.with_name("phase_d_tests.json"), {"schema_version": "phase-d.tests.v1", "tests": base.testsRun,
        "failures": len(base.failures), "errors": len(base.errors), "skipped": len(base.skipped),
        "successful": base.wasSuccessful() and not base.skipped, "implementation_sha256": d_hash()})
    passed = all(r.wasSuccessful() and not r.skipped for r in results)
    publish(output, {"schema_version": "phase-e.tests.v1", "tests": sum(r.testsRun for r in results),
        "failures": sum(len(r.failures) for r in results), "errors": sum(len(r.errors) for r in results),
        "skipped": sum(len(r.skipped) for r in results), "successful": passed, "modules": list(modules),
        "implementation_sha256": implementation_hash(), "regression_sha256": regression_hash()})
    raise SystemExit(not passed)

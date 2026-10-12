"""Run all implemented phase regressions; publish machine-readable test evidence."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

if __name__ == "__main__":
    suite = unittest.TestSuite()
    for module in ("test_phase_a", "test_phase_b", "test_phase_b_panels", "test_phase_c",
                   "test_phase_c_pipeline", "test_stabilization", "test_phase_d", "test_phase_d_pipeline"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    from edgefl.learning.storage import implementation_hash
    output = ROOT / "reports/generated/phase_d_tests.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "phase-d.tests.v1", "tests": result.testsRun, "failures": len(result.failures),
                                  "errors": len(result.errors), "skipped": len(result.skipped),
                                  "successful": result.wasSuccessful(),
                                  "implementation_sha256": implementation_hash()}, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())

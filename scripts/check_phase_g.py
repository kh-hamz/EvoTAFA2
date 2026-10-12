"""Run full A-G regressions and retain historical acceptance evidence."""
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "scripts")]
from check_phase_f import publish

GROUPS = [
    ["test_phase_a", "test_phase_b", "test_phase_b_panels", "test_phase_c", "test_phase_c_pipeline", "test_stabilization", "test_phase_d", "test_phase_d_pipeline"],
    ["test_phase_e_attacks", "test_phase_e_scoring", "test_phase_e_risk", "test_phase_e_reputation", "test_phase_e_aggregation", "test_phase_e_pipeline"],
    ["test_phase_f_chromosomes", "test_phase_f_initialization", "test_phase_f_fitness", "test_phase_f_pymoo", "test_phase_f_selection", "test_phase_f_pipeline"],
    ["test_phase_g_matrix", "test_phase_g_dependencies", "test_phase_g_policies", "test_phase_g_panels", "test_phase_g_runner", "test_phase_g_reporting", "test_phase_g_pipeline"]]


if __name__ == "__main__":
    results, durations = [], []
    for names in GROUPS:
        started = time.perf_counter()
        results.append(unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(
            unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)))
        durations.append(time.perf_counter() - started)
    from edgefl.learning.storage import implementation_hash as d_hash
    from edgefl.trust.storage import implementation_hash as e_hash, regression_hash as e_regression
    from edgefl.optimization.storage import implementation_hash as f_hash, regression_hash as f_regression
    from edgefl.experiments.storage import implementation_hash as g_hash, regression_hash as g_regression
    from edgefl.optimization.diagnostics import environment
    for i, phase in enumerate(("d", "e", "f", "g")):
        selected = results[:i+1]
        report = {"schema_version": f"phase-{phase}.tests.v1", "tests": sum(r.testsRun for r in selected),
            "failures": sum(len(r.failures) for r in selected), "errors": sum(len(r.errors) for r in selected),
            "skipped": sum(len(r.skipped) for r in selected), "successful": all(r.wasSuccessful() and not r.skipped for r in selected),
            "implementation_sha256": (d_hash, e_hash, f_hash, g_hash)[i](), "modules": sum(GROUPS[:i+1], []), "suite_seconds": durations[:i+1]}
        if i:
            report["regression_sha256"] = (e_regression, f_regression, g_regression)[i-1]()
        if i >= 2:
            report["search_environment"] = environment()
        publish(ROOT / f"reports/generated/phase_{phase}_tests.json", report)
    raise SystemExit(not all(r.wasSuccessful() and not r.skipped for r in results))

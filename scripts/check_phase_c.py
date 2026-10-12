"""Run Phase A, Phase B, and Phase C behavior tests without discovery ambiguity."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for module in ("test_phase_a", "test_phase_b", "test_phase_b_panels", "test_phase_c",
                   "test_phase_c_pipeline", "test_stabilization"):
        suite.addTests(loader.loadTestsFromName(module))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite).wasSuccessful())

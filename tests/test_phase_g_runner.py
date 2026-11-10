import json
import tempfile
import unittest
from pathlib import Path
from edgefl.experiments.ledger import Ledger, scheduler_lock


class RunnerTests(unittest.TestCase):
    def test_transactional_states_and_wrong_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.sqlite"
            with Ledger(path, "matrix") as ledger:
                ledger.register(["one", "two"])
                ledger.transition("one", "running", command="registered")
                ledger.transition("one", "failed", reason={"step": 2})
                self.assertEqual(ledger.rows()["two"]["status"], "planned")
                self.assertIn("step", ledger.rows()["one"]["reason"])
                with self.assertRaises(ValueError): ledger.transition("absent", "complete")
            with self.assertRaises(ValueError): Ledger(path, "other")

    def test_exclusive_lock_and_explicit_stale_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler.lock"
            with scheduler_lock(path):
                with self.assertRaisesRegex(ValueError, "still active"):
                    with scheduler_lock(path, recover=True): pass
            self.assertFalse(path.exists())
            path.write_text(json.dumps({"pid": 2147483647, "created": 0, "child": None}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit recovery"):
                with scheduler_lock(path): pass
            with scheduler_lock(path, recover=True):
                self.assertTrue(path.exists())

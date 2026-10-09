"""Temporal reputation ordering, dispositions and snapshot isolation."""

import unittest
from edgefl.trust.reputation import ReputationStore


class ReputationTests(unittest.TestCase):
    def test_valid_invalid_and_absent_are_distinct_and_provisional(self):
        store = ReputationStore(("good", "invalid", "absent"))
        pending, transitions = store.propose(1, {"good": .2}, {"invalid": "bad shape"}, ["good", "invalid"])
        self.assertEqual(store.prior(), dict.fromkeys(("good", "invalid", "absent"), .5))
        self.assertAlmostEqual(pending["good"]["value"], .53)
        self.assertAlmostEqual(pending["invalid"]["value"], .45)
        self.assertEqual(pending["absent"]["value"], .5)
        store.load_state_dict(pending)
        self.assertEqual(store.entries["invalid"]["last_valid"], 0)
        self.assertEqual(store.entries["good"]["last_valid"], 1)
        pending["good"]["value"] = 0
        self.assertAlmostEqual(store.prior()["good"], .53)

    def test_conflicting_dispositions_and_tampered_snapshot_fail(self):
        store = ReputationStore(("a",))
        with self.assertRaises(ValueError):
            store.propose(1, {"a": .1}, {"a": "bad"}, ["a"])
        with self.assertRaises(ValueError):
            store.propose(1, {}, {}, ["a"])
        state = store.state_dict()
        state["a"]["value"] = float("nan")
        with self.assertRaises(ValueError):
            store.load_state_dict(state)
        pending, _ = store.propose(2, {"a": 0.}, {}, ["a"])
        store.load_state_dict(pending)
        with self.assertRaisesRegex(ValueError, "future"):
            store.propose(2, {"a": 1.}, {}, ["a"])

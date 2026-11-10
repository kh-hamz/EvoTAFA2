import unittest
from edgefl.experiments.panels import select_panel, IndexedRows


class PanelTests(unittest.TestCase):
    def test_nested_stratified_subsets_and_fixed_missing_class(self):
        rows = [{"row": i, "observation_id": f"oid-{i}", "original": label, "target": 0 if label == "Normal" else 1}
                for i, label in enumerate(["Normal"]*16 + ["AttackA"]*8 + ["AttackB"]*4)]
        profiles = {"c": {"support": {"Normal": 20, "AttackA": 20, "AttackB": 20}}}
        quarter, half, full = [select_panel(rows, profiles, fraction) for fraction in (.25, .5, 1.)]
        sets = [set(r["observation_id"] for r in p["rows"]) for p in (quarter, half, full)]
        self.assertTrue(sets[0] < sets[1] < sets[2])
        self.assertEqual([len(s) for s in sets], [7, 14, 28])
        absent = select_panel(rows, profiles, missing_class=True)
        self.assertEqual(absent["removed_class"], "AttackB")
        self.assertNotIn("AttackB", absent["actual_support"])
        self.assertEqual(select_panel(list(reversed(rows)), profiles), full)

    def test_support_failure_and_role_view_indices(self):
        rows = [{"row": 0, "observation_id": "a", "original": "Normal"}, {"row": 1, "observation_id": "b", "original": "Attack"}]
        with self.assertRaisesRegex(ValueError, "UNSUCCESSFUL"):
            select_panel(rows, {"c": {"support": {"Attack": 1}}}, .25)
        with self.assertRaisesRegex(ValueError, "UNSUCCESSFUL"):
            select_panel(rows, {"c": {"support": {"Attack": 1}}}, missing_class=True)
        self.assertEqual(list(IndexedRows([10, 20, 30], [2, 0])), [30, 10])
        with self.assertRaises(ValueError): IndexedRows([1, 2], [0, 0])

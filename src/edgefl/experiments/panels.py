"""Immutable trusted subviews; source membership and label vocabulary are retained."""
from collections import Counter
import hashlib
import math
from edgefl.contracts.phase_g import digest


def select_panel(rows, profiles, fraction=1., missing_class=False):
    if fraction not in (.25, .5, 1.) or not rows:
        raise ValueError("Invalid/empty trusted panel")
    if len({r["observation_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trusted observation")
    support = Counter(r["original"] for r in rows)
    removed = None
    if missing_class:
        training = Counter()
        for profile in profiles.values():
            training.update(profile["support"])
        eligible = [label for label in support if label != "Normal" and training[label] > 0]
        if not eligible:
            raise ValueError("UNSUCCESSFUL: no eligible missing trusted attack class")
        removed = min(eligible, key=lambda label: (support[label], label))
    chosen = []
    for label in sorted(support):
        if label == removed:
            continue
        group = sorted((r for r in rows if r["original"] == label),
                       key=lambda r: (hashlib.sha256(r["observation_id"].encode()).hexdigest(), r["observation_id"]))
        chosen.extend(group[:math.floor(fraction * len(group))])
    chosen.sort(key=lambda r: r["row"])
    actual = Counter(r["original"] for r in chosen)
    if not actual["Normal"] or not any(n for label, n in actual.items() if label != "Normal"):
        raise ValueError("UNSUCCESSFUL: derived trusted panel lacks benign/attack support")
    value = {"schema_version": "phase-g.panel.v1", "fraction": fraction, "removed_class": removed,
             "rows": chosen, "reference_support": dict(support), "actual_support": dict(actual)}
    value["panel_id"] = digest(value)
    return value


class IndexedRows:
    def __init__(self, source, indices):
        self.source, self.indices = source, tuple(indices)
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(source) for i in indices):
            raise ValueError("Invalid trusted subset indices")

    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.source[self.indices[index]]


class ScopedLearningData:
    def __init__(self, source, panel):
        self.source, self.spec, self.clients = source, source.spec, source.clients
        self.panel = IndexedRows(source.view("trusted"), [r["row"] for r in panel["rows"]])

    def view(self, role, client=None):
        if role == "trusted":
            if client is not None:
                raise ValueError("Trusted data is not a client resource")
            return self.panel
        return self.source.view(role, client)


class ScopedMetadata:
    def __init__(self, source, panel):
        self.source, self.spec = source, source.spec
        self.panel = panel

    def rows(self, role, client=""):
        if role == "trusted":
            if client:
                raise ValueError("Invalid trusted client selector")
            return self.panel["rows"]
        return self.source.rows(role, client)

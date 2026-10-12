"""Simulator-only deterministic plans. No comparison-method identity enters randomness."""

import hashlib
import math
from collections import Counter
from edgefl.config import canonical_json
from edgefl.contracts.phase_e import CONDITIONS
from edgefl.reproducibility import derive_seed


def attack_seed(seed, purpose, client=None, round_id=None):
    base = derive_seed(seed, "poisoning", client=client, round_id=round_id)
    return int.from_bytes(hashlib.sha256(canonical_json(["phase-e.attack-seed.v1", base, purpose]).encode()).digest()[:4], "big")


def priority(seed, purpose, identity, client=None):
    return hashlib.sha256(canonical_json([attack_seed(seed, purpose, client), identity]).encode()).digest()


def build_plan(metadata, values, seed, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown attack condition")
    clients = sorted(metadata.spec["profiles"])
    count = 0 if condition == "clean" else math.floor(values["compromised_fraction"] * len(clients))
    if condition != "clean" and count == 0:
        raise ValueError("Attack fraction selects no compromised clients")
    compromised = sorted(sorted(clients, key=lambda c: priority(seed, "compromised", c))[:count])
    support, eligible = Counter(), set()
    for profile in metadata.spec["profiles"].values():
        support.update(profile["support"])
        eligible.update(label for label, n in profile["support"].items() if label != "Normal" and math.floor(n * values["label_fraction"]) > 0)
    target = min(eligible, key=lambda label: (-support[label], label)) if eligible else None
    if condition == "targeted_label" and target is None:
        raise ValueError("No eligible target attack class")
    changes, coverage = {}, {}
    vocabulary = sorted(metadata.spec["labels"].values())
    for client in compromised:
        candidates = metadata.rows("local_train", client)
        if condition == "targeted_label":
            candidates = [r for r in candidates if r["original"] == target]
        chosen = []
        if condition in ("untargeted_label", "targeted_label"):
            n = math.floor(len(candidates) * values["label_fraction"])
            chosen = sorted(candidates, key=lambda r: priority(seed, "poisoned_rows", r["observation_id"], client))[:n]
        entries = []
        for row in sorted(chosen, key=lambda r: r["row"]):
            if condition == "targeted_label":
                label = metadata.spec["labels"]["Normal"]
            else:
                alternatives = [v for v in vocabulary if v != row["target"]]
                if not alternatives:
                    raise ValueError("Label flipping requires at least two task labels")
                label = alternatives[int.from_bytes(priority(seed, "replacement", row["observation_id"], client)[:4], "big") % len(alternatives)]
            entries.append({"row": row["row"], "observation_id": row["observation_id"], "original_target": row["target"], "replacement": label})
        changes[client] = entries
        coverage[client] = {"eligible_rows": len(candidates), "changed_rows": len(entries), "lacks_target": condition == "targeted_label" and not entries}
    successful = condition not in ("targeted_label", "untargeted_label") or any(changes.values())
    return {"schema_version": "phase-e.attack-plan.v1", "condition": condition, "seed": seed,
            "clients": clients, "compromised": compromised, "target": target if condition == "targeted_label" else None,
            "activation_round": values["activation_round"], "period": values["intermittent_period"],
            "scale": values["update_scale"], "changes": changes, "coverage": coverage,
            "coverage_status": "PASS" if successful else "UNSUCCESSFUL"}


def active(plan, client, current):
    if client not in plan["compromised"] or current < plan["activation_round"]:
        return False
    return plan["condition"] != "intermittent_sign" or ((current - plan["activation_round"]) // plan["period"]) % 2 == 0

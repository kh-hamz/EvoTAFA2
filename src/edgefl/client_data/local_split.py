"""Client-local, group-preserving validation reservation before preprocessing."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from edgefl.contracts.phase_c import MANIFEST_FIELDS, SupportPolicy
from edgefl.client_data.policies import class_counts, metric_support
from edgefl.reproducibility import derive_seed
from edgefl.client_data.storage import rows, sha256, write_json, writer


def _priority(seed: int, client: str, group: str) -> str:
    return hashlib.sha256(f"{seed}:{client}:{group}".encode()).hexdigest()


def _training_valid(counts: Counter, size: int, minimum: int,
                    require_binary: bool) -> bool:
    if size < minimum:
        return False
    if not require_binary:
        return True
    return bool(counts.get("Normal")) and bool(sum(
        count for label, count in counts.items() if label != "Normal"))


def split_local(assignments: Path, directory: Path, seed: int, fraction: float,
                minimum_training: int, require_binary: bool, *, policy=None, attempts=50) -> dict:
    policy = policy or SupportPolicy(training=(int(require_binary), int(require_binary)))
    group_labels: dict[tuple[str, str], Counter] = defaultdict(Counter)
    group_sizes = Counter()
    clients = set()
    fold = scenario = None
    for row in rows(assignments):
        clients.add(row["client_id"])
        fold = fold or row["fold"]
        scenario = scenario or row["scenario"]
        if row["fold"] != fold or row["scenario"] != scenario:
            raise ValueError("Assignment manifest mixes folds or scenarios")
        key = row["client_id"], row["group_id"]
        group_labels[key][row["label"]] += 1
        group_sizes[key] += 1
    if not clients:
        raise ValueError("Assignment manifest is empty")
    partition_seed = derive_seed(seed, "partition", client=f"{fold}:{scenario}:local")
    roles, reports = {}, {}
    for client in sorted(clients):
        keys = sorted((key for key in group_labels if key[0] == client),
                      key=lambda key: (_priority(partition_seed, *key), key[1]))
        if len(keys) < 2:
            raise ValueError(f"{client} cannot reserve validation with fewer than two groups")
        total = sum(group_sizes[key] for key in keys)
        totals = sum((group_labels[key] for key in keys), Counter())
        candidates = []
        def evaluate(selected):
            size = sum(group_sizes[key] for key in selected)
            counts = sum((group_labels[key] for key in selected), Counter())
            deficits = policy.violations(counts, "validation") + policy.violations(totals - counts, "training")
            penalty = sum(item["minimum"] - item["actual"] for item in deficits)
            penalty += max(0, minimum_training - (total - size)) + int(size == 0) + int(size == total)
            return penalty, abs(size - total * fraction), tuple(sorted(selected))
        for attempt in range(attempts):
            ordered = sorted(keys, key=lambda key: (_priority(partition_seed + attempt, *key), key))
            selected = set()
            for key in ordered:
                candidate = selected | {key}
                if evaluate(candidate)[:2] < evaluate(selected)[:2]:
                    selected = candidate
            # Bounded moves and swaps also explore non-prefix validation subsets.
            for _ in range(min(32, len(keys) * 2)):
                rank = evaluate(selected)
                best = (rank, selected)
                sample = ordered[:128]
                for key in sample:
                    candidate = selected ^ {key}
                    candidate_rank = evaluate(candidate)
                    if candidate_rank < best[0]:
                        best = candidate_rank, candidate
                    if key in selected:
                        for other in sample:
                            if other not in selected:
                                candidate = (selected - {key}) | {other}
                                candidate_rank = evaluate(candidate)
                                if candidate_rank < best[0]:
                                    best = candidate_rank, candidate
                if best[0] >= rank:
                    break
                selected = best[1]
            rank = evaluate(selected)
            if rank[0] == 0:
                candidates.append((rank, selected))
                if rank[1] == 0:
                    break
        if not candidates:
            write_json(directory / "local_split_failure.json", {"client": client, "outcome": "search_exhausted",
                       "does_not_prove_infeasibility": True, "eligible_for_training": False})
            raise ValueError(f"{client}: local split search exhausted without a valid group-level split")
        _, validation = min(candidates, key=lambda item: item[0])
        validation_size = sum(group_sizes[key] for key in validation)
        training_size = total - validation_size
        for key in keys:
            roles[key] = "local_validation" if key in validation else "local_train"
        training_labels = totals - sum((group_labels[key] for key in validation), Counter())
        validation_labels = sum((group_labels[key] for key in validation), Counter())
        reports[client] = {
            "observations": total, "groups": len(keys),
            "local_train": training_size, "local_validation": validation_size,
            "achieved_validation_fraction": validation_size / total,
            "assigned_class_counts": class_counts(totals),
            "training_class_counts": class_counts(training_labels),
            "validation_class_counts": class_counts(validation_labels),
            "class_group_counts": {role: {label: sum(bool(group_labels[key][label]) for key in keys
                                                     if (key in validation) == (role == "validation"))
                                          for label in class_counts(totals)} for role in ("training", "validation")},
            "assigned_class_group_counts": {label: sum(bool(group_labels[key][label]) for key in keys)
                                             for label in class_counts(totals)},
            "metric_support": {"training": metric_support(training_labels), "validation": metric_support(validation_labels)},
            "training_binary_support": {"benign": training_labels["Normal"],
                "attack": sum(n for label, n in training_labels.items() if label != "Normal")},
        }
    stream, output = writer(directory / "local_splits.csv", MANIFEST_FIELDS["local_splits"])
    counts = Counter()
    with stream:
        for row in rows(assignments):
            role = roles[(row["client_id"], row["group_id"])]
            output.writerow({**row, "partition": role})
            counts[role] += 1
    manifest_hash = sha256(directory / "local_splits.csv")
    client_manifests = {
        client: {"client_id": client, "fold": fold, "scenario": scenario,
                 "partition_seed": partition_seed, "local_split_sha256": manifest_hash,
                 "training_role": "local_train", "validation_role": "local_validation",
                 "support": reports[client]}
        for client in sorted(clients)
    }
    write_json(directory / "client_manifests.json", client_manifests)
    fields = ("client_id", "label", "assigned", "assigned_groups", "training", "training_groups",
              "validation", "validation_groups")
    stream, table = writer(directory / "class_support.csv", fields)
    with stream:
        for client, report in sorted(reports.items()):
            for label in report["assigned_class_counts"]:
                table.writerow({"client_id": client, "label": label,
                    "assigned": report["assigned_class_counts"][label],
                    "assigned_groups": report["assigned_class_group_counts"][label],
                    "training": report["training_class_counts"][label],
                    "training_groups": report["class_group_counts"]["training"][label],
                    "validation": report["validation_class_counts"][label],
                    "validation_groups": report["class_group_counts"]["validation"][label]})
    return {"fold": fold, "scenario": scenario, "partition_seed": partition_seed,
            "target_validation_fraction": fraction, "counts": dict(counts),
            "clients": reports, "group_count": len(roles),
            "local_validation_is_attackable": False,
            "registered_training_count": counts["local_train"],
            "eligible_for_training": False}

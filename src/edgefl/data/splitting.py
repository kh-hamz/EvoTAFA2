"""Deterministic chronological allocation and independent whole-capture folds."""

import bisect
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.reproducibility import derive_seed
from edgefl.data.storage import database, rows, writer

ROLES = ("client_pool", "trusted", "selection_validation", "final_test")


def allocate(groups: list[dict], proportions: tuple[float, ...], purge: int):
    """Boundaries depend on ordered group counts, never labels or redraws."""
    ordered = sorted(groups, key=lambda g: (g["start"], g["end"], g["group_id"]))
    if not ordered:
        return {}, []
    total = sum(g["count"] for g in ordered)
    cumulative = [0]
    for group in ordered:
        cumulative.append(cumulative[-1] + group["count"])
    boundaries = []
    fraction = 0.0
    for proportion in proportions[:-1]:
        fraction += proportion
        index = min(range(len(ordered) + 1), key=lambda i: (abs(cumulative[i] - total * fraction), i))
        if index == 0:
            boundary = ordered[0]["start"] - purge - 1
        elif index == len(ordered):
            boundary = max(g["end"] for g in ordered) + purge + 1
        else:
            boundary = ordered[index]["start"]
        boundaries.append(boundary)
    result = {}
    for group in ordered:
        start, end = group["start"], group["end"]
        crosses = any(start < b <= end for b in boundaries)
        near = purge > 0 and any(start <= b + purge and end >= b - purge for b in boundaries)
        reason = "boundary_crossing_session_or_group" if crosses else "purge_interval" if near else ""
        role = "excluded" if reason else ROLES[bisect.bisect_right(boundaries, start)]
        result[group["group_id"]] = (role, reason)
    return result, boundaries


def split(groups_path: Path, directory: Path, protocol: str, seed: int, purge: int,
          expected_captures: list[str]) -> dict:
    db = database(directory / "_work/splits.sqlite")
    db.execute("CREATE TABLE observations(oid TEXT PRIMARY KEY,label TEXT,capture TEXT,session TEXT,gid TEXT,start REAL,end REAL)")
    try:
        all_labels = set()
        for row in rows(groups_path):
            if row["label"]:
                all_labels.add(row["label"])
            if row["status"] == "verified":
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                           (row["observation_id"],row["label"],row["capture_id"],row["session_id"],row["group_id"],float(row["start"]),float(row["end"])))
        db.commit()
        by_capture = defaultdict(list)
        for capture,gid,start,end,count in db.execute("SELECT capture,gid,min(start),max(end),count(*) FROM observations GROUP BY capture,gid ORDER BY capture,gid"):
            by_capture[capture].append({"group_id":gid,"start":start,"end":end,"count":count})
        labels_by_capture = defaultdict(set)
        for capture,label in db.execute("SELECT DISTINCT capture,label FROM observations"):
            labels_by_capture[capture].add(label)
        benign = sorted(c for c, labels in labels_by_capture.items() if labels == {"Normal"})
        attacks = sorted(c for c, labels in labels_by_capture.items() if "Normal" not in labels)
        split_seed = derive_seed(seed, "global_split")
        folds = []
        if protocol == "A":
            folds.append(("protocol_a", (), "session_time_closed_set"))
        else:
            for capture in sorted(by_capture):
                if capture in benign:
                    folds.append(("benign_" + capture, (capture,), "benign_source_generalization"))
                elif capture in attacks and benign:
                    index = int(hashlib.sha256(f"{split_seed}:{capture}".encode()).hexdigest(),16) % len(benign)
                    folds.append(("unseen_" + capture, (capture,benign[index]), "binary_unseen_attack_detection"))
        reports = {}
        stream, out = writer(directory / "splits.csv", MANIFEST_FIELDS["splits"])
        with stream:
            for fold, held, interpretation in folds:
                assignment, boundaries = {}, {}
                for capture, groups in sorted(by_capture.items()):
                    if capture in held:
                        assignment.update({g["group_id"]:("final_test","whole_capture_holdout") for g in groups})
                    else:
                        ratios = (.7,.075,.075,.15) if protocol == "A" else (.7/.85,.075/.85,.075/.85)
                        assigned, cuts = allocate(groups, ratios, purge)
                        assignment.update(assigned)
                        boundaries[capture] = cuts
                counts = Counter()
                class_counts = defaultdict(Counter)
                group_counts = defaultdict(lambda: defaultdict(set))
                for oid,label,capture,session,gid,_,_ in db.execute("SELECT * FROM observations ORDER BY oid"):
                    role, reason = assignment[gid]
                    out.writerow(dict(zip(MANIFEST_FIELDS["splits"],(fold,oid,label,capture,session,gid,role,reason))))
                    counts[role] += 1
                    class_counts[label][role] += 1
                    group_counts[label][role].add(gid)
                supported = sorted(label for label in all_labels if all(class_counts[label][role] for role in ROLES))
                binary = {}
                for role in ROLES:
                    binary[role] = {"benign":class_counts["Normal"][role],
                                    "attack":sum(counter[role] for label,counter in class_counts.items() if label != "Normal")}
                dev_ok = all(binary[role]["benign"] and binary[role]["attack"] for role in ROLES[:3])
                test_ok = bool(binary["final_test"]["benign"]) and (interpretation == "benign_source_generalization" or bool(binary["final_test"]["attack"]))
                retained = sum(counts[role] for role in ROLES)
                reports[fold] = {"interpretation":interpretation, "held_out_captures":list(held),
                    "counts":dict(counts), "achieved_proportions":{role:counts[role]/retained if retained else 0 for role in ROLES},
                    "class_counts":{label:dict(class_counts[label]) for label in sorted(all_labels)},
                    "independent_groups":{label:{role:len(gs) for role,gs in sorted(group_counts[label].items())} for label in sorted(all_labels)},
                    "binary_support":binary, "closed_set_supported_classes":supported if protocol=="A" else [],
                    "unsupported_classes":sorted(all_labels-set(supported)) if protocol=="A" else [],
                    "feasible":bool(dev_ok and test_ok), "boundaries":boundaries,
                    "scope_label":"reduced_support" if protocol=="A" and set(supported)!=all_labels else interpretation}
        if protocol == "A":
            supported = set(reports["protocol_a"]["closed_set_supported_classes"])
            while True:
                excluded_groups = {gid for gid,label in db.execute("SELECT DISTINCT gid,label FROM observations") if label not in supported}
                support = defaultdict(set)
                for label,gid in db.execute("SELECT label,gid FROM observations"):
                    role = assignment[gid][0]
                    if gid not in excluded_groups and role in ROLES:
                        support[label].add(role)
                narrowed = {label for label,roles in support.items() if roles == set(ROLES)}
                if narrowed == supported:
                    break
                supported = narrowed
            stream, out = writer(directory/"closed_set.csv",MANIFEST_FIELDS["splits"])
            with stream:
                for row in rows(directory/"splits.csv"):
                    if row["group_id"] in excluded_groups:
                        row["partition"], row["reason"] = "excluded", "unsupported_closed_set_group"
                    out.writerow(row)
            reports["protocol_a"]["closed_set_manifest_classes"] = sorted(supported)
        return {"protocol":protocol,"derived_split_seed":split_seed,"folds":reports,
                "unsupported_captures":sorted(set(expected_captures)-set(by_capture)),
                "ineligible_attack_folds_without_benign":attacks if not benign else [],
                "eligible_for_training":False}
    finally:
        db.close()

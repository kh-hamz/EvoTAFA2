"""Panel selection from trusted rows only, using a fixed seed and stable ordering."""

import hashlib
from collections import defaultdict
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.reproducibility import derive_seed
from edgefl.data.storage import database, rows, writer


def panel(splits: Path, split_report: dict, directory: Path, seed: int, budget: int) -> dict:
    db = database(directory / "_work/panel.sqlite")
    db.execute("CREATE TABLE trusted(fold TEXT,oid TEXT,label TEXT,capture TEXT,session TEXT,gid TEXT,priority TEXT,PRIMARY KEY(fold,oid))")
    derived = derive_seed(seed, "global_split")
    try:
        for row in rows(splits):
            if row["partition"] != "trusted":
                continue
            priority = hashlib.sha256(f"{derived}:{row['fold']}:{row['observation_id']}".encode()).hexdigest()
            db.execute("INSERT INTO trusted VALUES (?,?,?,?,?,?,?)",
                       (row["fold"],row["observation_id"],row["label"],row["capture_id"],row["session_id"],row["group_id"],priority))
        db.commit()
        db.execute("CREATE INDEX panel_order ON trusted(fold,label,priority,oid)")
        reports = {}
        stream, out = writer(directory / "panel.csv", MANIFEST_FIELDS["panel"])
        with stream:
            for fold, info in sorted(split_report["folds"].items()):
                counts, groups = {}, defaultdict(set)
                for label in sorted(info["class_counts"]):
                    selected = db.execute("SELECT oid,capture,session,gid FROM trusted WHERE fold=? AND label=? ORDER BY priority,oid LIMIT ?", (fold,label,budget))
                    count = 0
                    for oid,capture,session,gid in selected:
                        out.writerow(dict(zip(MANIFEST_FIELDS["panel"],(fold,oid,label,capture,session,gid,"trusted"))))
                        groups[label].add(gid)
                        count += 1
                    counts[label] = count
                reports[fold] = {"observations":counts,"independent_groups":{label:len(groups[label]) for label in counts},
                                 "missing_classes":[label for label,n in counts.items() if not n],
                                 "usable_for_fpr":bool(counts.get("Normal",0)),
                                 "role":"trusted"}
        return {"folds":reports,"per_class_budget":budget,"derived_split_seed":derived,
                "eligible_for_training":False}
    finally:
        db.close()

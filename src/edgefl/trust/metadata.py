"""Disk-backed original-label alignment over verified Phase D membership."""

import sqlite3
from contextlib import closing
import numpy as np
from edgefl.data.storage import rows, write_json, read_json
from edgefl.learning.storage import artifact, checked


def prepare(config, prepared, directory):
    spec = read_json(artifact(config, prepared, "data.json"))
    shards = {(s["role"], s["client"]): s for s in spec["shards"]}
    arrays = {key: np.load(artifact(config, prepared, s["y"]), mmap_mode="r", allow_pickle=False) for key, s in shards.items()}
    path = directory / "metadata.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE observations(role TEXT, client TEXT, row INTEGER, oid TEXT UNIQUE, original TEXT, target INTEGER, PRIMARY KEY(role,client,row))")
        for item in rows(artifact(config, prepared, "learning_rows.csv")):
            key, index = (item["role"], item["client"]), int(item["row"])
            if key not in shards or not 0 <= index < shards[key]["count"]:
                raise ValueError("Original-label metadata row outside registered shard")
            label = item["label"]
            mapped = ("Normal" if label == "Normal" else "Attack") if spec["identity"]["task"] == "binary" else label
            if mapped not in spec["labels"] or int(arrays[key][index]) != spec["labels"][mapped]:
                raise ValueError("Original-label metadata disagrees with task label")
            try:
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?)", (*key, index, item["observation_id"], label, spec["labels"][mapped]))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Duplicate original-label metadata row") from exc
        counts = {(r, c): n for r, c, n in db.execute("SELECT role,client,count(*) FROM observations GROUP BY role,client")}
        if counts != {key: s["count"] for key, s in shards.items()}:
            raise ValueError("Incomplete original-label metadata coverage")
        profiles = {}
        for client in sorted(c for r, c in shards if r == "local_train"):
            support = dict(db.execute("SELECT original,count(*) FROM observations WHERE role='local_train' AND client=? GROUP BY original", (client,)))
            total = sum(support.values())
            profiles[client] = {"count": total, "support": support, "concentrated": max(support.values()) / total >= .8}
        db.commit()
    write_json(directory / "metadata.json", {"schema_version": "phase-e.metadata.v1", "profiles": profiles,
               "task": spec["identity"]["task"], "labels": spec["labels"], "observations": sum(counts.values()),
               "declared_specialist_scenario": spec["identity"]["scenario"] == "specialist"})
    return {"clients": len(profiles), "observations": sum(counts.values())}


class Metadata:
    def __init__(self, root, completion):
        self.path = checked(root, completion["artifacts"]["metadata.sqlite"])
        self.spec = read_json(checked(root, completion["artifacts"]["metadata.json"]))
        if set(self.spec) != {"schema_version", "profiles", "task", "labels", "observations", "declared_specialist_scenario"} or self.spec["schema_version"] != "phase-e.metadata.v1":
            raise ValueError("Invalid Phase E metadata schema")

    def rows(self, role, client=""):
        if role not in ("local_train", "trusted") or (role == "trusted" and client):
            raise ValueError("Forbidden original-label metadata role")
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            return [{"row": row, "observation_id": oid, "original": original, "target": target}
                    for row, oid, original, target in db.execute(
                        "SELECT row,oid,original,target FROM observations WHERE role=? AND client=? ORDER BY row", (role, client))]

"""Bounded materialization from verified Phase C membership, never final-test rows."""

import bisect
import shutil
from pathlib import Path
import numpy as np
from edgefl.config import workspace_path
from edgefl.data.csv_reader import header, records
from edgefl.data.storage import database, read_json, write_json, rows
from edgefl.client_data.config import phase_b
from edgefl.client_data.preprocessing import FrozenPreprocessor
from edgefl.learning.config import phase_c
from edgefl.learning.storage import verify_gate, checked, reference


def materialize(config, gate, identity, directory):
    bundle = verify_gate(config, gate, identity)
    refs = {key: workspace_path(config.workspace, path) for key, path, digest in bundle.references}
    c = phase_c(config)
    gate_data = read_json(gate)
    provenance_completion = read_json(workspace_path(config.workspace, gate_data["inputs"]["provenance"]["path"]))
    provenance = workspace_path(config.workspace, provenance_completion["artifacts"]["provenance"]["path"])
    source = workspace_path(config.workspace, phase_b(c).values["selected"][identity["dataset"]])
    transformer = FrozenPreprocessor.load(refs["transformer"])
    features = read_json(refs["feature_contract"])
    mapping = read_json(refs["label_mappings"])[identity["task"]]
    variant = identity["variant"]
    dimension = features["variants"][variant]["output_features"]
    work = directory / "_work"
    work.mkdir()
    db = database(work / "membership.sqlite")
    db.execute("CREATE TABLE members(oid TEXT PRIMARY KEY,role TEXT,client TEXT,label TEXT,record INTEGER UNIQUE)")
    try:
        for row in rows(refs["local_splits"]):
            db.execute("INSERT INTO members(oid,role,client,label) VALUES (?,?,?,?)",
                       (row["observation_id"], row["partition"], row["client_id"], row["label"]))
        for key, permitted in (("global_splits", "selection_validation"), ("trusted_panel", "trusted")):
            for row in rows(refs[key]):
                if row["partition"] == permitted:
                    db.execute("INSERT INTO members(oid,role,client,label) VALUES (?,?,?,?)",
                               (row["observation_id"], permitted, "", row["label"]))
        for row in rows(provenance):
            db.execute("UPDATE members SET record=? WHERE oid=?", (int(row["record"]), row["observation_id"]))
        if db.execute("SELECT count(*) FROM members WHERE record IS NULL").fetchone()[0]:
            raise ValueError("Learning observation lacks verified source record")
        counts = list(db.execute("SELECT role,client,count(*) FROM members GROUP BY role,client ORDER BY role,client"))
        needed = sum(n for role, client, n in counts) * (dimension * 4 + 8)
        if shutil.disk_usage(directory).free < needed + 64 * 1024**2:
            raise ValueError("Insufficient disk space for learning arrays")
        arrays, shards, positions = {}, [], {}
        for i, (role, client, count) in enumerate(counts):
            stem = f"shard-{i:04d}"
            x_path, y_path = directory / (stem + "-x.npy"), directory / (stem + "-y.npy")
            arrays[(role, client)] = (np.lib.format.open_memmap(x_path, mode="w+", dtype="float32", shape=(count, dimension)),
                                      np.lib.format.open_memmap(y_path, mode="w+", dtype="int64", shape=(count,)))
            positions[(role, client)] = 0
            shards.append({"role": role, "client": client, "count": count, "x": x_path.name, "y": y_path.name})
        names = header(source)
        membership_path = directory / "learning_rows.csv"
        import csv
        with membership_path.open("w", newline="", encoding="utf-8") as stream:
            out = csv.writer(stream, lineterminator="\n")
            out.writerow(("observation_id", "record", "role", "client", "label", "row"))
            for record in records(source):
                member = db.execute("SELECT oid,role,client,label FROM members WHERE record=?", (record.number,)).fetchone()
                if member is None:
                    continue
                if record.error:
                    raise ValueError("Malformed admitted learning observation")
                oid, role, client, label = member
                key, index = (role, client), positions[(role, client)]
                vector = np.asarray(transformer.transform(names, record.values, variant), dtype=np.float32)
                if not np.isfinite(vector).all():
                    raise ValueError("Nonfinite Float32 feature conversion")
                target = ("Normal" if label == "Normal" else "Attack") if identity["task"] == "binary" else label
                arrays[key][0][index] = vector
                arrays[key][1][index] = mapping[target]
                positions[key] += 1
                out.writerow((oid, record.number, role, client, label, index))
        for role, client, count in counts:
            if positions[(role, client)] != count:
                raise ValueError("Incomplete learning materialization")
        for x, y in arrays.values():
            x.flush()
            y.flush()
        arrays.clear()
        spec = {"schema_version": "phase-d.data.v1", "identity": identity, "dimension": dimension,
                "labels": mapping, "features": features["variants"][variant],
                "transformer_sha256": features["transformer_sha256"],
                "label_map_sha256": features["label_mappings_sha256"], "shards": shards,
                "experiment": read_json(refs["experiment_contract"])}
        write_json(directory / "data.json", spec)
        return {"observations": sum(positions.values()), "dimension": dimension, "final_test_materialized": False}
    finally:
        db.close()


class ArrayRows:
    """Read-only indexed rows; only the selected role/client shards are exposed."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.ends = np.cumsum([len(y) for x, y in pairs]).tolist()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        import torch
        shard = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[shard - 1] if shard else 0)
        x, y = self.pairs[shard]
        return torch.from_numpy(np.array(x[offset], copy=True)), int(y[offset])


class LearningData:
    def __init__(self, config, completion):
        self.config, self.completion = config, completion
        self.spec = read_json(checked(config.workspace, completion["artifacts"]["data.json"]))
        self.clients = sorted({s["client"] for s in self.spec["shards"] if s["role"] == "local_train"})

    def view(self, role, client=None):
        if role not in ("local_train", "local_validation", "selection_validation", "trusted"):
            raise ValueError("Forbidden learning data role")
        pairs = []
        for shard in self.spec["shards"]:
            if shard["role"] != role or (client is not None and shard["client"] != client):
                continue
            x, y = [np.load(checked(self.config.workspace, self.completion["artifacts"][shard[k]]),
                            mmap_mode="r", allow_pickle=False) for k in ("x", "y")]
            if x.shape != (shard["count"], self.spec["dimension"]) or y.shape != (shard["count"],):
                raise ValueError("Learning array shape mismatch")
            if x.dtype != np.float32 or y.dtype != np.int64:
                raise ValueError("Learning array dtype mismatch")
            pairs.append((x, y))
        return ArrayRows(pairs)

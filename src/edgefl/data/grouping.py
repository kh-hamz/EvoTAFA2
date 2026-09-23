"""Observation deduplication, collision reports, and capture-scoped isolated groups."""

from collections import Counter
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.data.schema import digest
from edgefl.data.storage import database, rows, writer


def group(provenance: Path, evidence: Path, directory: Path, block_seconds: int) -> dict:
    db = database(directory / "_work/grouping.sqlite")
    db.executescript("""
    CREATE TABLE seen(identity TEXT PRIMARY KEY, oid TEXT) WITHOUT ROWID;
    CREATE TABLE features(key TEXT, label TEXT, oid TEXT);
    CREATE INDEX features_key ON features(key);
    CREATE TABLE support(label TEXT, capture TEXT, gid TEXT, oid TEXT);
    """)
    totals = Counter()
    group_stream, groups = writer(directory / "groups.csv", MANIFEST_FIELDS["groups"])
    duplicate_stream, duplicates = writer(directory / "duplicates.csv", MANIFEST_FIELDS["duplicates"])
    try:
        with group_stream, duplicate_stream:
            for index, (source, verified) in enumerate(zip(rows(provenance), rows(evidence), strict=True), 1):
                oid = source["observation_id"]
                if oid != verified["observation_id"]:
                    raise ValueError("Provenance/evidence observation order mismatch")
                if source["feature_key"]:
                    db.execute("INSERT INTO features VALUES (?,?,?)", (source["feature_key"], source["label"], oid))
                valid = verified["status"] == "verified"
                identity = ("packet:" + verified["capture_id"] + ":" + verified["packet"]) if valid else ("exact:" + source["exact_key"])
                first = db.execute("SELECT oid FROM seen WHERE identity=?", (identity,)).fetchone()
                duplicate = first is not None
                representative = first[0] if first else oid
                if duplicate:
                    duplicates.writerow({"observation_id": oid, "representative_id": representative,
                                          "reason": "repeated_packet" if valid else "repeated_exact_selected_record"})
                else:
                    db.execute("INSERT INTO seen VALUES (?,?)", (identity, oid))
                if valid and not duplicate:
                    session = verified["session_id"]
                    start, end = float(verified["session_start"]), float(verified["session_end"])
                    if session:
                        group_identity = [verified["capture_id"], "session", session]
                    else:
                        block = int(float(verified["epoch"]) // block_seconds)
                        group_identity = [verified["capture_id"], "temporal", block]
                        start, end = block * block_seconds, (block + 1) * block_seconds
                    gid = digest(group_identity)
                    status, reason = "verified", "verified_session" if session else "verified_temporal_block"
                    db.execute("INSERT INTO support VALUES (?,?,?,?)", (source["label"], verified["capture_id"], gid, oid))
                else:
                    gid, session, start, end = "", verified["session_id"], "", ""
                    status, reason = "quarantined", "duplicate_observation" if duplicate else verified["reason"]
                groups.writerow(dict(zip(MANIFEST_FIELDS["groups"],
                    (oid,source["label"],verified["capture_id"],session,gid,start,end,status,reason,representative))))
                totals[reason] += 1
                if index % 10000 == 0:
                    db.commit()
        db.commit()
        stream, out = writer(directory / "feature_collisions.csv", ("feature_key", "observations", "labels", "conflicting_labels"))
        collisions, conflicts = 0, 0
        with stream:
            for key, count, labels, distinct in db.execute("""SELECT key,count(*),group_concat(DISTINCT label),
                count(DISTINCT label) FROM features GROUP BY key HAVING count(*)>1 ORDER BY key"""):
                collisions += 1
                conflicts += distinct > 1
                out.writerow({"feature_key": key, "observations": count, "labels": "|".join(sorted(labels.split(","))),
                              "conflicting_labels": int(distinct > 1)})
        support = {label: {"observations": n, "independent_groups": g, "captures": c}
                   for label,n,g,c in db.execute("SELECT label,count(*),count(DISTINCT gid),count(DISTINCT capture) FROM support GROUP BY label")}
        largest = [{"group_id": gid, "observations": count} for gid,count in
                   db.execute("SELECT gid,count(*) FROM support GROUP BY gid ORDER BY count(*) DESC,gid LIMIT 20")]
        return {"reasons": dict(totals), "class_support": support, "largest_groups": largest,
                "feature_collision_vectors": collisions, "conflicting_label_vectors": conflicts,
                "feature_projection": "All declared predictors except time and source/destination host; no fitted transformation",
                "session_policy": "TCP stream spans stay intact, including long sessions; UDP uses verified stream and inactivity."}
    finally:
        db.close()

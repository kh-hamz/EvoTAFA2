"""Additional checks for source candidate identities and complete selected membership."""

from collections import Counter
from pathlib import Path

from edgefl.data.storage import database, rows


def validate_origins(provenance: Path, candidates: Path, registry: dict, audit: dict,
                     dataset: str, db_path: Path) -> list[str]:
    db = database(db_path)
    db.execute("CREATE TABLE origins(key TEXT,capture TEXT,n INTEGER,PRIMARY KEY(key,capture,n))")
    db.execute("CREATE INDEX origins_key ON origins(key)")
    files = {item["path"]:item for item in registry["files"]}
    pairs = {pair["capture_id"]:pair for pair in registry["pairs"]}
    selected = files[registry["selected"][dataset]]
    errors = set()
    try:
        for row in rows(candidates):
            pair = pairs.get(row["capture_id"])
            if not pair:
                errors.add("Candidate references unknown capture")
                continue
            source = files[pair["csv"]]
            number = int(row["source_record"])
            if (source["sha256"] != row["source_sha256"] or number < 1
                    or number > audit["files"][pair["csv"]].get("logical_records",0)):
                errors.add("Candidate source identity/record out of range")
            if len(row["exact_key"]) != 64 or len(row["record_key"]) != 64:
                errors.add("Invalid source candidate fingerprints")
            db.execute("INSERT INTO origins VALUES (?,?,?)",(row["record_key"],row["capture_id"],number))
        db.commit()
        db.execute("CREATE TABLE counts AS SELECT key,count(*) AS n FROM origins GROUP BY key")
        db.execute("CREATE UNIQUE INDEX counts_key ON counts(key)")
        labels = Counter()
        for number,row in enumerate(rows(provenance),1):
            if row["record"] != str(number) or row["observation_id"] != f"{selected['sha256']}:{number}":
                errors.add("Selected identity/order mismatch")
            result = db.execute("SELECT n FROM counts WHERE key=?",(row["record_key"],)).fetchone()
            actual = result[0] if result else 0
            if int(row["candidate_count"]) != actual:
                errors.add("Provenance candidate count mismatch")
            status = row["status"]
            if (status == "verified" and actual != 1 or status == "ambiguous" and actual < 2
                    or status == "unmatched" and actual != 0
                    or status not in ("verified","ambiguous","unmatched","quarantined","unsupported")):
                errors.add("Invalid provenance confidence classification")
            if row["label"]:
                labels[row["label"]] += 1
        expected_labels = audit["files"][registry["selected"][dataset]].get("canonical_labels",{})
        if dict(labels) != expected_labels:
            errors.add("Selected class membership differs from audit")
        return sorted(errors)
    finally:
        db.close()

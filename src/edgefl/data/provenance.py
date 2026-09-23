"""Preserve selected membership and export all candidate source origins."""

from collections import Counter
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.data.csv_reader import header, records
from edgefl.data.schema import (EXCLUDE_FEATURES, canonical_values, digest, evidence_key,
                                label, record_key, semantic_issues)
from edgefl.data.storage import database, writer


def recover(root: Path, registry: dict, audit: dict, selected_role: str,
            directory: Path, progress=lambda **kw: None, chunk_size=10000) -> dict:
    selected = registry["selected"][selected_role]
    identity = next(item["sha256"] for item in registry["files"] if item["path"] == selected)
    path, aliases = root / selected, registry["label_aliases"]
    names = header(path)
    db = database(directory / "_work/provenance.sqlite")
    db.executescript("""
    CREATE TABLE selected(n INTEGER PRIMARY KEY, observation_id TEXT, key TEXT, exact TEXT,
        feature TEXT, evidence TEXT, label TEXT, problem TEXT);
    CREATE INDEX selected_key ON selected(key);
    CREATE TABLE wanted(key TEXT PRIMARY KEY) WITHOUT ROWID;
    CREATE TABLE candidate(key TEXT, capture TEXT, source_sha TEXT, n INTEGER, exact TEXT);
    CREATE INDEX candidate_key ON candidate(key);
    """)
    totals = Counter()
    try:
        for record in records(path):
            problem = record.error
            if not problem:
                problem = ";".join(semantic_issues(names, record.values, aliases))
                key = record_key(names, record.values, aliases)
                normalized = canonical_values(names, record.values, aliases)
                feature = digest([(name, value) for name, value in zip(names, normalized)
                                  if name not in EXCLUDE_FEATURES])
                evidence = evidence_key(names, record.values) or ""
                canonical_label = label(record.values[names.index("Attack_type")], aliases)
                db.execute("INSERT OR IGNORE INTO wanted VALUES (?)", (key,))
            else:
                key, feature, evidence, canonical_label = "", "", "", ""
            db.execute("INSERT INTO selected VALUES (?,?,?,?,?,?,?,?)",
                       (record.number, f"{identity}:{record.number}", key, digest(record.values),
                        feature, evidence, canonical_label, problem))
            if record.number % chunk_size == 0:
                db.commit()
        db.commit()
        hashes = {entry["path"]: entry["sha256"] for entry in registry["files"]}
        skipped = []
        for pair in registry["pairs"]:
            source = root / pair["csv"]
            if audit["files"][pair["csv"]].get("structural_failure") or header(source) != names:
                skipped.append(pair["capture_id"])
                continue
            for record in records(source):
                if record.error:
                    continue
                key = record_key(names, record.values, aliases)
                if db.execute("SELECT 1 FROM wanted WHERE key=?", (key,)).fetchone():
                    db.execute("INSERT INTO candidate VALUES (?,?,?,?,?)",
                               (key, pair["capture_id"], hashes[pair["csv"]],
                                record.number, digest(record.values)))
                if record.number % chunk_size == 0:
                    db.commit()
                if record.number % 100000 == 0:
                    progress(event="provenance_progress", capture=pair["capture_id"], records=record.number)
            db.commit()
            progress(event="provenance_source_complete", capture=pair["capture_id"])
        db.execute("CREATE TABLE counts AS SELECT key,count(*) AS n FROM candidate GROUP BY key")
        db.execute("CREATE UNIQUE INDEX counts_key ON counts(key)")
        stream, out = writer(directory / "provenance.csv", MANIFEST_FIELDS["provenance"])
        with stream:
            query = "SELECT s.*,coalesce(c.n,0) FROM selected s LEFT JOIN counts c ON s.key=c.key ORDER BY s.n"
            for n, oid, key, exact, feature, evidence, canonical_label, problem, count in db.execute(query):
                status = "quarantined" if problem else ("verified" if count == 1 else "ambiguous" if count else "unmatched")
                reason = problem or ("unique_canonical_source" if count == 1 else "multiple_candidate_origins" if count else "no_canonical_source")
                totals[status] += 1
                out.writerow(dict(zip(MANIFEST_FIELDS["provenance"],
                                      (oid,n,key,exact,feature,evidence,canonical_label,status,count,reason))))
        stream, out = writer(directory / "candidates.csv", MANIFEST_FIELDS["candidates"])
        with stream:
            for entry in db.execute("SELECT * FROM candidate ORDER BY key,capture,n"):
                out.writerow(dict(zip(MANIFEST_FIELDS["candidates"], entry)))
    finally:
        db.close()
    return {"selected": selected, "selected_sha256": identity, "observations": sum(totals.values()),
            "statuses": dict(totals), "skipped_source_captures": skipped,
            "membership": "Original selected logical records; never a source concatenation"}

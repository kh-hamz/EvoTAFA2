"""Memory-bounded audits with disk-backed duplicate and cardinality indexes."""

from collections import Counter
from pathlib import Path

from edgefl.data.csv_reader import header, records
from edgefl.data.schema import canonical, digest, label, semantic_issues, time_category, projected_time, value_digest
from edgefl.data.storage import database


def audit_csv(path: Path, db_path: Path, aliases: dict, progress=lambda **kw: None,
              chunk_size=10000) -> dict:
    names = header(path)
    db = database(db_path)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS duplicate(digest TEXT PRIMARY KEY) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS cardinality(field INTEGER, digest TEXT, PRIMARY KEY(field,digest)) WITHOUT ROWID;
    DELETE FROM duplicate; DELETE FROM cardinality;
    """)
    counts, labels, raw_labels, issues, stamps = Counter(), Counter(), Counter(), Counter(), Counter()
    examples, missing = [], Counter()
    caches = [set() for _ in names]
    previous = None
    try:
        for record in records(path):
            counts["logical_records"] += 1
            if record.error:
                counts["malformed_records"] += 1
                if len(examples) < 20:
                    examples.append({"record": record.number, "issue": record.error, "width": len(record.values)})
                continue
            counts["valid_width_records"] += 1
            inserted = db.execute("INSERT OR IGNORE INTO duplicate VALUES (?)", (digest(record.values),)).rowcount
            counts["exact_duplicate_occurrences"] += 1 - inserted
            value = dict(zip(names, record.values))
            raw = value.get("Attack_type", "")
            raw_labels[raw] += 1
            labels[label(raw, aliases)] += 1
            for issue in semantic_issues(names, record.values, aliases):
                issues[issue] += 1
                if len(examples) < 20:
                    examples.append({"record": record.number, "issue": issue})
            stamp = value.get("frame.time", "")
            stamps[time_category(stamp)] += 1
            projected = projected_time(stamp)
            if projected and previous and projected < previous:
                counts["time_regressions_or_day_rollovers"] += 1
            if projected:
                previous = projected
            for index, raw in enumerate(record.values):
                if not raw.strip():
                    missing[names[index]] += 1
                caches[index].add(value_digest(raw))
                if len(caches[index]) >= 8192:
                    db.executemany("INSERT OR IGNORE INTO cardinality VALUES (?,?)",
                                   ((index, entry) for entry in caches[index]))
                    caches[index].clear()
            if record.number % chunk_size == 0:
                db.commit()
                progress(event="audit_progress", path=path.name, records=record.number)
        for index, cache in enumerate(caches):
            db.executemany("INSERT OR IGNORE INTO cardinality VALUES (?,?)", ((index, entry) for entry in cache))
        db.commit()
        cardinalities = {names[i]: n for i, n in db.execute("SELECT field,count(*) FROM cardinality GROUP BY field")}
    finally:
        db.close()
    return {"header": names, "columns": len(names), **dict(counts),
            "raw_labels": dict(sorted(raw_labels.items())), "canonical_labels": dict(sorted(labels.items())),
            "semantic_issues": dict(sorted(issues.items())), "diagnostic_examples": examples,
            "missing": dict(sorted(missing.items())), "field_cardinalities": cardinalities,
            "timestamp_formats": dict(sorted(stamps.items())),
            "timestamp_ordering": "Regressions may be resets or midnight rollovers; dates are not inferred."}

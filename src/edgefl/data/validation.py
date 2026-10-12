"""Independent integrity and leakage checks across portable Phase B manifests."""

from collections import Counter
from pathlib import Path

from edgefl.data.storage import database, rows
from edgefl.data.splitting import ROLES


def validate(paths: dict[str, Path], summaries: dict, directory: Path, purge: int,
             panel_budget: int, expected_records: int) -> dict:
    db = database(directory / "_work/validation.sqlite")
    db.executescript("""
    CREATE TABLE provenance(oid TEXT PRIMARY KEY,status TEXT,label TEXT);
    CREATE TABLE evidence(oid TEXT PRIMARY KEY,status TEXT,capture TEXT,packet TEXT);
    CREATE TABLE groups(oid TEXT PRIMARY KEY,label TEXT,capture TEXT,session TEXT,gid TEXT,
                        start REAL,end REAL,status TEXT,rep TEXT);
    CREATE TABLE split(protocol TEXT,fold TEXT,oid TEXT,label TEXT,capture TEXT,session TEXT,
                       gid TEXT,role TEXT,PRIMARY KEY(protocol,fold,oid));
    CREATE INDEX split_capture_role ON split(protocol,fold,capture,role,oid);
    CREATE INDEX split_group_role ON split(protocol,fold,gid,role);
    CREATE TABLE panel(protocol TEXT,fold TEXT,oid TEXT,label TEXT,PRIMARY KEY(protocol,fold,oid));
    """)
    errors, limitations = [], []
    def require(condition, message):
        if not condition and message not in errors:
            errors.append(message)
    try:
        for row in rows(paths["provenance"]):
            db.execute("INSERT INTO provenance VALUES (?,?,?)", (row["observation_id"],row["status"],row["label"]))
        for row in rows(paths["evidence"]):
            db.execute("INSERT INTO evidence VALUES (?,?,?,?)",
                       (row["observation_id"],row["status"],row["capture_id"],row["packet"]))
        for row in rows(paths["groups"]):
            db.execute("INSERT INTO groups VALUES (?,?,?,?,?,?,?,?,?)",
                       (row["observation_id"],row["label"],row["capture_id"],row["session_id"],row["group_id"],
                        float(row["start"]) if row["start"] else None,float(row["end"]) if row["end"] else None,
                        row["status"],row["representative_id"]))
        db.commit()
        require(db.execute("SELECT count(*) FROM provenance").fetchone()[0] == expected_records,
                "Selected logical-record coverage mismatch")
        for table in ("evidence","groups"):
            require(not db.execute(f"SELECT oid FROM provenance EXCEPT SELECT oid FROM {table}").fetchone(),
                    f"Missing observations in {table}")
            require(not db.execute(f"SELECT oid FROM {table} EXCEPT SELECT oid FROM provenance").fetchone(),
                    f"Unknown observations in {table}")
        require(not db.execute("""SELECT 1 FROM groups g JOIN evidence e ON g.oid=e.oid
            WHERE g.status='verified' AND (e.status!='verified' OR g.capture!=e.capture
                OR g.gid='' OR g.start IS NULL OR g.end<g.start) LIMIT 1""").fetchone(),
                "Retained groups lack valid capture evidence")
        require(not db.execute("""SELECT 1 FROM groups g JOIN provenance p ON g.oid=p.oid
            WHERE g.label!=p.label OR (g.status='verified' AND p.status!='verified') LIMIT 1""").fetchone(),
                "Group labels/provenance disagree")
        require(not db.execute("""SELECT 1 FROM groups g JOIN evidence e ON e.oid=g.oid
            WHERE g.status='verified' GROUP BY e.capture,e.packet HAVING count(*)>1 LIMIT 1""").fetchone(),
                "Repeated packet retained more than once")
        require(not db.execute("""SELECT 1 FROM groups g LEFT JOIN groups r ON r.oid=g.rep
            WHERE r.oid IS NULL OR r.rep!=r.oid LIMIT 1""").fetchone(),
                "Duplicate representative missing or chained")
        actual_duplicates = set()
        for row in rows(paths["duplicates"]):
            pair = (row["observation_id"],row["representative_id"])
            require(pair not in actual_duplicates, "Duplicate lineage row repeated")
            actual_duplicates.add(pair)
            require(bool(db.execute("SELECT 1 FROM groups WHERE oid=? AND rep=? AND oid!=rep", pair).fetchone()),
                    "Duplicate lineage disagrees with grouping")
        require(db.execute("SELECT count(*) FROM groups WHERE oid!=rep").fetchone()[0] == len(actual_duplicates),
                "Duplicate lineage incomplete")
        verified_count = db.execute("SELECT count(*) FROM groups WHERE status='verified'").fetchone()[0]
        quarantined = db.execute("SELECT count(*) FROM groups WHERE status!='verified'").fetchone()[0]
        if quarantined:
            limitations.append(f"{quarantined} observations quarantined or deduplicated")
        require(verified_count > 0, "No verified observations available")
        usable_folds = 0
        for protocol in ("a","b"):
            report = summaries["splits_" + protocol]
            for row in rows(paths["splits_" + protocol]):
                require(row["partition"] in (*ROLES,"excluded"), "Unknown partition role")
                require(row["fold"] in report["folds"], "Unknown split fold")
                db.execute("INSERT INTO split VALUES (?,?,?,?,?,?,?,?)",
                           (protocol,row["fold"],row["observation_id"],row["label"],row["capture_id"],
                            row["session_id"],row["group_id"],row["partition"]))
            db.commit()
            require(not db.execute("""SELECT 1 FROM split s LEFT JOIN groups g ON g.oid=s.oid
                WHERE s.protocol=? AND (g.oid IS NULL OR g.status!='verified' OR s.gid!=g.gid
                OR s.capture!=g.capture OR s.session!=g.session OR s.label!=g.label) LIMIT 1""", (protocol,)).fetchone(),
                    "Split references excluded/unknown/mismatched observation")
            require(not db.execute("""SELECT 1 FROM split WHERE protocol=? GROUP BY fold,gid
                HAVING count(DISTINCT role)>1 LIMIT 1""",(protocol,)).fetchone(), "Group crosses split roles")
            require(not db.execute("""SELECT 1 FROM split WHERE protocol=? AND session!=''
                GROUP BY fold,capture,session HAVING count(DISTINCT role)>1 LIMIT 1""",(protocol,)).fetchone(),
                    "Session crosses split roles")
            for fold, info in report["folds"].items():
                require(db.execute("SELECT count(*) FROM split WHERE protocol=? AND fold=?",(protocol,fold)).fetchone()[0] == verified_count,
                        "Split fold does not cover every retained observation")
                counts = dict(db.execute("SELECT role,count(*) FROM split WHERE protocol=? AND fold=? GROUP BY role",(protocol,fold)))
                require(counts == info["counts"], "Split support report does not match manifest")
                for capture in info["held_out_captures"]:
                    require(not db.execute("SELECT 1 FROM split WHERE protocol=? AND fold=? AND capture=? AND role!='final_test' LIMIT 1",(protocol,fold,capture)).fetchone(),
                            "Held-out capture appears in development")
                captures = [r[0] for r in db.execute("SELECT DISTINCT capture FROM split WHERE protocol=? AND fold=?",(protocol,fold))]
                for capture in captures:
                    if capture in info["held_out_captures"]:
                        continue
                    previous_end = None
                    for role in ROLES:
                        bounds = db.execute("""SELECT min(g.start),max(g.end) FROM split s JOIN groups g ON s.oid=g.oid
                            WHERE s.protocol=? AND s.fold=? AND s.capture=? AND s.role=?""",(protocol,fold,capture,role)).fetchone()
                        if bounds[0] is not None:
                            require(previous_end is None or bounds[0]-previous_end >= 2*purge,
                                    "Chronological or purge separation violated")
                            previous_end = bounds[1]
                actual_support = {}
                actual_groups = {}
                for label,role,n,g in db.execute(
                        "SELECT label,role,count(*),count(DISTINCT gid) FROM split WHERE protocol=? AND fold=? GROUP BY label,role",
                        (protocol,fold)):
                    actual_support.setdefault(label,{})[role] = n
                    actual_groups.setdefault(label,{})[role] = g
                for label,declared in info["class_counts"].items():
                    require(declared == actual_support.get(label,{}),
                            "Per-class split support report mismatch")
                    require(info["independent_groups"].get(label,{}) == actual_groups.get(label,{}),
                            "Independent-group support report mismatch")
                require(set(actual_support) <= set(info["class_counts"]),
                        "Split support report omits a retained class")
                binary_support = {
                    role: {"benign":actual_support.get("Normal",{}).get(role,0),
                           "attack":sum(counts.get(role,0) for label,counts in actual_support.items()
                                        if label != "Normal")}
                    for role in ROLES}
                require(binary_support == info["binary_support"],"Binary support report mismatch")
                development_ok = all(v["benign"] and v["attack"]
                                     for role,v in binary_support.items() if role != "final_test")
                test_support = binary_support["final_test"]
                test_ok = bool(test_support["benign"]) and (
                    info["interpretation"] == "benign_source_generalization" or bool(test_support["attack"]))
                feasible = bool(development_ok and test_ok)
                require(info["feasible"] is feasible,"Reported fold feasibility is incorrect")
                if not feasible:
                    limitations.append(f"{protocol}/{fold}: insufficient binary pool support")
                if info.get("unsupported_classes"):
                    limitations.append(f"{protocol}/{fold}: unsupported closed-set classes: " + ", ".join(info["unsupported_classes"]))
            panel_info = summaries["panel_" + protocol]
            for row in rows(paths["panel_" + protocol]):
                require(row["partition"] == "trusted", "Panel contains non-trusted role")
                require(bool(db.execute("""SELECT 1 FROM split WHERE protocol=? AND fold=? AND oid=?
                    AND role='trusted' AND gid=? AND label=? AND capture=? AND session=?""",
                    (protocol,row["fold"],row["observation_id"],row["group_id"],row["label"],row["capture_id"],row["session_id"])).fetchone()),
                    "Panel contains observation outside matching trusted partition")
                db.execute("INSERT INTO panel VALUES (?,?,?,?)",(protocol,row["fold"],row["observation_id"],row["label"]))
            db.commit()
            for fold, info in panel_info["folds"].items():
                counts = dict(db.execute("SELECT label,count(*) FROM panel WHERE protocol=? AND fold=? GROUP BY label",(protocol,fold)))
                require(all(n <= panel_budget for n in counts.values()), "Panel class budget exceeded")
                require(all(counts.get(label,0)==n for label,n in info["observations"].items()),
                        "Panel support report mismatch")
                group_counts = dict(db.execute(
                    "SELECT p.label,count(DISTINCT s.gid) FROM panel p JOIN split s "
                    "ON p.protocol=s.protocol AND p.fold=s.fold AND p.oid=s.oid "
                    "WHERE p.protocol=? AND p.fold=? GROUP BY p.label",(protocol,fold)))
                require(all(group_counts.get(label,0)==n for label,n in info["independent_groups"].items()),
                        "Panel independent-group support mismatch")
                if not counts.get("Normal",0):
                    limitations.append(f"{protocol}/{fold}: panel unusable for FPR")
                elif report["folds"][fold]["feasible"]:
                    usable_folds += 1
        require(usable_folds > 0, "No feasible split with a benign-supported trusted panel")
        status = "FAIL" if errors else "PASS_WITH_LIMITATIONS" if limitations else "PASS"
        return {"status":status,"errors":errors,"limitations":limitations,
                "verified_observations":verified_count,"excluded_observations":quarantined,
                "usable_folds":usable_folds,"eligible_for_training":False,
                "phase_c_handoff":"Client allocation and pretraining gate are still required; no training authorization."}
    finally:
        db.close()

"""Independent Phase C leakage, compatibility, determinism, and readiness gate."""

import hashlib
from collections import Counter
from pathlib import Path

from edgefl.config import canonical_json
from edgefl.data.schema import LABELS
from edgefl.client_data.assignment import assign
from edgefl.client_data.local_split import split_local
from edgefl.client_data.preprocessing import FrozenPreprocessor, fit, verify_finite
from edgefl.client_data.storage import database, read_json, rows, sha256
from edgefl.client_data.policies import support_policy, class_counts, metric_support, MANDATORY_EXCLUSIONS, feature_policy_identity


class PretrainingGate:
    """Accumulates all gate failures so one run gives a complete diagnostic report."""

    def __init__(self):
        self.errors: list[str] = []
        self.limitations: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition and message not in self.errors:
            self.errors.append(message)

    def run(self, *, splits: Path, assignments: Path, local_splits: Path,
            client_manifests: Path, provenance: Path, selected: Path, transformer: Path,
            feature_dictionary: Path, feature_contract: Path, label_mappings: Path,
            fit_rows: Path, directory: Path, config: dict, fold: str, scenario: str,
            experiment=None, trusted_panel=None) -> dict:
        policy = support_policy(config)
        db = database(directory / "_work/gate.sqlite")
        db.executescript("""
        CREATE TABLE global_rows(oid TEXT PRIMARY KEY,label TEXT,capture TEXT,session TEXT,gid TEXT,role TEXT);
        CREATE TABLE assignment(oid TEXT PRIMARY KEY,client TEXT,label TEXT,capture TEXT,session TEXT,gid TEXT);
        CREATE TABLE local(oid TEXT PRIMARY KEY,client TEXT,label TEXT,capture TEXT,session TEXT,gid TEXT,role TEXT);
        CREATE TABLE fitted(oid TEXT PRIMARY KEY,client TEXT,record INTEGER,label TEXT);
        CREATE TABLE provenance(oid TEXT PRIMARY KEY,record INTEGER,label TEXT);
        """)
        try:
            for row in rows(splits):
                if row["fold"] == fold:
                    db.execute("INSERT INTO global_rows VALUES (?,?,?,?,?,?)", (
                        row["observation_id"], row["label"], row["capture_id"],
                        row["session_id"], row["group_id"], row["partition"]))
            for row in rows(assignments):
                self.require(row["fold"] == fold and row["scenario"] == scenario,
                             "Assignment manifest mixes fold or scenario")
                db.execute("INSERT INTO assignment VALUES (?,?,?,?,?,?)", (
                    row["observation_id"], row["client_id"], row["label"],
                    row["capture_id"], row["session_id"], row["group_id"]))
            for row in rows(local_splits):
                self.require(row["fold"] == fold and row["scenario"] == scenario,
                             "Local manifest mixes fold or scenario")
                self.require(row["partition"] in ("local_train", "local_validation"),
                             "Unknown client-local role")
                db.execute("INSERT INTO local VALUES (?,?,?,?,?,?,?)", (
                    row["observation_id"], row["client_id"], row["label"],
                    row["capture_id"], row["session_id"], row["group_id"],
                    row["partition"]))
            for row in rows(fit_rows):
                db.execute("INSERT INTO fitted VALUES (?,?,?,?)", (
                    row["observation_id"], row["client_id"], int(row["record"]), row["label"]))
            for row in rows(provenance):
                db.execute("INSERT INTO provenance VALUES (?,?,?)", (
                    row["observation_id"], int(row["record"]), row["label"]))
            db.commit()
            self.require(not db.execute("""SELECT oid FROM global_rows WHERE role='client_pool'
                EXCEPT SELECT oid FROM assignment""").fetchone(),
                         "Client assignment omits client-pool observations")
            self.require(not db.execute("""SELECT oid FROM assignment EXCEPT
                SELECT oid FROM global_rows WHERE role='client_pool'""").fetchone(),
                         "Client assignment references a protected or unknown observation")
            self.require(not db.execute("""SELECT 1 FROM assignment a JOIN global_rows g ON a.oid=g.oid
                WHERE a.label!=g.label OR a.capture!=g.capture OR a.session!=g.session OR a.gid!=g.gid
                LIMIT 1""").fetchone(), "Assignment metadata differs from Phase B split")
            self.require(not db.execute("""SELECT 1 FROM assignment GROUP BY gid
                HAVING count(DISTINCT client)>1 LIMIT 1""").fetchone(),
                         "A group crosses clients")
            self.require(not db.execute("""SELECT 1 FROM assignment WHERE session!=''
                GROUP BY capture,session HAVING count(DISTINCT client)>1 LIMIT 1""").fetchone(),
                         "A verified session crosses clients")
            self.require(not db.execute("SELECT oid FROM assignment EXCEPT SELECT oid FROM local").fetchone(),
                         "Local split omits assigned observations")
            self.require(not db.execute("SELECT oid FROM local EXCEPT SELECT oid FROM assignment").fetchone(),
                         "Local split contains unassigned observations")
            self.require(not db.execute("""SELECT 1 FROM local l JOIN assignment a ON l.oid=a.oid
                WHERE l.client!=a.client OR l.label!=a.label OR l.capture!=a.capture
                OR l.session!=a.session OR l.gid!=a.gid LIMIT 1""").fetchone(),
                         "Local split metadata differs from assignment")
            self.require(not db.execute("""SELECT 1 FROM local GROUP BY client,gid
                HAVING count(DISTINCT role)>1 LIMIT 1""").fetchone(),
                         "A group crosses local train and validation")
            self.require(not db.execute("""SELECT 1 FROM local WHERE session!=''
                GROUP BY client,capture,session HAVING count(DISTINCT role)>1 LIMIT 1""").fetchone(),
                         "A session crosses local train and validation")
            self.require(not db.execute("""SELECT oid FROM local WHERE role='local_train'
                EXCEPT SELECT oid FROM fitted""").fetchone(),
                         "Preprocessing fitting-row manifest is incomplete")
            self.require(not db.execute("""SELECT oid FROM fitted EXCEPT
                SELECT oid FROM local WHERE role='local_train'""").fetchone(),
                         "Preprocessing used a non-training observation")
            self.require(not db.execute("""SELECT 1 FROM fitted f JOIN provenance p ON f.oid=p.oid
                WHERE f.record!=p.record OR f.label!=p.label LIMIT 1""").fetchone(),
                         "Fitting-row provenance disagrees")
            self.require(not db.execute("SELECT 1 FROM local WHERE label NOT IN (%s) LIMIT 1" %
                                        ",".join("?" * len(LABELS)), tuple(LABELS)).fetchone(),
                         "Client manifests contain an invalid label")
            minimum = config["minimum_client_observations"]
            minimum_training = config["minimum_training_observations"]
            clients = sorted(row[0] for row in db.execute("SELECT DISTINCT client FROM assignment"))
            pool_counts = dict(db.execute("SELECT label,count(*) FROM global_rows WHERE role='client_pool' GROUP BY label"))
            pool_total = sum(pool_counts.values())
            self.require(len(clients) == config["clients"], "Client count differs from configuration")
            client_reports = {}
            for client in clients:
                total = db.execute("SELECT count(*) FROM local WHERE client=?", (client,)).fetchone()[0]
                training = db.execute("SELECT count(*) FROM local WHERE client=? AND role='local_train'",
                                      (client,)).fetchone()[0]
                validation = db.execute("SELECT count(*) FROM local WHERE client=? AND role='local_validation'",
                                        (client,)).fetchone()[0]
                groups = db.execute("SELECT count(DISTINCT gid) FROM local WHERE client=?", (client,)).fetchone()[0]
                support = dict(db.execute("SELECT label,count(*) FROM local WHERE client=? AND role='local_train' GROUP BY label",
                                          (client,)))
                validation_support = dict(db.execute("SELECT label,count(*) FROM local WHERE client=? AND role='local_validation' GROUP BY label", (client,)))
                assigned_support = dict(db.execute("SELECT label,count(*) FROM local WHERE client=? GROUP BY label", (client,)))
                benign = support.get("Normal", 0)
                attack = sum(n for label, n in support.items() if label != "Normal")
                self.require(total >= minimum, f"{client} is below the client observation minimum")
                self.require(training >= minimum_training,
                             f"{client} is below the training observation minimum")
                self.require(validation > 0, f"{client} has no local-validation observations")
                self.require(groups >= config["minimum_groups_per_client"],
                             f"{client} is below the independent-group minimum")
                for role, counts in (("assigned", assigned_support), ("training", support), ("validation", validation_support)):
                    self.require(not policy.violations(counts, role), f"{client} violates {role} support policy")
                if config["scenarios"][scenario]["kind"] == "near_iid":
                    target = pool_total / config["clients"]
                    deviation = abs(total - target) / max(target, 1)
                    distance = sum(abs(assigned_support.get(label, 0) / max(total, 1) - count / max(pool_total, 1))
                                   for label, count in pool_counts.items()) / 2
                    limits = config.get("near_iid_limits", {"maximum_size_deviation": 0.1, "maximum_total_variation": 0.05})
                    self.require(deviation <= limits["maximum_size_deviation"] + 1e-12, f"{client} violates near-IID size deviation")
                    self.require(distance <= limits["maximum_total_variation"] + 1e-12, f"{client} violates near-IID total variation")
                client_reports[client] = {"total": total, "local_train": training,
                                          "local_validation": validation, "groups": groups,
                                          "assigned_class_counts": class_counts(assigned_support),
                                          "training_class_counts": class_counts(support),
                                          "validation_class_counts": class_counts(validation_support),
                                          "metric_support": {"training": metric_support(support), "validation": metric_support(validation_support)}}
                group_support = {}
                for role, partition in (("assigned", None), ("training", "local_train"), ("validation", "local_validation")):
                    query = "SELECT label,count(DISTINCT gid) FROM local WHERE client=?"
                    parameters = [client]
                    if partition is not None:
                        query += " AND role=?"
                        parameters.append(partition)
                    counts = dict(db.execute(query + " GROUP BY label", parameters))
                    group_support[role] = class_counts(counts)
                client_reports[client]["class_group_counts"] = group_support
            manifests = read_json(client_manifests)
            self.require(set(manifests) == set(clients),
                         "Client-manifest identities differ from assignments")
            for client in clients:
                entry = manifests.get(client, {})
                support = entry.get("support", {})
                actual = client_reports[client]
                self.require(entry.get("local_split_sha256") == sha256(local_splits),
                             f"{client} manifest references the wrong local split")
                self.require(entry.get("training_role") == "local_train" and
                             entry.get("validation_role") == "local_validation",
                             f"{client} manifest has invalid local roles")
                self.require(support.get("observations") == actual["total"] and
                             support.get("local_train") == actual["local_train"] and
                             support.get("local_validation") == actual["local_validation"] and
                             support.get("groups") == actual["groups"],
                             f"{client} manifest support differs from local rows")
                for role in ("assigned", "training", "validation"):
                    self.require(support.get(role + "_class_counts") == actual[role + "_class_counts"],
                                 f"{client} {role} class table differs from local rows")
                    declared_groups = support.get("assigned_class_group_counts") if role == "assigned" else support.get("class_group_counts", {}).get(role)
                    self.require(declared_groups == actual["class_group_counts"][role],
                                 f"{client} {role} class group table differs from local rows")
                self.require(support.get("metric_support") == actual["metric_support"],
                             f"{client} metric support flags differ from local rows")
            observed_labels = {row[0] for row in db.execute(
                "SELECT DISTINCT label FROM local WHERE role='local_train'")}
            task = experiment["task"] if experiment else "multiclass"
            required_labels = set(experiment["labels"]) if experiment and task == "multiclass" else set(config.get("required_labels", ()))
            unsupported = sorted(required_labels - observed_labels)
            if unsupported:
                self.errors.append("Required classes unsupported by client training: " +
                                   ", ".join(unsupported))
            optional = sorted(set(LABELS) - required_labels - observed_labels)
            if experiment:
                self.require(experiment["fold"] == fold, "Experiment fold mismatch")
                for role in ("client_pool", "trusted", "selection_validation", "final_test"):
                    counts = dict(db.execute("SELECT label,count(*) FROM global_rows WHERE role=? GROUP BY label", (role,)))
                    self.require(bool(counts.get("Normal")), f"{role} lacks benign support")
                    benign_only = role == "final_test" and experiment["interpretation"] == "benign_source_generalization"
                    self.require(benign_only or any(n for label, n in counts.items() if label != "Normal"),
                                 f"{role} lacks attack support")
                    if task == "multiclass":
                        self.require(set(counts) == required_labels, f"{role} differs from the declared closed-set classes")
                if experiment["held_out_captures"]:
                    for capture in experiment["held_out_captures"]:
                        self.require(not db.execute("SELECT 1 FROM global_rows WHERE capture=? AND role!='final_test' LIMIT 1", (capture,)).fetchone(),
                                     "Held-out capture appears in development")
            panel_counts = Counter()
            if trusted_panel is not None:
                seen_panel = set()
                for row in rows(trusted_panel):
                    oid = row["observation_id"]
                    self.require(oid not in seen_panel, "Trusted panel contains duplicate observation")
                    seen_panel.add(oid)
                    actual = db.execute("SELECT label,capture,session,gid,role FROM global_rows WHERE oid=?", (oid,)).fetchone()
                    self.require(row["fold"] == fold and row["partition"] == "trusted" and actual == (
                        row["label"], row["capture_id"], row["session_id"], row["group_id"], "trusted"),
                        "Trusted panel differs from scoped trusted rows")
                    panel_counts[row["label"]] += 1
                self.require(bool(panel_counts.get("Normal")) and any(n for label, n in panel_counts.items() if label != "Normal"),
                             "Trusted panel lacks binary scoring support")
                if task == "multiclass":
                    self.require(set(panel_counts) == required_labels, "Trusted panel lacks declared-class scoring support")
            mapping = read_json(label_mappings)
            self.require(set(mapping["multiclass"]) == observed_labels,
                         "Multiclass label map differs from training labels")
            self.require(mapping.get("binary") == {"Normal": 0, "Attack": 1}, "Binary label map differs from fixed vocabulary")
            self.require(mapping.get("multiclass") == {label: index for index, label in enumerate(sorted(observed_labels))},
                         "Multiclass label ordering differs from common vocabulary")
            contract = read_json(feature_contract)
            self.require(contract.get("transformer_sha256") == sha256(transformer),
                         "Feature contract references the wrong transformer")
            self.require(contract.get("fit_rows_sha256") == sha256(fit_rows),
                         "Feature contract references the wrong fitting rows")
            self.require(contract.get("label_mappings_sha256") == sha256(label_mappings),
                         "Feature contract references the wrong label map")
            frozen = FrozenPreprocessor.load(transformer)
            identity = feature_policy_identity(config["preprocessing"])
            self.require(contract.get("feature_policy") == identity and frozen.specification.get("feature_policy") == identity,
                         "Feature policy identity differs from configuration")
            dictionary = list(rows(feature_dictionary))
            for variant, specification in frozen.specification["variants"].items():
                actual = [row for row in dictionary if row["variant"] == variant]
                actual.sort(key=lambda row: int(row["index"]))
                expected = specification["output_features"]
                self.require(not any(item["source"] in MANDATORY_EXCLUSIONS for item in expected),
                             f"{variant} retains a forbidden identifier/payload feature")
                order = [item["name"] for item in expected]
                declared = contract.get("variants", {}).get(variant, {})
                self.require(declared.get("output_features") == len(order) and
                             declared.get("feature_order_sha256") ==
                             hashlib.sha256(canonical_json(order).encode()).hexdigest(),
                             f"{variant} feature contract differs from transformer")
                self.require([int(row["index"]) for row in actual] == list(range(len(expected))),
                             f"{variant} feature dictionary indices are not contiguous")
                self.require([row["output_feature"] for row in actual] ==
                             [item["name"] for item in expected],
                             f"{variant} feature order differs from transformer")
            membership = database(directory / "_work/transform_membership.sqlite")
            membership.execute("CREATE TABLE membership(record INTEGER PRIMARY KEY,role TEXT)")
            for oid, role in db.execute("SELECT oid,role FROM local"):
                record = db.execute("SELECT record FROM provenance WHERE oid=?", (oid,)).fetchone()
                self.require(record is not None, "Client observation lacks provenance record")
                if record:
                    membership.execute("INSERT INTO membership VALUES (?,?)", (record[0], role))
            for oid, role in db.execute("""SELECT oid,role FROM global_rows
                WHERE role IN ('trusted','selection_validation','final_test')"""):
                record = db.execute("SELECT record FROM provenance WHERE oid=?", (oid,)).fetchone()
                self.require(record is not None, "Global evaluation observation lacks provenance record")
                if record:
                    membership.execute("INSERT INTO membership VALUES (?,?)", (record[0], role))
            membership.commit()
            try:
                finite_checks = verify_finite(selected, membership, frozen)
            except ValueError as exc:
                finite_checks = {}
                self.errors.append(str(exc))
            finally:
                membership.close()
            reproducibility = self._regenerate(
                splits=splits, assignments=assignments, local_splits=local_splits,
                provenance=provenance, selected=selected, transformer=transformer,
                client_manifests=client_manifests, feature_dictionary=feature_dictionary,
                feature_contract=feature_contract, label_mappings=label_mappings,
                fit_rows=fit_rows, directory=directory, config=config, fold=fold,
                scenario=scenario)
            self.require(all(reproducibility.values()), "Phase C artifacts do not regenerate deterministically")
            status = "FAIL" if self.errors else "PASS_WITH_LIMITATIONS" if self.limitations else "PASS"
            return {"status": status, "errors": self.errors, "limitations": self.limitations,
                    "eligible_for_training": status == "PASS", "fold": fold,
                    "scenario": scenario, "clients": client_reports,
                    "task": task, "experiment": experiment, "trusted_panel_support": dict(panel_counts),
                    "unsupported_required_classes": unsupported,
                    "unsupported_optional_classes": optional,
                    "finite_transform_checks": finite_checks,
                    "deterministic_regeneration": reproducibility,
                    "fitting_rows": db.execute("SELECT count(*) FROM fitted").fetchone()[0],
                    "local_validation_rows": db.execute(
                        "SELECT count(*) FROM local WHERE role='local_validation'").fetchone()[0]}
        finally:
            db.close()

    def _regenerate(self, *, splits, assignments, local_splits, client_manifests,
                    provenance, selected, transformer, feature_dictionary,
                    feature_contract, label_mappings, fit_rows, directory, config,
                    fold, scenario):
        root = directory / "_work/reproduction"
        assignment_dir = root / "assignment"
        local_dir = root / "local"
        preprocessing_dir = root / "preprocessing"
        assignment_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        assign(splits, fold, scenario, config["scenarios"][scenario], assignment_dir,
               config["seed"], config["clients"], config["max_partition_attempts"],
               config["minimum_client_observations"], config["minimum_groups_per_client"],
               False, policy=support_policy(config), limits=config.get("near_iid_limits"))
        split_local(assignment_dir / "assignments.csv", local_dir, config["seed"],
                    config["local_validation_fraction"],
                    config["minimum_training_observations"],
                    False, policy=support_policy(config), attempts=config["max_partition_attempts"])
        fit(local_dir / "local_splits.csv", provenance, selected, preprocessing_dir,
            config["preprocessing"])
        comparisons = {
            "assignments": (assignments, assignment_dir / "assignments.csv"),
            "local_splits": (local_splits, local_dir / "local_splits.csv"),
            "client_manifests": (client_manifests, local_dir / "client_manifests.json"),
            "fit_rows": (fit_rows, preprocessing_dir / "fit_rows.csv"),
            "transformer": (transformer, preprocessing_dir / "transformer.json"),
            "feature_dictionary": (feature_dictionary,
                                   preprocessing_dir / "feature_dictionary.csv"),
            "feature_contract": (feature_contract, preprocessing_dir / "feature_contract.json"),
            "label_mappings": (label_mappings, preprocessing_dir / "label_mappings.json"),
        }
        return {name: sha256(actual) == sha256(regenerated)
                for name, (actual, regenerated) in comparisons.items()}


def validate(**arguments) -> dict:
    return PretrainingGate().run(**arguments)


def validate_support_table(path, report):
    expected = {(client, label): {
        "client_id": client, "label": label,
        "assigned": str(value["assigned_class_counts"][label]),
        "assigned_groups": str(value["class_group_counts"]["assigned"][label]),
        "training": str(value["training_class_counts"][label]),
        "training_groups": str(value["class_group_counts"]["training"][label]),
        "validation": str(value["validation_class_counts"][label]),
        "validation_groups": str(value["class_group_counts"]["validation"][label]),
    } for client, value in report["clients"].items() for label in sorted(LABELS)}
    seen = set()
    for row in rows(path):
        key = row["client_id"], row["label"]
        if key in seen or expected.get(key) != row:
            raise ValueError("Class support CSV differs from independently checked local rows")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Class support CSV coverage is incomplete")

"""Training-only fitted preprocessing and immutable feature contracts."""

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from edgefl.config import canonical_json
from edgefl.contracts.phase_c import MANIFEST_FIELDS
from edgefl.data.csv_reader import header, records
from edgefl.data.schema import LABELS, NUMERIC, canonical
from edgefl.client_data.policies import MANDATORY_EXCLUSIONS, validate_feature_policy, feature_policy_identity
from edgefl.client_data.storage import database, read_json, rows, sha256, write_json, writer


MISSING = "<MISSING>"
OTHER = "<OTHER>"


def _numeric(field: str, raw: str) -> float | None:
    value = canonical(field, raw)
    if raw.strip() == "":
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    result = float(number)
    return result if math.isfinite(result) else None


@dataclass
class RunningStat:
    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, value: float | None) -> None:
        if value is None:
            self.missing += 1
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0


class PreprocessorTrainer:
    """Accumulates statistics without loading selected observations into memory."""

    def __init__(self, names: tuple[str, ...], exclusions: set[str], strict: set[str],
                 max_categories: int, database_path: Path):
        unknown = (exclusions | strict) - set(names) - MANDATORY_EXCLUSIONS
        if unknown:
            raise ValueError("Configured preprocessing fields absent from source: " +
                             ", ".join(sorted(unknown)))
        self.names = names
        self.exclusions = exclusions
        self.strict = strict
        self.max_categories = max_categories
        self.numeric = tuple(name for name in names if name not in exclusions and name in NUMERIC)
        self.categorical = tuple(name for name in names if name not in exclusions and name not in NUMERIC)
        self.stats = {name: RunningStat() for name in self.numeric}
        self.labels = Counter()
        self.observations = 0
        self.db = database(database_path)
        self.db.execute("CREATE TABLE categories(field TEXT,value TEXT,n INTEGER,PRIMARY KEY(field,value)) WITHOUT ROWID")

    def observe(self, values: tuple[str, ...], label: str) -> None:
        fields = dict(zip(self.names, values, strict=True))
        for name in self.numeric:
            self.stats[name].observe(_numeric(name, fields[name]))
        for name in self.categorical:
            value = canonical(name, fields[name]) if fields[name].strip() else MISSING
            self.db.execute("""INSERT INTO categories VALUES (?,?,1)
                ON CONFLICT(field,value) DO UPDATE SET n=n+1""", (name, value))
        self.labels[label] += 1
        self.observations += 1
        if self.observations % 10000 == 0:
            self.db.commit()

    def _variant(self, name: str, extra_exclusions: set[str]) -> dict:
        excluded = self.exclusions | extra_exclusions
        numeric, categorical, outputs, removed = {}, {}, [], []
        for field in self.names:
            if field in excluded:
                continue
            if field in self.stats:
                stat = self.stats[field]
                if stat.count == 0 or stat.variance <= 0:
                    removed.append({"field": field,
                                    "reason": "all_missing" if stat.count == 0 else "constant"})
                    continue
                scale = math.sqrt(stat.variance)
                numeric[field] = {"imputation": "mean", "impute": stat.mean,
                                  "mean": stat.mean, "scale": scale,
                                  "observed": stat.count, "missing": stat.missing,
                                  "minimum": stat.minimum, "maximum": stat.maximum}
                outputs.append({"name": field, "source": field, "kind": "numeric",
                                "detail": "mean_imputed_standard_scaled"})
            else:
                cardinality = self.db.execute(
                    "SELECT count(*) FROM categories WHERE field=?", (field,)).fetchone()[0]
                if cardinality <= 1:
                    removed.append({"field": field, "reason": "constant"})
                    continue
                selected = [row[0] for row in self.db.execute(
                    "SELECT value FROM categories WHERE field=? ORDER BY n DESC,value LIMIT ?",
                    (field, self.max_categories))]
                has_other = cardinality > len(selected)
                categorical[field] = {"categories": selected, "other": has_other,
                                      "cardinality": cardinality,
                                      "unknown_handling": OTHER if has_other else "all_zero"}
                for value in selected:
                    outputs.append({"name": f"{field}=={value}", "source": field,
                                    "kind": "categorical", "detail": value})
                if has_other:
                    outputs.append({"name": f"{field}=={OTHER}", "source": field,
                                    "kind": "categorical", "detail": OTHER})
        if not outputs:
            raise ValueError(f"Preprocessing variant {name} has no nonconstant output features")
        return {"name": name, "excluded_fields": sorted(excluded), "numeric": numeric,
                "categorical": categorical, "removed_constant_fields": removed,
                "output_features": outputs}

    def finish(self) -> dict:
        self.db.commit()
        variants = {"base": self._variant("base", set()),
                    "strict": self._variant("strict", self.strict)}
        result = {"schema_version": "phase-c.transformer.v2", "fitting_observations": self.observations,
                  "input_fields": list(self.names), "variants": variants,
                  "label_support": dict(sorted(self.labels.items())),
                  "numeric_imputation": "mean", "scaling": "standard",
                  "categorical_encoding": "bounded_one_hot"}
        return result

    def close(self) -> None:
        self.db.close()


class FrozenPreprocessor:
    """Pure transformation object reconstructed from the serialized JSON contract."""

    def __init__(self, specification: dict):
        if specification.get("schema_version") != "phase-c.transformer.v2":
            raise ValueError("Unsupported transformer schema")
        self.specification = specification
        self.names = tuple(specification["input_fields"])

    @classmethod
    def load(cls, path: Path) -> "FrozenPreprocessor":
        return cls(read_json(path))

    def transform(self, names: tuple[str, ...], values: tuple[str, ...],
                  variant: str = "base") -> tuple[float, ...]:
        if names != self.names:
            raise ValueError("Input feature ordering differs from fitted transformer")
        fields = dict(zip(names, values, strict=True))
        spec = self.specification["variants"][variant]
        output = []
        for descriptor in spec["output_features"]:
            field = descriptor["source"]
            if descriptor["kind"] == "numeric":
                rule = spec["numeric"][field]
                value = _numeric(field, fields[field])
                value = rule["impute"] if value is None else value
                output.append((value - rule["mean"]) / rule["scale"])
            else:
                rule = spec["categorical"][field]
                value = canonical(field, fields[field]) if fields[field].strip() else MISSING
                detail = descriptor["detail"]
                if detail == OTHER:
                    output.append(float(value not in rule["categories"]))
                else:
                    output.append(float(value == detail))
        if len(output) != len(spec["output_features"]) or not all(map(math.isfinite, output)):
            raise ValueError("Transformer produced a non-finite or incompatible feature vector")
        return tuple(output)


def _membership(local_splits: Path, provenance: Path, db_path: Path):
    db = database(db_path)
    db.executescript("""
    CREATE TABLE membership(oid TEXT PRIMARY KEY,client TEXT,role TEXT,label TEXT,record INTEGER);
    CREATE UNIQUE INDEX membership_record ON membership(record);
    """)
    for row in rows(local_splits):
        db.execute("INSERT INTO membership(oid,client,role,label) VALUES (?,?,?,?)",
                   (row["observation_id"], row["client_id"], row["partition"], row["label"]))
    db.commit()
    for row in rows(provenance):
        db.execute("UPDATE membership SET record=? WHERE oid=?",
                   (int(row["record"]), row["observation_id"]))
    db.commit()
    missing = db.execute("SELECT count(*) FROM membership WHERE record IS NULL").fetchone()[0]
    if missing:
        db.close()
        raise ValueError(f"Local manifest contains {missing} observations absent from provenance")
    return db


def _write_dictionary(path: Path, transformer: dict) -> None:
    stream, output = writer(path, MANIFEST_FIELDS["feature_dictionary"])
    with stream:
        for variant, specification in sorted(transformer["variants"].items()):
            for index, descriptor in enumerate(specification["output_features"]):
                output.writerow({"variant": variant, "index": index,
                                 "output_feature": descriptor["name"],
                                 "source_feature": descriptor["source"],
                                 "kind": descriptor["kind"], "detail": descriptor["detail"]})


def verify_finite(selected: Path, membership_db, transformer: FrozenPreprocessor,
                  roles: set[str] | None = None) -> dict:
    names = header(selected)
    counts = Counter()
    for record in records(selected):
        member = membership_db.execute("SELECT role FROM membership WHERE record=?",
                                       (record.number,)).fetchone()
        if not member or roles is not None and member[0] not in roles:
            continue
        for variant in sorted(transformer.specification["variants"]):
            transformer.transform(names, record.values, variant)
            counts[variant] += 1
    return dict(counts)


def fit(local_splits: Path, provenance: Path, selected: Path, directory: Path,
        preprocessing: dict) -> dict:
    validate_feature_policy(preprocessing)
    names = header(selected)
    exclusions = set(preprocessing["identifier_payload_exclusions"])
    strict = set(preprocessing["strict_shortcut_exclusions"])
    membership = _membership(local_splits, provenance, directory / "_work/membership.sqlite")
    trainer = PreprocessorTrainer(names, exclusions, strict,
                                  preprocessing["max_categories"],
                                  directory / "_work/categories.sqlite")
    fit_stream, fit_rows = writer(directory / "fit_rows.csv", MANIFEST_FIELDS["fit_rows"])
    try:
        with fit_stream:
            for record in records(selected):
                member = membership.execute(
                    "SELECT oid,client,role,label FROM membership WHERE record=?",
                    (record.number,)).fetchone()
                if not member or member[2] != "local_train":
                    continue
                oid, client, _, label = member
                if label not in LABELS:
                    raise ValueError(f"Unknown fitting label: {label}")
                trainer.observe(record.values, label)
                fit_rows.writerow({"client_id": client, "observation_id": oid,
                                   "record": record.number, "label": label})
        if not trainer.observations:
            raise ValueError("No local-training observations available for preprocessing")
        transformer = trainer.finish()
        transformer["feature_policy"] = feature_policy_identity(preprocessing)
    finally:
        trainer.close()
    labels = sorted(transformer["label_support"])
    label_mappings = {"schema_version": "phase-c.labels.v2", "binary": {"Normal": 0, "Attack": 1},
                      "multiclass": {label: index for index, label in enumerate(labels)},
                      "observed_labels": labels}
    write_json(directory / "transformer.json", transformer)
    write_json(directory / "label_mappings.json", label_mappings)
    _write_dictionary(directory / "feature_dictionary.csv", transformer)
    frozen = FrozenPreprocessor(transformer)
    finite = verify_finite(selected, membership, frozen)
    feature_orders = {name: [item["name"] for item in spec["output_features"]]
                      for name, spec in transformer["variants"].items()}
    contract = {
        "schema_version": "phase-c.features.v2",
        "transformer_sha256": sha256(directory / "transformer.json"),
        "fit_rows_sha256": sha256(directory / "fit_rows.csv"),
        "label_mappings_sha256": sha256(directory / "label_mappings.json"),
        "input_fields": list(names), "variants": {},
        "feature_policy": feature_policy_identity(preprocessing),
    }
    for name, order in feature_orders.items():
        contract["variants"][name] = {
            "output_features": len(order),
            "feature_order_sha256": hashlib.sha256(canonical_json(order).encode()).hexdigest(),
        }
    write_json(directory / "feature_contract.json", contract)
    membership.close()
    shortcut_inspection = {}
    for field in sorted(strict):
        if field in transformer["variants"]["base"]["numeric"]:
            rule = transformer["variants"]["base"]["numeric"][field]
            shortcut_inspection[field] = {"kind": "numeric", "minimum": rule["minimum"],
                                          "maximum": rule["maximum"]}
        elif field in transformer["variants"]["base"]["categorical"]:
            rule = transformer["variants"]["base"]["categorical"][field]
            shortcut_inspection[field] = {"kind": "categorical",
                                          "cardinality": rule["cardinality"]}
        else:
            shortcut_inspection[field] = {"kind": "constant_or_absent_from_base_output"}
    return {"fitting_observations": transformer["fitting_observations"],
            "label_support": transformer["label_support"], "finite_transform_checks": finite,
            "features": {name: value["output_features"]
                         for name, value in contract["variants"].items()},
            "fit_rows_sha256": contract["fit_rows_sha256"],
            "shortcut_inspection": shortcut_inspection,
            "preprocessing_shared_across_methods": True,
            "eligible_for_training": False}

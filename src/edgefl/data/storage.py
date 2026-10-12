"""Verified artifact IO, deterministic CSV exports, and bounded SQLite workspaces."""

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from edgefl.config import canonical_json, workspace_path, _unique_keys, _reject_constant
from edgefl.contracts.phase_b import VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys,
                      parse_constant=_reject_constant)


def ref(root: Path, path: Path, role: str) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Artifact outside workspace or absent: {path}")
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "sha256": sha256(resolved),
            "role": role, "schema_version": VERSION}


def checked(root: Path, item: dict, role: str | None = None) -> Path:
    if set(item) != {"path", "sha256", "role", "schema_version"}:
        raise ValueError("Invalid artifact reference fields")
    if item["schema_version"] != VERSION or (role and item["role"] != role):
        raise ValueError("Artifact schema/role mismatch")
    path = workspace_path(root, item["path"])
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {item['path']}")
    return path


def writer(path: Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w", newline="", encoding="utf-8")
    output = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    output.writeheader()
    return stream, output


def rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    return db


def implementation_hash(stage: str | None = None) -> str:
    package = Path(__file__).resolve().parents[1]
    modules = {
        "audit":("audit", "inventory"),
        "provenance":("provenance", "inventory"),
        "verify-captures":("pcap", "verification", "inventory"),
        "group":("grouping",),
        "global-split":("splitting",),
        "trusted-panel":("trusted_panel",),
        "validate-phase-b":("validation", "origin_validation", "inventory", "splitting"),
    }
    pipelines = {"audit":"audit","provenance":"provenance","verify-captures":"captures",
                 "group":"group","global-split":"split","trusted-panel":"panel","validate-phase-b":"validate"}
    paths = [package/"data"/(name+".py") for name in ("config","storage","schema","csv_reader")]
    paths += [package/"contracts/phase_b.py",package/"schemas/phase_b.schema.json",
              package/"pipelines/phase_b_common.py",package/"reproducibility.py"]
    if stage in modules:
        paths += [package/"data"/(name+".py") for name in modules[stage]]
        paths += [package/"pipelines"/("phase_b_"+pipelines[stage]+".py")]
    else:
        paths += list(package.joinpath("data").glob("*.py"))
    return hashlib.sha256(canonical_json([(p.relative_to(package).as_posix(), sha256(p))
                                        for p in sorted(set(paths))]).encode()).hexdigest()


def load_completion(root: Path, path: Path, stage: str, config_hash: str, seen=None) -> dict:
    resolved = workspace_path(root, path.relative_to(root).as_posix())
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError("Cyclic stage lineage")
    seen.add(resolved)
    metadata = read_json(resolved)
    expected = {"schema_version","status","stage","dataset","configuration_sha256","implementation_sha256","inputs","options","artifacts","eligible_for_training"}
    if set(metadata) != expected or metadata.get("eligible_for_training") is not False:
        raise ValueError("Invalid completion contract")
    if (metadata.get("schema_version") != VERSION or metadata.get("stage") != stage
            or metadata.get("status") != "complete"):
        raise ValueError("Incomplete stage or stage/schema mismatch")
    if metadata.get("configuration_sha256") != config_hash:
        raise ValueError("Stale Phase B configuration")
    if metadata.get("implementation_sha256") != implementation_hash(stage):
        raise ValueError("Stale Phase B implementation")
    for name, item in metadata["artifacts"].items():
        checked(root, item, name)
    for item in metadata["inputs"].values():
        upstream = checked(root, item)
        if item["role"] == "foundation":
            load_foundation(root, upstream)
            continue
        if item["role"] != "completion":
            raise ValueError("Unknown upstream role")
        upstream_meta = read_json(upstream)
        load_completion(root, upstream, upstream_meta["stage"], config_hash, seen.copy())
    return metadata



def load_foundation(root: Path, path: Path) -> dict:
    value = read_json(path)
    if value.get("schema_version") != "1.0" or value.get("status") != "foundation_ready":
        raise ValueError("Phase B requires a completed Phase A foundation")
    if value.get("eligible_for_training") is not False:
        raise ValueError("Invalid foundation training eligibility")
    required = {"configuration","environment","seeds","source_manifest"}
    if not required <= set(value.get("artifacts",{})):
        raise ValueError("Foundation artifacts incomplete")
    for item in value["artifacts"].values():
        target = workspace_path(root,(path.parent/item["path"]).relative_to(root).as_posix())
        if not target.is_relative_to(path.parent) or sha256(target) != item["sha256"]:
            raise ValueError("Foundation artifact path/hash mismatch")
    return value

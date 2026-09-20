"""Create auditable Phase A records, without reading or hashing research datasets."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from edgefl.config import Configuration, canonical_json, workspace_path
from edgefl.reproducibility import seed_manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_snapshot(root: Path) -> dict[str, object]:
    """Hash only implementation/configuration/protocol inputs, never archive/data/runs."""
    paths = [
        path for path in root.iterdir()
        if path.is_file() and (path.suffix in (".md", ".toml") or path.name in (".gitignore", ".gitattributes", "requirements.lock"))
    ]
    for directory in ("src", "scripts", "tests", "configs", "docs"):
        folder = root / directory
        if folder.exists():
            paths.extend(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix in (".py", ".json", ".md", ".toml")
                and "__pycache__" not in path.parts
            )
    records = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source snapshot cannot follow external symlink: {path}")
        records.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256(path.read_bytes())})
    return {"files": records, "sha256": _sha256(canonical_json(records).encode("utf-8"))}


def _git_state(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip()
    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if top != root.resolve():
            return {"status": "not_a_workspace_repository", "commit": None, "dirty": None}
        return {
            "status": "available", "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable_or_no_commit", "commit": None, "dirty": None}


def _write_json(path: Path, value: object) -> dict[str, str]:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return {"path": path.name, "sha256": _sha256(text.encode("utf-8"))}


def initialize_foundation_run(config: Configuration, master_seed: int) -> Path:
    values = config.values
    if master_seed not in values["randomness"]["matched_seeds"] or type(master_seed) is not int:
        raise ValueError("Run seed must be listed in randomness.matched_seeds")
    protocol = workspace_path(config.workspace, values["protocol_document"])
    now = datetime.now(timezone.utc)
    run_id = f"foundation-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
    # All metadata is collected before creating the run directory.
    snapshot = source_snapshot(config.workspace)
    environment = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
        )),
    }
    revision = _git_state(config.workspace)
    seeds = seed_manifest(master_seed, values["randomness"]["global_split_seed"])
    metadata = {
        "schema_version": "1.0", "run_id": run_id, "kind": "foundation",
        "created_at_utc": now.isoformat(), "status": "foundation_ready",
        "eligible_for_training": False,
        "protocol_id": values["protocol_id"],
        "protocol_sha256": _sha256(protocol.read_bytes()),
        "configuration_sha256": config.sha256,
        "source_snapshot_sha256": snapshot["sha256"],
        "code_revision": revision,
        "master_seed": master_seed,
        "dataset_registration": {
            "status": "pending_phase_b",
            "primary": {"path": values["dataset"]["primary"], "sha256": None},
            "smoke": {"path": values["dataset"]["smoke"], "sha256": None},
            "manifest": None,
        },
        "completed_roadmap_steps": [1, 2, 3],
        "pretraining_gate": "not_run",
    }
    destination = workspace_path(config.workspace, values["paths"]["runs"] + "/" + run_id)
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {
        "configuration": _write_json(destination / "config.json", values),
        "environment": _write_json(destination / "environment.json", environment),
        "seeds": _write_json(destination / "seeds.json", seeds),
        "source_manifest": _write_json(destination / "source_manifest.json", snapshot),
    }
    event = {
        "timestamp_utc": now.isoformat(), "run_id": run_id, "pipeline": "foundation",
        "level": "INFO", "event": "foundation_prepared",
        "message": "Phase A only; dataset registration and experiment execution are pending.",
    }
    with (destination / "events.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(event) + "\n")
    metadata["artifacts"] = outputs
    # A run is complete only when run.json exists; interrupted directories remain auditable.
    _write_json(destination / "run.json", metadata)
    return destination

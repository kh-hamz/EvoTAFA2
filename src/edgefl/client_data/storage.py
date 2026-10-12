"""Immutable Phase C artifacts with verified Phase B and Phase C lineage."""

import hashlib
import platform
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from edgefl.config import canonical_json, workspace_path
from edgefl.contracts.phase_c import VERSION, TrainingReadyBundle
from edgefl.data.integrity import VerificationContext, assert_registered_sources
from edgefl.data.storage import (database, read_json, rows, sha256, write_json, writer)


def artifact_ref(root: Path, path: Path, role: str, version: str = VERSION) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Artifact outside workspace or absent: {path}")
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "sha256": sha256(resolved),
            "role": role, "schema_version": version}


def checked(root: Path, item: dict, role: str | None = None,
            versions: tuple[str, ...] = (VERSION,)) -> Path:
    if set(item) != {"path", "sha256", "role", "schema_version"}:
        raise ValueError("Invalid artifact reference fields")
    if item["schema_version"] not in versions or (role and item["role"] != role):
        raise ValueError("Artifact schema/role mismatch")
    path = workspace_path(root, item["path"])
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {item['path']}")
    return path


def _completion_ref(root: Path, path: Path) -> dict:
    metadata = read_json(path)
    version = metadata.get("schema_version")
    if version not in (VERSION, "phase-b.v2"):
        raise ValueError("Unsupported upstream completion schema")
    return artifact_ref(root, path, "completion", version)


def implementation_hash(stage: str) -> str:
    package = Path(__file__).resolve().parents[1]
    modules = {
        "assign-clients": ("assignment",),
        "local-split": ("local_split",),
        "fit-preprocessing": ("preprocessing",),
        "validate-phase-c": ("validation", "assignment", "local_split", "preprocessing"),
    }
    pipelines = {
        "assign-clients": "assign",
        "local-split": "local_split",
        "fit-preprocessing": "preprocessing",
        "validate-phase-c": "validate",
    }
    paths = [package / "client_data" / name for name in
             ("config.py", "storage.py", "policies.py", "experiments.py")]
    paths += [package / "contracts/phase_c.py", package / "schemas/phase_c.schema.json",
              package / "pipelines/phase_c_common.py", package / "reproducibility.py"]
    paths += [package / "client_data" / f"{name}.py" for name in modules.get(stage, ())]
    if stage in pipelines:
        paths.append(package / "pipelines" / f"phase_c_{pipelines[stage]}.py")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Phase C implementation incomplete: " + ", ".join(str(p) for p in missing))
    values = [(path.relative_to(package).as_posix(), sha256(path)) for path in sorted(set(paths))]
    return hashlib.sha256(canonical_json(values).encode()).hexdigest()


def load_completion(config, path: Path, stage: str, dataset: str | None = None,
                    seen: set[Path] | None = None, context=None) -> dict:
    resolved = workspace_path(config.workspace, path.relative_to(config.workspace).as_posix())
    visited = set() if seen is None else seen
    if resolved in visited:
        raise ValueError("Cyclic Phase C stage lineage")
    context = context if context is not None else VerificationContext()
    cache_key = (resolved, stage, config.sha256, dataset)
    if cache_key in context.completions:
        return context.completions[cache_key]
    visited.add(resolved)
    metadata = read_json(resolved)
    expected = {"schema_version", "status", "stage", "dataset", "configuration_sha256",
                "implementation_sha256", "inputs", "options", "artifacts",
                "eligible_for_training"}
    if set(metadata) != expected or metadata.get("schema_version") != VERSION:
        raise ValueError("Invalid Phase C completion contract")
    if metadata.get("status") != "complete" or metadata.get("stage") != stage:
        raise ValueError("Incomplete Phase C stage or stage mismatch")
    if dataset and metadata.get("dataset") != dataset:
        raise ValueError("Phase C selected dataset mismatch")
    if metadata.get("configuration_sha256") != config.sha256:
        raise ValueError("Stale Phase C configuration")
    if metadata.get("implementation_sha256") != implementation_hash(stage):
        raise ValueError("Stale Phase C implementation")
    if type(metadata.get("eligible_for_training")) is not bool:
        raise ValueError("Invalid Phase C eligibility state")
    if stage != "validate-phase-c" and metadata["eligible_for_training"]:
        raise ValueError("Only the pretraining gate can enable training")
    dependencies = {
        "assign-clients": {"phase_b_validation": "validate-phase-b", "split": "global-split"},
        "local-split": {"assignments": "assign-clients"},
        "fit-preprocessing": {"local_split": "local-split", "provenance": "provenance"},
        "validate-phase-c": {"phase_b_validation": "validate-phase-b", "split": "global-split",
                             "assignments": "assign-clients", "local_split": "local-split",
                             "preprocessing": "fit-preprocessing", "provenance": "provenance"},
    }
    expected_dependencies = dependencies.get(stage)
    if expected_dependencies is None or set(metadata["inputs"]) != set(expected_dependencies):
        raise ValueError("Invalid Phase C stage dependencies")
    for name, item in metadata["artifacts"].items():
        checked(config.workspace, item, name)
    from edgefl.client_data.config import phase_b
    from edgefl.data.storage import load_completion as load_phase_b_completion
    phase_b_config = phase_b(config)
    upstream_values = {}
    for key, item in metadata["inputs"].items():
        upstream = checked(config.workspace, item, "completion", (VERSION, "phase-b.v2"))
        upstream_meta = read_json(upstream)
        if upstream_meta["stage"] != expected_dependencies[key]:
            raise ValueError("Phase C upstream stage mismatch")
        if item["schema_version"] == "phase-b.v2":
            value = load_phase_b_completion(config.workspace, upstream, expected_dependencies[key],
                                           phase_b_config.sha256, context=context)
            if value["dataset"] != metadata["dataset"]:
                raise ValueError("Phase C upstream dataset mismatch")
            assert_registered_sources(phase_b_config, upstream, context)
        else:
            value = load_completion(config, upstream, expected_dependencies[key], metadata["dataset"],
                                    visited.copy(), context)
            if value["options"] != metadata["options"]:
                raise ValueError("Phase C task/fold/scenario mismatch")
        upstream_values[key] = value
    if stage == "assign-clients":
        from edgefl.client_data.experiments import verify_experiment
        experiment = read_json(artifact(config, metadata, "experiment_contract"))
        if any(experiment.get(key) != metadata["options"].get(key) for key in ("fold", "task")):
            raise ValueError("Assignment experiment task/fold mismatch")
        verify_experiment(config, experiment, upstream_values["split"], upstream_values["phase_b_validation"])
    if stage == "validate-phase-c":
        _verify_training_export(config, metadata, upstream_values)
    context.completions[cache_key] = metadata
    return metadata


def load_phase_b_completion(config, path: Path, expected: str,
                            dataset: str | None = None, context=None) -> dict:
    from edgefl.client_data.config import phase_b
    from edgefl.data.storage import load_completion as load_b
    context = context if context is not None else VerificationContext()
    phase_b_config = phase_b(config)
    metadata = load_b(config.workspace, path, expected, phase_b_config.sha256, context=context)
    if dataset and metadata["dataset"] != dataset:
        raise ValueError("Phase B selected dataset mismatch")
    assert_registered_sources(phase_b_config, path, context)
    return metadata


def _verify_training_export(config, metadata, upstream):
    required = {"pretraining_report", "training_inputs"}
    if not required <= set(metadata["artifacts"]):
        raise ValueError("Missing training-readiness artifacts")
    report = read_json(artifact(config, metadata, "pretraining_report"))
    exported = read_json(artifact(config, metadata, "training_inputs"))
    eligible = report.get("status") == "PASS"
    if eligible and report.get("errors"):
        raise ValueError("PASS report contains validation errors")
    if (metadata["eligible_for_training"] != eligible or report.get("eligible_for_training") is not eligible
            or exported.get("eligible_for_training") is not eligible):
        raise ValueError("Training eligibility disagrees with the pretraining report")
    options = metadata["options"]
    if exported.get("schema_version") != "phase-c.training-inputs.v2":
        raise ValueError("Unsupported training export schema")
    for key in ("fold", "scenario", "task"):
        if exported.get(key) != options.get(key) or report.get(key) != options.get(key):
            raise ValueError("Training export task/fold/scenario mismatch")
    if exported.get("dataset") != metadata["dataset"]:
        raise ValueError("Training export dataset mismatch")
    expected = {**{key: upstream["local_split"]["artifacts"][key]
                   for key in ("local_splits", "client_manifests", "class_support")},
                **{key: upstream["preprocessing"]["artifacts"][key]
                   for key in ("transformer", "feature_contract", "label_mappings")},
                "phase_b_validation": metadata["inputs"]["phase_b_validation"]}
    for key, reference in expected.items():
        if exported.get(key) != reference:
            raise ValueError(f"Training export lineage mismatch: {key}")
    experiment = read_json(artifact(config, upstream["assignments"], "experiment_contract"))
    if exported.get("experiment_contract") != upstream["assignments"]["artifacts"]["experiment_contract"]:
        raise ValueError("Training export experiment mismatch")
    if report.get("experiment") != experiment:
        raise ValueError("Pretraining report experiment mismatch")
    from edgefl.client_data.experiments import verify_experiment
    verify_experiment(config, experiment, upstream["split"], upstream["phase_b_validation"])
    if exported.get("trusted_panel") != upstream["assignments"]["artifacts"]["trusted_panel"]:
        raise ValueError("Training export trusted panel mismatch")
    if exported.get("global_splits") != upstream["assignments"]["artifacts"]["global_splits"]:
        raise ValueError("Training export global split mismatch")


def load_training_ready(config, path: Path, dataset: str, *, task=None, fold=None, scenario=None):
    metadata = load_completion(config, path, "validate-phase-c", dataset)
    if not metadata["eligible_for_training"]:
        raise ValueError("A fresh Phase C PASS is required for training")
    options = metadata["options"]
    for key, requested in (("task", task), ("fold", fold), ("scenario", scenario)):
        if requested is not None and requested != options[key]:
            raise ValueError(f"Training-ready {key} mismatch")
    exported = read_json(artifact(config, metadata, "training_inputs"))
    references = tuple((key, value["path"], value["sha256"]) for key, value in sorted(exported.items())
                       if isinstance(value, dict) and "path" in value)
    return TrainingReadyBundle(dataset, options["task"], options["fold"], options["scenario"], references)


def artifact(config, metadata: dict, name: str) -> Path:
    item = metadata["artifacts"][name]
    return checked(config.workspace, item, name, (VERSION, "phase-b.v2"))


def require_link(config, metadata: dict, key: str, path: Path) -> None:
    if metadata["inputs"].get(key) != _completion_ref(config.workspace, path):
        raise ValueError(f"Upstream lineage mismatch for {key}")


class Stage:
    """One immutable, restart-safe Phase C stage attempt."""

    def __init__(self, config, name: str, dataset: str, inputs: dict[str, Path],
                 options: dict | None = None, context=None):
        self.config, self.name, self.dataset = config, name, dataset
        self.inputs = {key: _completion_ref(config.workspace, value)
                       for key, value in inputs.items()}
        self.options = options or {}
        self.fingerprint = implementation_hash(name)
        identity = hashlib.sha256(canonical_json(
            [config.sha256, self.fingerprint, name, dataset, self.inputs, self.options]
        ).encode()).hexdigest()
        base = workspace_path(config.workspace, config.values["output"]) / config.sha256[:16]
        base = base / dataset / f"{name}-{identity[:16]}"
        self.reused: Path | None = None
        if base.exists():
            for completion in sorted(base.glob("*/completion.json")):
                metadata = load_completion(config, completion, name, dataset, context=context)
                if metadata["inputs"] == self.inputs and metadata["options"] == self.options:
                    self.reused, self.directory = completion, completion.parent
                    return
        self.directory = base / ("attempt-" + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.event(event="started")
        write_json(self.directory / "configuration.json", config.values)
        write_json(self.directory / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        })

    def event(self, **value) -> None:
        entry = {"stage": self.name,
                 "elapsed_seconds": round(time.monotonic() - self.started, 3), **value}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(entry) + "\n")
        print(canonical_json(entry), flush=True)

    def finish(self, summary: dict, artifacts: dict[str, Path],
               eligible_for_training: bool = False) -> Path:
        write_json(self.directory / "summary.json", summary)
        self.event(event="completed")
        paths = {**artifacts, "summary": self.directory / "summary.json",
                 "configuration": self.directory / "configuration.json",
                 "environment": self.directory / "environment.json",
                 "events": self.directory / "events.jsonl"}
        metadata = {
            "schema_version": VERSION, "status": "complete", "stage": self.name,
            "dataset": self.dataset, "configuration_sha256": self.config.sha256,
            "implementation_sha256": self.fingerprint, "inputs": self.inputs,
            "options": self.options,
            "artifacts": {key: artifact_ref(self.config.workspace, path, key)
                          for key, path in paths.items()},
            "eligible_for_training": eligible_for_training,
        }
        write_json(self.directory / "completion.json", metadata)
        return self.directory / "completion.json"


@contextmanager
def stage(config, name: str, dataset: str, inputs: dict[str, Path],
          options: dict | None = None, context=None):
    run = Stage(config, name, dataset, inputs, options, context)
    try:
        yield run
    except BaseException as exc:
        if not run.reused:
            run.event(event="failed", error=type(exc).__name__)
            write_json(run.directory / "failure.json", {
                "status": "incomplete", "error": str(exc),
                "exception": type(exc).__name__, "eligible_for_training": False,
            })
        raise


__all__ = [
    "Stage", "artifact", "artifact_ref", "checked", "database", "load_completion",
    "load_phase_b_completion", "load_training_ready", "read_json", "require_link", "rows", "sha256", "stage",
    "write_json", "writer",
]

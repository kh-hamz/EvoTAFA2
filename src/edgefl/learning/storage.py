"""Immutable Phase D stage attempts and recursively verified lineage."""

import hashlib
import platform
import uuid
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
from edgefl.config import canonical_json, workspace_path
from edgefl.data.storage import read_json, write_json, sha256
from edgefl.client_data.storage import load_training_ready
from edgefl.learning.config import phase_c
from edgefl.contracts.phase_d import VERSION, ExperimentIdentity


def reference(root, path):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact outside workspace or missing")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def checked(root, ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Invalid Phase D reference")
    path = workspace_path(root, ref["path"])
    if not path.is_file() or sha256(path) != ref["sha256"]:
        raise ValueError("Phase D artifact hash mismatch: " + ref["path"])
    return path


def implementation_hash():
    package = Path(__file__).parents[1]
    files = list((package / "learning").glob("*.py"))
    files += list((package / "pipelines").glob("phase_d_*.py"))
    files += [package / "contracts/phase_d.py", package / "schemas/phase_d.schema.json"]
    return hashlib.sha256(canonical_json([(p.relative_to(package).as_posix(), sha256(p))
                                        for p in sorted(files)]).encode()).hexdigest()


def verify_gate(config, gate, identity):
    return load_training_ready(phase_c(config), gate, identity["dataset"],
                               task=identity["task"], fold=identity["fold"], scenario=identity["scenario"])


def load_stage(config, path, expected=None, memo=None, active=None):
    memo = {} if memo is None else memo
    active = set() if active is None else active
    path = path.resolve()
    if path in active:
        raise ValueError("Cyclic Phase D lineage")
    if path in memo:
        result = memo[path]
        if expected and result["stage"] != expected:
            raise ValueError("Phase D stage mismatch")
        return result
    if not path.is_relative_to(config.workspace):
        raise ValueError("Completion outside workspace")
    active.add(path)
    value = read_json(path)
    required = {"schema_version", "status", "stage", "configuration_sha256", "implementation_sha256",
                "identity", "gate", "inputs", "options", "artifacts", "summary"}
    if set(value) != required or value["schema_version"] != VERSION or value["status"] != "complete":
        raise ValueError("Invalid or incomplete Phase D completion")
    if expected and value["stage"] != expected:
        raise ValueError("Phase D stage mismatch")
    required_inputs = {"prepare-learning-data": set(), "initialize-model": {"prepared"},
                       "train-centralized": {"prepared", "initialization"},
                       "run-federated": {"prepared", "initialization"},
                       "review-learning": {"run"}, "validate-phase-d": set()}
    if value["stage"] not in required_inputs:
        raise ValueError("Unknown Phase D stage")
    inputs = set(value["inputs"])
    required_set = required_inputs[value["stage"]]
    if not required_set <= inputs or any(not k.startswith("review_") for k in inputs - required_set):
        raise ValueError("Invalid Phase D dependencies")
    if value["stage"] in ("prepare-learning-data", "initialize-model", "train-centralized", "review-learning") and inputs != required_set:
        raise ValueError("Unexpected Phase D dependency")
    if value["configuration_sha256"] != config.sha256 or value["implementation_sha256"] != implementation_hash():
        raise ValueError("Stale Phase D configuration or implementation")
    if value["identity"]["variant"] != config.values["variant"]:
        raise ValueError("Feature variant mismatch")
    gate = checked(config.workspace, value["gate"])
    gate_key = ("gate", gate, canonical_json(value["identity"]))
    if gate_key not in memo:
        verify_gate(config, gate, value["identity"])
        memo[gate_key] = True
    for ref in value["artifacts"].values():
        checked(config.workspace, ref)
    for key, ref in value["inputs"].items():
        dependency_stage = {"prepared": "prepare-learning-data", "initialization": "initialize-model"}.get(key)
        if key.startswith("review_"):
            dependency_stage = "review-learning"
        upstream = load_stage(config, checked(config.workspace, ref), expected=dependency_stage, memo=memo, active=active)
        if key in ("prepared", "initialization", "run") and (upstream["identity"] != value["identity"] or upstream["gate"] != value["gate"]):
            raise ValueError("Phase D dependency identity mismatch")
    if "resume" in value["options"]:
        marker = read_json(checked(config.workspace, value["options"]["resume"]))
        checked(config.workspace, marker["checkpoint"])
    for name in ("final_checkpoint", "software_evidence"):
        if name in value["summary"]:
            checked(config.workspace, value["summary"][name])
    if value["summary"].get("best"):
        checked(config.workspace, value["summary"]["best"]["reference"])
    active.remove(path)
    memo[path] = value
    return value


def artifact(config, completion, name):
    return checked(config.workspace, completion["artifacts"][name])


class Stage:
    def __init__(self, config, name, identity, gate, inputs=None, options=None):
        self.config, self.name = config, name
        self.identity = asdict(identity) if isinstance(identity, ExperimentIdentity) else dict(identity)
        self.gate = reference(config.workspace, gate)
        self.inputs = {k: reference(config.workspace, p) for k, p in (inputs or {}).items()}
        self.options = options or {}
        self.directory = workspace_path(config.workspace, config.values["output"]) / (name + "-" + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True)
        write_json(self.directory / "configuration.json", config.values)
        write_json(self.directory / "environment.json", {"python": platform.python_version(), "platform": platform.platform()})

    def finish(self, summary):
        artifacts = {p.relative_to(self.directory).as_posix(): reference(self.config.workspace, p)
                     for p in sorted(self.directory.rglob("*")) if p.is_file()
                     and "_work" not in p.parts and not p.name.endswith(".partial")}
        value = {"schema_version": VERSION, "status": "complete", "stage": self.name,
                 "configuration_sha256": self.config.sha256, "implementation_sha256": implementation_hash(),
                 "identity": self.identity, "gate": self.gate, "inputs": self.inputs,
                 "options": self.options, "artifacts": artifacts, "summary": summary}
        path = self.directory / "completion.json"
        write_json(path, value)
        return path


@contextmanager
def stage(*args, **kwargs):
    run = Stage(*args, **kwargs)
    try:
        yield run
    except BaseException as exc:
        write_json(run.directory / "failure.json", {"status": "incomplete", "error": str(exc), "type": type(exc).__name__})
        raise

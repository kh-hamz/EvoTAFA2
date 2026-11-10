"""Immutable experiment attempts, recursively checked references, no ledger authority."""
from pathlib import Path
from contextlib import contextmanager
import uuid
from edgefl.config import workspace_path
from edgefl.data.storage import read_json, write_json, sha256
from edgefl.learning.storage import reference, checked
from edgefl.contracts.phase_g import VERSION, digest


def implementation_hash():
    package = Path(__file__).parents[1]
    root = package.parents[1]
    files = sorted([*package.rglob("*.py"), *package.joinpath("schemas").glob("*.json"),
                    root / "pyproject.toml", root / "requirements-phase-f.lock", root / "requirements-phase-d.lock"])
    return digest([(p.relative_to(root).as_posix(), sha256(p)) for p in files])


def regression_hash():
    root = Path(__file__).resolve().parents[3]
    files = sorted([*root.joinpath("tests").glob("*.py"), *root.joinpath("scripts").glob("check_phase_*.py")])
    return digest([(p.relative_to(root).as_posix(), sha256(p)) for p in files])


class Stage:
    def __init__(self, config, name, inputs, options=None, identity=None, gate=None, matrix_id="unbound", case_id="shared"):
        self.config, self.name, self.options = config, name, options or {}
        self.identity, self.gate = identity, gate
        self.inputs = {k: reference(config.workspace, p) for k, p in inputs.items()}
        base = workspace_path(config.workspace, config.values["output"])
        self.directory = base / matrix_id / "attempts" / case_id / (name + "-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fingerprint = implementation_hash()

    def finish(self, summary):
        destination = self.directory / "completion.json"
        if destination.exists():
            raise ValueError("Experiment attempt already completed")
        artifacts = {p.relative_to(self.directory).as_posix(): reference(self.config.workspace, p)
                     for p in sorted(self.directory.rglob("*")) if p.is_file() and "_work" not in p.parts
                     and not p.name.endswith(".partial")}
        write_json(destination, {"schema_version": VERSION, "stage": self.name, "status": "complete",
            "configuration_sha256": self.config.sha256, "implementation_sha256": self.fingerprint,
            "evidence_kind": self.config.values["evidence_kind"], "identity": self.identity, "gate": self.gate,
            "inputs": self.inputs, "options": self.options, "artifacts": artifacts, "summary": summary})
        return destination


@contextmanager
def stage(*args, **kwargs):
    attempt = Stage(*args, **kwargs)
    try:
        yield attempt
    except BaseException as exc:
        write_json(attempt.directory / "failure.json", {"status": "incomplete", "type": type(exc).__name__, "error": str(exc)})
        raise


def load_stage(config, path, expected=None, memo=None, active=None):
    memo, active = ({} if memo is None else memo), (set() if active is None else active)
    path = path.resolve()
    if not path.is_relative_to(config.workspace) or path in active:
        raise ValueError("Invalid/cyclic experiment lineage")
    if path in memo:
        result = memo[path]
        if expected and result["stage"] != expected:
            raise ValueError("Experiment stage mismatch")
        return result
    value = read_json(path)
    required = {"schema_version", "stage", "status", "configuration_sha256", "implementation_sha256", "evidence_kind",
                "identity", "gate", "inputs", "options", "artifacts", "summary"}
    if set(value) != required or value["schema_version"] != VERSION or value["status"] != "complete":
        raise ValueError("Invalid experiment completion")
    if "implementation" not in memo:
        memo["implementation"] = implementation_hash()
    if (value["configuration_sha256"] != config.sha256 or value["implementation_sha256"] != memo["implementation"]
            or value["evidence_kind"] != config.values["evidence_kind"]):
        raise ValueError("Stale experiment configuration, implementation or evidence kind")
    if expected and value["stage"] != expected:
        raise ValueError("Experiment stage mismatch")
    active.add(path)
    for ref in value["artifacts"].values():
        checked(config.workspace, ref)
    for name, ref in value["inputs"].items():
        parent = checked(config.workspace, ref)
        if parent.name == "completion.json" and read_json(parent).get("schema_version") == VERSION:
            load_stage(config, parent, memo=memo, active=active)
    if value["gate"]:
        checked(config.workspace, value["gate"])
    for name in ("final_checkpoint", "software_evidence"):
        if value["summary"].get(name):
            checked(config.workspace, value["summary"][name])
    if value["summary"].get("best"):
        checked(config.workspace, value["summary"]["best"]["reference"])
    if value["stage"] == "prepare-experiment":
        from edgefl.experiments.preparation import verify_binding
        verify_binding(config, value, memo)
    if value["stage"] == "compile-experiments":
        from edgefl.experiments.matrix import compile_matrix
        expected_matrix = compile_matrix(config.values)
        expected_matrix["definition_id"] = expected_matrix["matrix_id"]
        expected_matrix["matrix_id"] = digest({"definition": expected_matrix["definition_id"],
            "configuration": config.sha256, "implementation": memo["implementation"]})
        if read_json(checked(config.workspace, value["artifacts"]["cases.json"])) != expected_matrix:
            raise ValueError("Compiled matrix differs from the frozen specification")
    if value["stage"] == "validate-phase-g":
        software_evidence(checked(config.workspace, value["summary"]["software_evidence"]))
    if "round_index.json" in value["artifacts"]:
        index = read_json(checked(config.workspace, value["artifacts"]["round_index.json"]))
        expected_rounds = {str(i) for i in range(1, value["summary"]["completed_steps"] + 1)}
        if set(index["rounds"]) != expected_rounds:
            raise ValueError("Incomplete experiment round lineage")
        for refs in index["rounds"].values():
            for ref in refs.values():
                checked(config.workspace, ref)
    active.remove(path)
    memo[path] = value
    return value


def artifact(config, completion, name):
    return checked(config.workspace, completion["artifacts"][name])


def software_evidence(path):
    value = read_json(path)
    if (value.get("schema_version") != "phase-g.tests.v1" or value.get("successful") is not True
            or any(value.get(k) != 0 for k in ("failures", "errors", "skipped"))
            or type(value.get("tests")) is not int or value["tests"] <= 0
            or value.get("implementation_sha256") != implementation_hash()
            or value.get("regression_sha256") != regression_hash()):
        raise ValueError("Current complete A-G software evidence required")
    return value

"""Immutable F stages with recursively checked A-E and search lineage."""
import hashlib
from pathlib import Path
from contextlib import contextmanager
from edgefl.config import canonical_json
from edgefl.data.storage import read_json, write_json, sha256
from edgefl.learning import storage as d
from edgefl.trust import storage as e
from edgefl.trust.review import check_acceptance
from edgefl.contracts.phase_f import VERSION, STAGES

reference, checked, artifact = d.reference, d.checked, d.artifact

def implementation_hash():
    package = Path(__file__).parents[1]
    root = package.parents[1]
    files = list((package / "optimization").glob("*.py")) + list((package / "pipelines").glob("phase_f_*.py"))
    files += [package / p for p in ("contracts/phase_f.py", "schemas/phase_f.schema.json")]
    files += [root / "requirements-phase-f.lock", root / "requirements-phase-d.lock", root / "pyproject.toml"]
    return hashlib.sha256(canonical_json([e.implementation_hash(),
        [(p.relative_to(root).as_posix(), sha256(p)) for p in sorted(files)]]).encode()).hexdigest()

def regression_hash():
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "tests").glob("*.py")) + sorted((root / "scripts").glob("check_phase_*.py"))
    return hashlib.sha256(canonical_json([(p.relative_to(root).as_posix(), sha256(p)) for p in files]).encode()).hexdigest()

def load_stage(config, path, expected=None, memo=None, active=None):
    memo, active = ({} if memo is None else memo), (set() if active is None else active)
    path = path.resolve()
    if path in active:
        raise ValueError("Cyclic Phase F lineage")
    if path in memo:
        value = memo[path]
        if expected and value["stage"] != expected:
            raise ValueError("Phase F stage mismatch")
        return value
    if not path.is_relative_to(config.workspace):
        raise ValueError("Phase F stage outside workspace")
    value = read_json(path)
    keys = {"schema_version", "status", "stage", "configuration_sha256", "implementation_sha256",
            "identity", "gate", "inputs", "options", "artifacts", "summary"}
    if set(value) != keys or value["schema_version"] != VERSION or value["status"] != "complete" or value["stage"] not in STAGES:
        raise ValueError("Invalid Phase F completion")
    if expected and value["stage"] != expected:
        raise ValueError("Phase F stage mismatch")
    if value["configuration_sha256"] != config.sha256 or value["implementation_sha256"] != implementation_hash():
        raise ValueError("Stale Phase F configuration/implementation")
    if set(value["inputs"]) != STAGES[value["stage"]]:
        raise ValueError("Invalid Phase F dependencies")
    active.add(path)
    gate = checked(config.workspace, value["gate"])
    gate_key = ("f_gate", gate, canonical_json(value["identity"]))
    if gate_key not in memo:
        d.verify_gate(config.learning, gate, value["identity"])
        memo[gate_key] = True
    for ref in value["artifacts"].values():
        checked(config.workspace, ref)
    for name, ref in value["inputs"].items():
        upstream = checked(config.workspace, ref)
        if name == "phase_e_acceptance":
            parent = check_acceptance(config.trust, upstream, memo)
        elif name == "initialization":
            parent = d.load_stage(config.learning, upstream, "initialize-model", memo=memo)
        elif name == "attack_plan":
            parent = e.load_stage(config.trust, upstream, "plan-attacks", memo=memo)
        elif name == "run" and read_json(upstream).get("schema_version") == "phase-e.v1":
            if value["stage"] != "optimize-round":
                raise ValueError("Only standalone replay accepts E runs")
            parent = e.load_stage(config.trust, upstream, "run-trust-federated", memo=memo)
        else:
            target = {"prepared_f": "prepare-phase-f", "run": "run-evolution-federated",
                      "run_review": "review-phase-f", "profile_review": "review-phase-f"}.get(name)
            parent = load_stage(config, upstream, target, memo, active)
        if parent["identity"] != value["identity"] or parent["gate"] != value["gate"]:
            raise ValueError("Phase F dependency scope mismatch")
    for name in ("final_checkpoint", "software_evidence"):
        if name in value["summary"]:
            checked(config.workspace, value["summary"][name])
    if value["summary"].get("best"):
        checked(config.workspace, value["summary"]["best"]["reference"])
    if "round_index.json" in value["artifacts"]:
        index = read_json(artifact(config, value, "round_index.json"))
        expected_rounds = {str(r) for r in range(1, value["summary"]["completed_steps"] + 1)}
        if index["schema_version"] != "phase-f.round-index.v1" or set(index["rounds"]) != expected_rounds:
            raise ValueError("Incomplete Phase F round lineage")
        for refs in index["rounds"].values():
            if set(refs) != {"assessment", "simulator", "reputation", "search"}:
                raise ValueError("Invalid Phase F round index")
            for ref in refs.values():
                checked(config.workspace, ref)
    if "resume" in value["options"]:
        marker = read_json(checked(config.workspace, value["options"]["resume"]))
        checked(config.workspace, marker["checkpoint"])
    if value["stage"] == "validate-phase-f":
        from edgefl.optimization.review import software_evidence
        software_evidence(checked(config.workspace, value["summary"]["software_evidence"]))
    active.remove(path)
    memo[path] = value
    return value

class Stage(d.Stage):
    def finish(self, summary):
        path = self.directory / "completion.json"
        if path.exists():
            raise ValueError("Phase F attempt already completed")
        artifacts = {p.relative_to(self.directory).as_posix(): reference(self.config.workspace, p)
                     for p in sorted(self.directory.rglob("*")) if p.is_file()
                     and "_work" not in p.parts and not p.name.endswith(".partial")}
        write_json(path, {"schema_version": VERSION, "status": "complete", "stage": self.name,
            "configuration_sha256": self.config.sha256, "implementation_sha256": implementation_hash(),
            "identity": self.identity, "gate": self.gate, "inputs": self.inputs, "options": self.options,
            "artifacts": artifacts, "summary": summary})
        return path

@contextmanager
def stage(*args, **kwargs):
    attempt = Stage(*args, **kwargs)
    try:
        yield attempt
    except BaseException as exc:
        write_json(attempt.directory / "failure.json", {"status": "incomplete", "error": str(exc), "type": type(exc).__name__})
        raise

def bundle(config, prepared_f, memo=None):
    memo = {} if memo is None else memo
    accepted = check_acceptance(config.trust, checked(config.workspace, prepared_f["inputs"]["phase_e_acceptance"]), memo)
    paths = {k: checked(config.workspace, accepted["inputs"][k]) for k in ("metadata", "calibration", "calibration_review")}
    metadata = e.load_stage(config.trust, paths["metadata"], "prepare-phase-e", memo=memo)
    prepared = e.prepared_for(config.trust, metadata, memo)
    calibration_stage = e.load_stage(config.trust, paths["calibration"], "calibrate-risk", memo=memo)
    calibration = read_json(artifact(config, calibration_stage, "calibration.json"))
    if calibration["scope"] != e.calibration_scope(config.trust, metadata, prepared):
        raise ValueError("Frozen calibration scope mismatch")
    initial = d.load_stage(config.learning, checked(config.workspace, prepared_f["inputs"]["initialization"]), "initialize-model", memo=memo)
    seed = accepted["summary"]["seed"]
    if initial["options"]["seed"] != seed or initial["inputs"]["prepared"] != metadata["inputs"]["prepared"]:
        raise ValueError("Phase F initialization/data mismatch")
    return {"acceptance": accepted, "paths": paths, "metadata": metadata, "prepared": prepared,
            "calibration": calibration, "initialization": initial, "seed": seed}

def publish_round_index(config, directory, checkpoint):
    from edgefl.learning.checkpoints import load
    state = load(checked(config.workspace, checkpoint)).get("phase_state", {})
    rounds = state.get("trust", {}).get("round_artifacts", {})
    searches = state.get("search_artifacts", {})
    if set(rounds) != set(searches):
        raise ValueError("Incomplete search state")
    write_json(directory / "round_index.json", {"schema_version": "phase-f.round-index.v1",
               "rounds": {r: {**refs, "search": searches[r]} for r, refs in rounds.items()}})
    return state

"""Immutable E attempts with recursively revalidated B/C/D and E dependencies."""

import hashlib
from pathlib import Path
from contextlib import contextmanager
from edgefl.config import canonical_json
from edgefl.data.storage import read_json, write_json, sha256
from edgefl.learning import storage as d
from edgefl.learning.review import validate_scope, accepted_review
from edgefl.contracts.phase_e import VERSION, STAGES

reference, checked, artifact = d.reference, d.checked, d.artifact


def implementation_hash():
    package = Path(__file__).parents[1]
    files = list((package / "trust").glob("*.py")) + list((package / "attacks").glob("*.py"))
    files += list((package / "pipelines").glob("phase_e_*.py"))
    files += [package / p for p in ("contracts/phase_e.py", "schemas/phase_e.schema.json", "contracts/interfaces.py",
                                    "contracts/records.py", "reproducibility.py", "config.py", "cli.py", "pipelines/catalog.py")]
    values = [(p.relative_to(package).as_posix(), sha256(p)) for p in sorted(files)]
    return hashlib.sha256(canonical_json([d.implementation_hash(), values]).encode()).hexdigest()


def regression_hash():
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "tests").glob("*.py")) + [root / "scripts/check_phase_e.py"]
    return hashlib.sha256(canonical_json([(p.relative_to(root).as_posix(), sha256(p)) for p in files]).encode()).hexdigest()


def check_d_acceptance(config, path, prepared=None, seed=None, memo=None):
    memo = {} if memo is None else memo
    acceptance = d.load_stage(config.learning, path, "validate-phase-d", memo=memo)
    key = ("e_d_acceptance", path.resolve())
    if key not in memo:
        reviews = [checked(config.workspace, ref) for name, ref in acceptance["inputs"].items() if name.startswith("review_")]
        report = validate_scope(config.learning, reviews, checked(config.workspace, acceptance["summary"]["software_evidence"]))
        if report["status"] != "PASS" or acceptance["summary"]["status"] != "PASS":
            raise ValueError("Current complete Phase D acceptance required")
        first = accepted_review(config.learning, reviews[0], memo=memo)
        memo[key] = (report, first["inputs"]["prepared"])
    report, prepared_ref = memo[key]
    if prepared is not None:
        upstream = d.load_stage(config.learning, checked(config.workspace, prepared_ref), "prepare-learning-data", memo=memo)
        if upstream["artifacts"]["data.json"] != prepared["artifacts"]["data.json"] or upstream["identity"] != prepared["identity"]:
            raise ValueError("Phase D acceptance prepared-data mismatch")
    if seed is not None and report["seed"] != seed:
        raise ValueError("Phase D acceptance comparison seed mismatch")
    return acceptance


def load_stage(config, path, expected=None, memo=None, active=None):
    memo, active = ({} if memo is None else memo), (set() if active is None else active)
    path = path.resolve()
    if path in active:
        raise ValueError("Cyclic Phase E lineage")
    if path in memo:
        value = memo[path]
        if expected and value["stage"] != expected:
            raise ValueError("Phase E stage mismatch")
        return value
    if not path.is_relative_to(config.workspace):
        raise ValueError("Phase E completion outside workspace")
    active.add(path)
    value = read_json(path)
    keys = {"schema_version", "status", "stage", "configuration_sha256", "implementation_sha256", "identity", "gate", "inputs", "options", "artifacts", "summary"}
    if set(value) != keys or value["schema_version"] != VERSION or value["status"] != "complete" or value["stage"] not in STAGES:
        raise ValueError("Invalid Phase E completion")
    if expected and value["stage"] != expected:
        raise ValueError("Phase E stage mismatch")
    if value["configuration_sha256"] != config.sha256 or value["implementation_sha256"] != implementation_hash():
        raise ValueError("Stale Phase E configuration or implementation")
    required = STAGES[value["stage"]]
    extras = set(value["inputs"]) - required
    if not required <= set(value["inputs"]) or (extras and (value["stage"] != "validate-phase-e" or any(not k.startswith("review_") for k in extras))):
        raise ValueError("Invalid Phase E dependencies")
    checked(config.workspace, value["gate"])
    for ref in value["artifacts"].values():
        checked(config.workspace, ref)
    if value["stage"] == "run-clean-pilot":
        initial = read_json(artifact(config, value, "initialization.json"))
        if (set(initial) != {"identity", "options", "artifacts"} or initial["identity"] != value["identity"]
                or initial["options"] != {"seed": value["options"]["seed"]} or set(initial["artifacts"]) != {"mlp.pt"}):
            raise ValueError("Invalid clean pilot initialization lineage")
        checked(config.workspace, initial["artifacts"]["mlp.pt"])
    if "round_index.json" in value["artifacts"]:
        index = read_json(artifact(config, value, "round_index.json"))
        if set(index) != {"schema_version", "rounds"} or index["schema_version"] != "phase-e.round-index.v1":
            raise ValueError("Invalid Phase E round index")
        if set(index["rounds"]) != {str(r) for r in range(1, value["summary"]["completed_steps"] + 1)}:
            raise ValueError("Incomplete Phase E round lineage")
        for refs in index["rounds"].values():
            if set(refs) != {"assessment", "simulator", "reputation"}:
                raise ValueError("Invalid round artifact references")
            for ref in refs.values():
                checked(config.workspace, ref)
    parents = {}
    for name, ref in value["inputs"].items():
        upstream_path = checked(config.workspace, ref)
        if name in ("prepared", "initialization", "phase_d_acceptance"):
            stage_name = {"prepared": "prepare-learning-data", "initialization": "initialize-model", "phase_d_acceptance": "validate-phase-d"}[name]
            parent = d.load_stage(config.learning, upstream_path, stage_name, memo=memo)
        else:
            stage_name = {"metadata": "prepare-phase-e", "attack_plan": "plan-attacks", "calibration": "calibrate-risk", "calibration_review": "review-phase-e"}.get(name)
            if name.startswith(("fit_", "audit_")):
                stage_name = "run-clean-pilot"
            if name.startswith("review_"):
                stage_name = "review-phase-e"
            parent = load_stage(config, upstream_path, stage_name, memo, active)
        if parent["identity"] != value["identity"] or parent["gate"] != value["gate"]:
            raise ValueError("Phase E dependency identity mismatch")
        parents[name] = parent
    if "phase_d_acceptance" in parents:
        prepared = parents.get("prepared")
        if prepared is None:
            prepared = d.load_stage(config.learning, checked(config.workspace, parents["metadata"]["inputs"]["prepared"]), "prepare-learning-data", memo=memo)
        check_d_acceptance(config, checked(config.workspace, value["inputs"]["phase_d_acceptance"]), prepared,
                           value["options"]["seed"] if value["stage"] == "run-trust-federated" else None, memo)
    for key in ("final_checkpoint", "software_evidence"):
        if key in value["summary"]:
            checked(config.workspace, value["summary"][key])
    if value["summary"].get("best"):
        checked(config.workspace, value["summary"]["best"]["reference"])
    if "resume" in value["options"]:
        marker = read_json(checked(config.workspace, value["options"]["resume"]))
        checked(config.workspace, marker["checkpoint"])
    active.remove(path)
    memo[path] = value
    return value


class Stage(d.Stage):
    def finish(self, summary):
        path = self.directory / "completion.json"
        if path.exists():
            raise ValueError("Phase E attempt already completed")
        artifacts = {p.relative_to(self.directory).as_posix(): reference(self.config.workspace, p)
                     for p in sorted(self.directory.rglob("*")) if p.is_file() and "_work" not in p.parts and not p.name.endswith(".partial")}
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


def prepared_for(config, metadata, memo=None):
    return d.load_stage(config.learning, checked(config.workspace, metadata["inputs"]["prepared"]), "prepare-learning-data", memo=memo)


def calibration_scope(config, metadata, prepared):
    values = config.learning.values
    return {"metadata": metadata["artifacts"]["metadata.sqlite"], "prepared": prepared["artifacts"]["data.json"],
            "identity": metadata["identity"], "learning_configuration": config.learning.sha256,
            "phase_e_configuration": config.sha256, "implementation": implementation_hash(),
            "rounds": values["rounds"], "participation": values["participation"], "full_first_round": True}


def publish_round_index(config, directory, checkpoint):
    from edgefl.learning.checkpoints import load
    state = load(checked(config.workspace, checkpoint)).get("phase_state", {})
    write_json(directory / "round_index.json", {"schema_version": "phase-e.round-index.v1", "rounds": state.get("round_artifacts", {})})
    return state

"""Capture diagnostics, including explicit inspection of historical evidence."""

import uuid
from pathlib import Path

from edgefl.config import workspace_path
from edgefl.data.inventory import verify_sources
from edgefl.data.storage import read_json, sha256, write_json, implementation_hash
from edgefl.data.verification import verify
from edgefl.pipelines.phase_b_common import stage, upstream, artifact, require_link


def historical_inputs(config, audit_path, provenance_path, dataset):
    """Inspect hashes and registered bytes without granting stale code eligibility."""
    active, done = set(), set()
    def inspect(path):
        path = path.resolve()
        if not path.is_relative_to(config.workspace.resolve()):
            raise ValueError("Historical evidence outside workspace")
        if path in active:
            raise ValueError("Cyclic historical evidence")
        if path in done:
            return read_json(path)
        active.add(path)
        value = read_json(path)
        if (value.get("schema_version") not in ("phase-b.v1", "phase-b.v2")
                or value.get("status") != "complete" or value.get("eligible_for_training") is not False):
            raise ValueError("Historical review requires ineligible completed Phase B evidence")
        for reference in (*value["artifacts"].values(), *value["inputs"].values()):
            target = workspace_path(config.workspace, reference["path"])
            if not target.is_file() or sha256(target) != reference["sha256"]:
                raise ValueError("Historical artifact hash mismatch: " + reference["path"])
            if reference["role"] == "completion":
                inspect(target)
        active.remove(path); done.add(path)
        return value
    audit, provenance = inspect(audit_path), inspect(provenance_path)
    if audit["stage"] != "audit" or provenance["stage"] != "provenance" or provenance["dataset"] != dataset:
        raise ValueError("Wrong historical audit/provenance identity")
    if provenance["inputs"]["audit"]["sha256"] != sha256(audit_path):
        raise ValueError("Historical provenance/audit lineage mismatch")
    registry = read_json(artifact(config, audit, "registry"))
    if registry["selected"][dataset] != config.values["selected"][dataset]:
        raise ValueError("Historical selected dataset differs from current configuration")
    verify_sources(config.workspace, registry)
    return audit, provenance, registry


def execute(config, audit_path, provenance_path, dataset, captures=(), historical=False):
    if historical:
        fingerprint = implementation_hash("diagnose-captures")
        audit, provenance, registry = historical_inputs(config, audit_path, provenance_path, dataset)
        if set(captures) - {pair["capture_id"] for pair in registry["pairs"]}:
            raise ValueError("Unknown diagnostic capture")
        directory = workspace_path(config.workspace, config.values["output"]) / "diagnostic-review" / ("attempt-" + uuid.uuid4().hex[:12])
        directory.mkdir(parents=True, exist_ok=False)
        summary = verify(config.workspace, registry, dataset, artifact(config, provenance, "provenance"),
                         artifact(config, provenance, "candidates"), directory,
                         config.values["session_inactivity_seconds"], diagnostic=True, captures=captures)
        summary["historical_inspection"] = True
        summary["acceptance_policy_status"] = "AWAITING_EVIDENCE_REVIEW"
        summary["eligible_for_training"] = False
        write_json(directory / "summary.json", summary)
        paths = ("summary.json", "timing_anomalies.csv", "anchor_reversals.csv", "evidence.csv")
        metadata = {"schema_version": "phase-b.diagnostic-review.v2", "status": "complete",
                    "dataset": dataset, "stage": "diagnose-captures", "eligible_for_training": False,
                    "implementation_sha256": fingerprint,
                    "configuration_sha256": config.sha256,
                    "inputs": {key: {"path": path.relative_to(config.workspace).as_posix(), "sha256": sha256(path)}
                               for key, path in (("audit", audit_path), ("provenance", provenance_path))},
                    "artifacts": {name: {"path": (directory / name).relative_to(config.workspace).as_posix(),
                                         "sha256": sha256(directory / name)} for name in paths}}
        write_json(directory / "completion.json", metadata)
        return directory / "completion.json"
    audit = upstream(config, audit_path, "audit")
    provenance = upstream(config, provenance_path, "provenance", dataset)
    require_link(config, provenance, "audit", audit_path)
    registry = read_json(artifact(config, audit, "registry"))
    verify_sources(config.workspace, registry)
    known = {pair["capture_id"] for pair in registry["pairs"]}
    if set(captures) - known:
        raise ValueError("Unknown diagnostic capture")
    with stage(config, "diagnose-captures", dataset, {"audit": audit_path, "provenance": provenance_path},
               {"captures": sorted(captures)}) as run:
        if run.reused:
            return run.reused
        summary = verify(config.workspace, registry, dataset, artifact(config, provenance, "provenance"),
                         artifact(config, provenance, "candidates"), run.directory,
                         config.values["session_inactivity_seconds"], progress=run.event,
                         diagnostic=True, captures=captures)
        summary["acceptance_policy_status"] = "AWAITING_EVIDENCE_REVIEW"
        return run.finish(summary, {name: run.directory / (name + ".csv")
                                   for name in ("timing_anomalies", "anchor_reversals")})

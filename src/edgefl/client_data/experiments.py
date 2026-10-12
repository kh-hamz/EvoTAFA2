"""Resolve and materialize a frozen task-specific view of verified global pools."""

from dataclasses import asdict

from edgefl.config import workspace_path
from edgefl.contracts.phase_c import ExperimentContract
from edgefl.data.schema import LABELS
from edgefl.data.storage import read_json, rows, writer, database
from edgefl.contracts.phase_b import MANIFEST_FIELDS


def resolve_experiment(values, dataset, split, summary, fold, task):
    protocol = split["options"]["protocol"]
    info = summary["folds"][fold]
    if not info["feasible"]:
        raise ValueError("Selected fold is infeasible")
    if protocol == "B" and task != "binary":
        raise ValueError("Protocol B currently supports binary evaluation only")
    scope = values["closed_set_scope"] if task == "multiclass" else "binary_all_verified"
    if task == "multiclass":
        supported = set(info.get("closed_set_manifest_classes", ()))
        if not supported <= set(LABELS):
            raise ValueError("Closed-set scope contains unknown labels")
        if scope == "full_15" and supported != set(LABELS):
            raise ValueError("Full-15 experiment lacks verified closed-set class support")
        if "Normal" not in supported or len(supported) < 2:
            raise ValueError("Closed-set scope requires benign and attack support")
        labels = tuple(sorted(supported))
    else:
        labels = ("Normal", "Attack")
    experiment = ExperimentContract(dataset, task, protocol, fold, info["interpretation"],
                                    scope, labels, tuple(info["held_out_captures"]))
    return {"schema_version": "phase-c.experiment.v2", **asdict(experiment),
            "labels": list(labels), "held_out_captures": list(experiment.held_out_captures),
            "manifest_role": "closed_set" if task == "multiclass" else "splits",
            "excluded_classes": sorted(set(LABELS) - set(labels)) if task == "multiclass" else [],
            "support_policy": values["support_policy"]}


def verify_experiment(config, experiment, split, validation):
    report = read_json(workspace_path(config.workspace, validation["artifacts"]["summary"]["path"]))
    if report["status"] not in ("PASS", "PASS_WITH_LIMITATIONS"):
        raise ValueError("Phase B validation blocks the experiment")
    protocol = split["options"]["protocol"].lower()
    reference = validation["inputs"].get("splits_" + protocol)
    if reference is None or reference["sha256"] != experiment["split_completion_sha256"]:
        raise ValueError("Experiment split was not accepted by Phase B")
    expected = resolve_experiment(config.values, split["dataset"], split,
                                 read_json(workspace_path(config.workspace, split["artifacts"]["summary"]["path"])),
                                 experiment["fold"], experiment["task"])
    expected["split_completion_sha256"] = reference["sha256"]
    if experiment != expected:
        raise ValueError("Stale or mismatched experiment contract")


def materialize_experiment(config, experiment, split, validation, directory):
    source = workspace_path(config.workspace, split["artifacts"][experiment["manifest_role"]]["path"])
    panel_completion = workspace_path(config.workspace,
                                     validation["inputs"]["panel_" + experiment["protocol"].lower()]["path"])
    panel_meta = read_json(panel_completion)
    panel = workspace_path(config.workspace, panel_meta["artifacts"]["panel"]["path"])
    db = database(directory / "_work/experiment.sqlite")
    db.execute("CREATE TABLE trusted(oid TEXT PRIMARY KEY)")
    try:
        stream, output = writer(directory / "global_splits.csv", MANIFEST_FIELDS["splits"])
        with stream:
            for row in rows(source):
                if row["fold"] != experiment["fold"]:
                    continue
                output.writerow(row)
                if row["partition"] == "trusted":
                    db.execute("INSERT INTO trusted VALUES (?)", (row["observation_id"],))
        db.commit()
        stream, output = writer(directory / "trusted_panel.csv", MANIFEST_FIELDS["panel"])
        with stream:
            for row in rows(panel):
                if row["fold"] == experiment["fold"] and db.execute(
                        "SELECT 1 FROM trusted WHERE oid=?", (row["observation_id"],)).fetchone():
                    output.writerow(row)
    finally:
        db.close()

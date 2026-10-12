"""Grouping orchestration."""

from edgefl.data.grouping import group
from edgefl.data.storage import read_json,write_json,ref
from edgefl.contracts.records import DatasetManifest,ArtifactRef
from dataclasses import asdict
from edgefl.pipelines.phase_b_common import artifact,require_link,stage,upstream


def execute(config,provenance_path,evidence_path,dataset):
    source = upstream(config,provenance_path,"provenance",dataset)
    evidence = upstream(config,evidence_path,"verify-captures",dataset)
    require_link(config,evidence,"provenance",provenance_path)
    with stage(config,"group",dataset,{"provenance":provenance_path,"evidence":evidence_path}) as run:
        if run.reused:
            return run.reused
        summary = group(artifact(config,source,"provenance"),artifact(config,evidence,"evidence"),
                        run.directory,config.values["temporal_block_seconds"])
        audit_path = config.workspace/source["inputs"]["audit"]["path"]
        audit = upstream(config,audit_path,"audit")
        registry = read_json(artifact(config,audit,"registry"))
        schema_ref = audit["artifacts"]["field_schema"]
        provenance_ref = source["artifacts"]["provenance"]
        retained = sum(item["observations"] for item in summary["class_support"].values())
        manifest = None
        if retained:
            manifest = asdict(DatasetManifest(
                dataset_id=read_json(artifact(config,source,"summary"))["selected_sha256"],
                sources=tuple(ArtifactRef(item["path"],item["sha256"]) for item in registry["files"]),
                schema=ArtifactRef(schema_ref["path"],schema_ref["sha256"]),
                provenance=ArtifactRef(provenance_ref["path"],provenance_ref["sha256"]),
                provenance_status="verified_subset"))
        write_json(run.directory/"dataset.json",{"dataset_manifest":manifest,
            "retained_scope":ref(config.workspace,run.directory/"groups.csv","groups"),
            "verified_observations":retained,"eligible_for_training":False})
        artifacts = {name:run.directory/(name+".csv") for name in ("groups","duplicates","feature_collisions")}
        artifacts["dataset"] = run.directory/"dataset.json"
        return run.finish(summary,artifacts)

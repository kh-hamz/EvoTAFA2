"""Cross-stage lineage and acceptance orchestration."""

from edgefl.data.inventory import verify_sources
from edgefl.data.storage import read_json
from edgefl.data.validation import validate
from edgefl.data.origin_validation import validate_origins
from edgefl.pipelines.phase_b_common import artifact,require_link,stage,upstream

EXPECTED = {"audit":"audit","provenance":"provenance","evidence":"verify-captures","groups":"group",
            "splits_a":"global-split","splits_b":"global-split","panel_a":"trusted-panel","panel_b":"trusted-panel"}


def execute(config,inputs,dataset):
    metadata = {key:upstream(config,path,EXPECTED[key],None if key=="audit" else dataset) for key,path in inputs.items()}
    require_link(config,metadata["provenance"],"audit",inputs["audit"])
    require_link(config,metadata["evidence"],"provenance",inputs["provenance"])
    require_link(config,metadata["groups"],"evidence",inputs["evidence"])
    require_link(config,metadata["groups"],"provenance",inputs["provenance"])
    for protocol in ("a","b"):
        require_link(config,metadata["splits_"+protocol],"groups",inputs["groups"])
        require_link(config,metadata["panel_"+protocol],"splits",inputs["splits_"+protocol])
        if metadata["splits_"+protocol]["options"]["protocol"] != protocol.upper():
            raise ValueError("Wrong protocol supplied to validator")
    with stage(config,"validate-phase-b",dataset,inputs) as run:
        if run.reused:
            return run.reused
        registry = read_json(artifact(config,metadata["audit"],"registry"))
        verify_sources(config.workspace,registry)
        paths = {key:artifact(config,metadata[key],key.split("_")[0]) for key in
                 ("provenance","evidence","groups","splits_a","splits_b","panel_a","panel_b")}
        paths["duplicates"] = artifact(config,metadata["groups"],"duplicates")
        summaries = {key:read_json(artifact(config,value,"summary")) for key,value in metadata.items()}
        audit = summaries["audit"]["files"][registry["selected"][dataset]]
        summary = validate(paths,summaries,run.directory,config.values["purge_seconds"],
                           config.values["panel_per_class"],audit.get("logical_records",0))
        origin_errors = validate_origins(paths["provenance"],artifact(config,metadata["provenance"],"candidates"),
            registry,summaries["audit"],dataset,run.directory/"_work/origins.sqlite")
        summary["errors"].extend(origin_errors)
        if origin_errors:
            summary["status"] = "FAIL"
        return run.finish(summary,{})

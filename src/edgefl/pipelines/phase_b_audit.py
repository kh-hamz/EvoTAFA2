"""Source-registration and audit orchestration only."""

from edgefl.data.audit import audit_csv
from edgefl.data import schema
from edgefl.data.inventory import register
from edgefl.data.storage import write_json
from edgefl.pipelines.phase_b_common import disk_check, foundation, stage


def execute(config,foundation_path):
    foundation(config.workspace,foundation_path)
    with stage(config,"audit","archive",{"foundation":foundation_path}) as run:
        if run.reused:
            return run.reused
        registry = register(config.workspace,config.values,run.event)
        registry["label_aliases"] = config.values["label_aliases"]
        capacity = disk_check(config.workspace,registry)
        write_json(run.directory/"registry.json",registry)
        write_json(run.directory/"field_schema.json", {"version":schema.VERSION, "numeric":sorted(schema.NUMERIC),
            "text":sorted(schema.TEXT), "boolean":sorted(schema.BOOLEAN), "hex":sorted(schema.HEX),
            "feature_exclusions":sorted(schema.EXCLUDE_FEATURES), "label_aliases":registry["label_aliases"],
            "timestamp_policy":"Dates only from verified packet correspondence",
            "zero_sentinel_policy":"Empty, 0 and 0.0 are equivalent only for declared non-time/non-label fields"})
        reports = {}
        for item in registry["files"]:
            if item["kind"] != "csv":
                continue
            try:
                reports[item["path"]] = audit_csv(config.workspace/item["path"],
                    run.directory/"_work/audit.sqlite",config.values["label_aliases"],run.event,100000)
            except ValueError as exc:
                reports[item["path"]] = {"structural_failure":str(exc)}
            run.event(event="audit_source_complete",path=item["path"])
        expected = {"logical_records":2219201,"columns":63,"classes":15,"exact_duplicate_occurrences":815}
        primary = reports[registry["selected"]["primary"]]
        actual = {key:(len(primary.get("canonical_labels",{})) if key=="classes" else primary.get(key)) for key in expected}
        comparison = {key:{"expected":number,"actual":actual[key],"matches":number==actual[key]} for key,number in expected.items()}
        summary = {"files":reports,"dnn_expectations":comparison,"disk":capacity,
                   "discrepancy_policy":"Raw logical CSV records and exact decoded field tuples; semantic errors are counted separately."}
        return run.finish(summary,{"registry":run.directory/"registry.json","field_schema":run.directory/"field_schema.json"})

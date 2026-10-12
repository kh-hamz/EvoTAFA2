"""Capture evidence orchestration; tool failures remain explicit quarantine evidence."""

from edgefl.data.inventory import verify_sources
from edgefl.data.verification import verify
from edgefl.data.storage import read_json
from edgefl.pipelines.phase_b_common import artifact,require_link,stage,upstream


def execute(config,audit_path,provenance_path,dataset):
    audit = upstream(config,audit_path,"audit")
    source = upstream(config,provenance_path,"provenance",dataset)
    require_link(config,source,"audit",audit_path)
    with stage(config,"verify-captures",dataset,{"audit":audit_path,"provenance":provenance_path}) as run:
        if run.reused:
            return run.reused
        registry = read_json(artifact(config,audit,"registry"))
        verify_sources(config.workspace,registry)
        summary = verify(config.workspace,registry,dataset,artifact(config,source,"provenance"),
                         artifact(config,source,"candidates"),run.directory,
                         config.values["session_inactivity_seconds"],run.event)
        return run.finish(summary,{"evidence":run.directory/"evidence.csv"})

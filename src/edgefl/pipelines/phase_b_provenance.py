"""Selected-to-source provenance orchestration."""

from edgefl.data.inventory import verify_sources
from edgefl.data.provenance import recover
from edgefl.data.storage import read_json
from edgefl.pipelines.phase_b_common import artifact,stage,upstream


def execute(config,audit_path,dataset):
    audit = upstream(config,audit_path,"audit")
    registry = read_json(artifact(config,audit,"registry"))
    with stage(config,"provenance",dataset,{"audit":audit_path}) as run:
        if run.reused:
            return run.reused
        verify_sources(config.workspace,registry)
        summary = recover(config.workspace,registry,read_json(artifact(config,audit,"summary")),
                          dataset,run.directory,run.event)
        return run.finish(summary,{name:run.directory/(name+".csv") for name in ("provenance","candidates")})

"""Global split orchestration."""

from edgefl.data.splitting import split
from edgefl.data.storage import read_json
from edgefl.pipelines.phase_b_common import artifact,stage,upstream


def execute(config,groups_path,dataset,protocol):
    groups = upstream(config,groups_path,"group",dataset)
    with stage(config,"global-split",dataset,{"groups":groups_path},{"protocol":protocol}) as run:
        if run.reused:
            return run.reused
        summary = split(artifact(config,groups,"groups"),run.directory,protocol,config.values["seed"],
                        config.values["purge_seconds"],[name.removesuffix(".csv") for name in config.values["pairings"]])
        artifacts = {"splits":run.directory/"splits.csv"}
        if protocol == "A":
            artifacts["closed_set"] = run.directory/"closed_set.csv"
        return run.finish(summary,artifacts)

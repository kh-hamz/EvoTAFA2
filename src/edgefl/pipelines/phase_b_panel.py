"""Trusted-panel orchestration."""

from edgefl.data.trusted_panel import panel
from edgefl.data.storage import read_json
from edgefl.pipelines.phase_b_common import artifact,stage,upstream


def execute(config,splits_path,dataset):
    splits = upstream(config,splits_path,"global-split",dataset)
    with stage(config,"trusted-panel",dataset,{"splits":splits_path}) as run:
        if run.reused:
            return run.reused
        summary = panel(artifact(config,splits,"splits"),read_json(artifact(config,splits,"summary")),
                        run.directory,config.values["seed"],config.values["panel_per_class"])
        return run.finish(summary,{"panel":run.directory/"panel.csv"})

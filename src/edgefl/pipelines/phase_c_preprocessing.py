"""Phase C training-only preprocessing orchestration."""

from edgefl.client_data.config import phase_b
from edgefl.client_data.preprocessing import fit
from edgefl.client_data.storage import (artifact, load_completion,
                                        load_phase_b_completion, require_link, stage)
from edgefl.config import workspace_path
from edgefl.data.integrity import VerificationContext


def execute(config, local_split_path, provenance_path, dataset, context=None):
    context = context if context is not None else VerificationContext()
    local = load_completion(config, local_split_path, "local-split", dataset, context=context)
    provenance = load_phase_b_completion(config, provenance_path, "provenance", dataset, context)
    assignment_path = workspace_path(config.workspace, local["inputs"]["assignments"]["path"])
    assignment = load_completion(config, assignment_path, "assign-clients", dataset, context=context)
    split_path = workspace_path(config.workspace, assignment["inputs"]["split"]["path"])
    split = load_phase_b_completion(config, split_path, "global-split", dataset, context)
    groups_path = workspace_path(config.workspace, split["inputs"]["groups"]["path"])
    groups = load_phase_b_completion(config, groups_path, "group", dataset, context)
    require_link(config, groups, "provenance", provenance_path)
    options = assignment["options"]
    with stage(config, "fit-preprocessing", dataset,
               {"local_split": local_split_path, "provenance": provenance_path}, options, context) as run:
        if run.reused:
            return run.reused
        selected = workspace_path(config.workspace, phase_b(config).values["selected"][dataset])
        summary = fit(artifact(config, local, "local_splits"),
                      artifact(config, provenance, "provenance"), selected,
                      run.directory, config.values["preprocessing"])
        return run.finish(summary, {
            "transformer": run.directory / "transformer.json",
            "feature_dictionary": run.directory / "feature_dictionary.csv",
            "label_mappings": run.directory / "label_mappings.json",
            "fit_rows": run.directory / "fit_rows.csv",
            "feature_contract": run.directory / "feature_contract.json",
        })

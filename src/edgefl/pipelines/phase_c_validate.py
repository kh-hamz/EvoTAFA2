"""Phase C complete pretraining-gate orchestration."""

from edgefl.client_data.config import phase_b
from edgefl.client_data.storage import (artifact, artifact_ref, load_completion,
                                        load_phase_b_completion, require_link, stage,
                                        write_json, read_json)
from edgefl.client_data.validation import validate
from edgefl.config import workspace_path
from edgefl.pipelines.phase_c_common import accepted_phase_b
from edgefl.data.integrity import VerificationContext


def execute(config, inputs, dataset, context=None):
    context = context if context is not None else VerificationContext()
    validation, split = accepted_phase_b(
        config, inputs["phase_b_validation"], inputs["split"], dataset,
        inputs["fold"], context)
    assignments = load_completion(config, inputs["assignments"], "assign-clients", dataset, context=context)
    local = load_completion(config, inputs["local_split"], "local-split", dataset, context=context)
    preprocessing = load_completion(config, inputs["preprocessing"],
                                    "fit-preprocessing", dataset, context=context)
    provenance = load_phase_b_completion(config, inputs["provenance"], "provenance", dataset, context)
    require_link(config, assignments, "phase_b_validation", inputs["phase_b_validation"])
    require_link(config, assignments, "split", inputs["split"])
    require_link(config, local, "assignments", inputs["assignments"])
    require_link(config, preprocessing, "local_split", inputs["local_split"])
    require_link(config, preprocessing, "provenance", inputs["provenance"])
    options = {"fold": inputs["fold"], "scenario": inputs["scenario"], "task": assignments["options"]["task"]}
    experiment = read_json(artifact(config, assignments, "experiment_contract"))
    if assignments["options"] != options or local["options"] != options or preprocessing["options"] != options:
        raise ValueError("Phase C fold/scenario mismatch across stages")
    stage_inputs = {key: value for key, value in inputs.items()
                    if key not in ("fold", "scenario")}
    with stage(config, "validate-phase-c", dataset, stage_inputs, options, context) as run:
        if run.reused:
            return run.reused
        selected = workspace_path(config.workspace, phase_b(config).values["selected"][dataset])
        report = validate(
            splits=artifact(config, assignments, "global_splits"),
            assignments=artifact(config, assignments, "assignments"),
            local_splits=artifact(config, local, "local_splits"),
            client_manifests=artifact(config, local, "client_manifests"),
            provenance=artifact(config, provenance, "provenance"), selected=selected,
            transformer=artifact(config, preprocessing, "transformer"),
            feature_dictionary=artifact(config, preprocessing, "feature_dictionary"),
            feature_contract=artifact(config, preprocessing, "feature_contract"),
            label_mappings=artifact(config, preprocessing, "label_mappings"),
            fit_rows=artifact(config, preprocessing, "fit_rows"), directory=run.directory,
            config=config.values, fold=inputs["fold"], scenario=inputs["scenario"],
            experiment=experiment, trusted_panel=artifact(config, assignments, "trusted_panel"))
        # The count table is derived from independently checked local manifests.
        from edgefl.client_data.validation import validate_support_table
        validate_support_table(artifact(config, local, "class_support"), report)
        write_json(run.directory / "pretraining_report.json", report)
        training_inputs = {
            "eligible_for_training": report["status"] == "PASS",
            "dataset": dataset, "fold": inputs["fold"], "scenario": inputs["scenario"],
            "schema_version": "phase-c.training-inputs.v2", "task": options["task"],
            "experiment_contract": assignments["artifacts"]["experiment_contract"],
            "trusted_panel": assignments["artifacts"]["trusted_panel"],
            "global_splits": assignments["artifacts"]["global_splits"],
            "local_splits": local["artifacts"]["local_splits"],
            "client_manifests": local["artifacts"]["client_manifests"],
            "class_support": local["artifacts"]["class_support"],
            "transformer": preprocessing["artifacts"]["transformer"],
            "feature_contract": preprocessing["artifacts"]["feature_contract"],
            "label_mappings": preprocessing["artifacts"]["label_mappings"],
            "phase_b_validation": artifact_ref(config.workspace,
                                                inputs["phase_b_validation"],
                                                "completion", "phase-b.v2"),
        }
        write_json(run.directory / "training_inputs.json", training_inputs)
        return run.finish(report, {
            "pretraining_report": run.directory / "pretraining_report.json",
            "training_inputs": run.directory / "training_inputs.json",
        }, eligible_for_training=report["status"] == "PASS")

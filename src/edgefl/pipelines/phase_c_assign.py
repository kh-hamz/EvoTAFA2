"""Phase C immutable client-assignment orchestration."""

from edgefl.client_data.assignment import assign
from edgefl.client_data.storage import artifact, stage, read_json, write_json, artifact_ref
from edgefl.client_data.experiments import resolve_experiment, materialize_experiment
from edgefl.client_data.policies import support_policy
from edgefl.data.integrity import VerificationContext
from edgefl.pipelines.phase_c_common import accepted_phase_b


def execute(config, validation_path, split_path, dataset, fold, scenario, task, context=None):
    context = context if context is not None else VerificationContext()
    if scenario not in config.values["scenarios"]:
        raise ValueError(f"Unknown Phase C scenario: {scenario}")
    validation, split = accepted_phase_b(config, validation_path, split_path, dataset, fold, context)
    experiment = resolve_experiment(config.values, dataset, split,
                                    read_json(artifact(config, split, "summary")), fold, task)
    experiment["split_completion_sha256"] = artifact_ref(config.workspace, split_path, "completion", "phase-b.v2")["sha256"]
    options = {"fold": fold, "scenario": scenario, "task": task}
    with stage(config, "assign-clients", dataset,
               {"phase_b_validation": validation_path, "split": split_path}, options, context) as run:
        if run.reused:
            return run.reused
        values = config.values
        materialize_experiment(config, experiment, split, validation, run.directory)
        write_json(run.directory / "experiment_contract.json", experiment)
        summary = assign(
            run.directory / "global_splits.csv", fold, scenario, values["scenarios"][scenario],
            run.directory, values["seed"], values["clients"],
            values["max_partition_attempts"], values["minimum_client_observations"],
            values["minimum_groups_per_client"], False, policy=support_policy(values),
            limits=values["near_iid_limits"])
        return run.finish(summary, {
            "assignments": run.directory / "assignments.csv",
            "clients": run.directory / "clients.json",
            "partition_attempts": run.directory / "partition_attempts.json",
            "experiment_contract": run.directory / "experiment_contract.json",
            "global_splits": run.directory / "global_splits.csv",
            "trusted_panel": run.directory / "trusted_panel.csv",
        })

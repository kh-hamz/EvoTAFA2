"""Phase C client-local validation orchestration."""

from edgefl.client_data.local_split import split_local
from edgefl.client_data.storage import artifact, load_completion, stage
from edgefl.client_data.policies import support_policy
from edgefl.data.integrity import VerificationContext


def execute(config, assignments_path, dataset, context=None):
    context = context if context is not None else VerificationContext()
    assignments = load_completion(config, assignments_path, "assign-clients", dataset, context=context)
    options = assignments["options"]
    with stage(config, "local-split", dataset, {"assignments": assignments_path}, options, context) as run:
        if run.reused:
            return run.reused
        values = config.values
        summary = split_local(
            artifact(config, assignments, "assignments"), run.directory, values["seed"],
            values["local_validation_fraction"], values["minimum_training_observations"],
            False, policy=support_policy(values), attempts=values["max_partition_attempts"])
        return run.finish(summary, {
            "local_splits": run.directory / "local_splits.csv",
            "client_manifests": run.directory / "client_manifests.json",
            "class_support": run.directory / "class_support.csv",
        })

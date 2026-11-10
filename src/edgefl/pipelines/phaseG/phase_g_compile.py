"""Compile experiment intent without training or accessing evaluation data."""
from edgefl.experiments.matrix import compile_matrix
from edgefl.experiments.dependencies import inventory, materialize_configurations
from edgefl.experiments.storage import stage, implementation_hash
from edgefl.contracts.phase_g import digest
from edgefl.data.storage import write_json


def execute(config):
    matrix = compile_matrix(config.values)
    matrix["definition_id"] = matrix["matrix_id"]
    matrix["matrix_id"] = digest({"definition": matrix["definition_id"], "configuration": config.sha256, "implementation": implementation_hash()})
    with stage(config, "compile-experiments", {}, matrix_id=matrix["matrix_id"]) as attempt:
        write_json(attempt.directory / "specification.json", config.values)
        write_json(attempt.directory / "cases.json", matrix)
        dependencies = inventory(matrix)
        write_json(attempt.directory / "dependencies.json", dependencies)
        generated = materialize_configurations(config, matrix, attempt.directory / "configurations")
        write_json(attempt.directory / "configuration_bindings.json", generated)
        return attempt.finish({"matrix_id": matrix["matrix_id"], "counts": matrix["counts"], "unique_cases": matrix["unique_cases"],
            "nominal_upstream_training_runs": dependencies["nominal_upstream_training_runs"], "status": "COMPILED",
            "capture_folds_unregistered": matrix["capture_folds_unregistered"], "deferred": matrix["deferred"]})

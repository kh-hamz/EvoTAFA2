"""Prepare learning arrays without fitting or altering Phase C membership."""
from dataclasses import asdict
from edgefl.contracts.phase_d import ExperimentIdentity
from edgefl.learning.storage import stage, verify_gate


def execute(config, gate, dataset, task, fold, scenario):
    identity = asdict(ExperimentIdentity(dataset, task, fold, scenario, config.values["variant"]))
    verify_gate(config, gate, identity)
    from edgefl.learning.data import materialize
    with stage(config, "prepare-learning-data", identity, gate) as run:
        return run.finish(materialize(config, gate, identity, run.directory))

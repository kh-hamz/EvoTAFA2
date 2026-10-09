"""Verify D acceptance and derive immutable original-label metadata."""

from edgefl.trust.storage import stage, checked, check_d_acceptance
from edgefl.learning.storage import load_stage


def execute(config, prepared_path, acceptance_path):
    memo = {}
    prepared = load_stage(config.learning, prepared_path, "prepare-learning-data", memo=memo)
    check_d_acceptance(config, acceptance_path, prepared, memo=memo)
    from edgefl.trust.metadata import prepare
    with stage(config, "prepare-phase-e", prepared["identity"], checked(config.workspace, prepared["gate"]),
               {"prepared": prepared_path, "phase_d_acceptance": acceptance_path}) as attempt:
        return attempt.finish(prepare(config.learning, prepared, attempt.directory))

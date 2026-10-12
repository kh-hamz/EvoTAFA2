"""Freeze one simulator condition before matched methods train."""

from edgefl.trust.storage import stage, load_stage, checked
from edgefl.data.storage import write_json


def execute(config, metadata_path, seed, condition):
    if seed not in config.learning.values["seeds"]:
        raise ValueError("Attack plans require a registered comparison seed")
    metadata = load_stage(config, metadata_path, "prepare-phase-e")
    from edgefl.trust.metadata import Metadata
    from edgefl.attacks.plan import build_plan
    plan = build_plan(Metadata(config.workspace, metadata), config.values, seed, condition)
    with stage(config, "plan-attacks", metadata["identity"], checked(config.workspace, metadata["gate"]),
               {"metadata": metadata_path}, {"seed": seed, "condition": condition}) as attempt:
        write_json(attempt.directory / "attack_plan.json", plan)
        return attempt.finish({"status": plan["coverage_status"], "seed": seed, "condition": condition,
                               "compromised_clients": len(plan["compromised"]), "coverage": plan["coverage"]})

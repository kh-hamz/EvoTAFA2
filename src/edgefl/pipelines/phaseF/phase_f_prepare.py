"""Verify current E acceptance and matching initialization before F execution."""
from edgefl.optimization.storage import stage, checked, reference, bundle
from edgefl.trust.review import check_acceptance
from edgefl.learning import storage as d

def execute(config, acceptance_path, initialization_path):
    memo = {}
    accepted = check_acceptance(config.trust, acceptance_path, memo)
    initial = d.load_stage(config.learning, initialization_path, "initialize-model", memo=memo)
    if initial["identity"] != accepted["identity"] or initial["gate"] != accepted["gate"]:
        raise ValueError("Initialization scope mismatch")
    inputs = {"phase_e_acceptance": acceptance_path, "initialization": initialization_path}
    value = {"inputs": {k: reference(config.workspace, p) for k, p in inputs.items()}}
    context = bundle(config, value, memo)
    with stage(config, "prepare-phase-f", accepted["identity"], checked(config.workspace, accepted["gate"]), inputs,
               {"seed": context["seed"]}) as attempt:
        return attempt.finish({"status": "PASS", "seed": context["seed"], "source_verified_gate": "PASS"})

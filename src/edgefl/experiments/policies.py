"""Translate named, auditable interventions into domain policy objects."""
from dataclasses import replace
from edgefl.optimization.policies import SearchPolicy, OBJECTIVES
from edgefl.trust.policies import TrustPolicy
from edgefl.experiments.matrix import ABLATIONS


def resolve(case, clipping_threshold=None):
    if case.intervention not in ("full", "clipping", *ABLATIONS):
        raise ValueError("Unknown registered intervention")
    search, trust = SearchPolicy(), TrustPolicy()
    name = case.intervention
    if case.method == "random":
        search = replace(search, backend="random")
    if case.method == "adaptive_capped":
        trust = replace(trust, fixed_weight_cap=.5)
    if name == "no_reputation":
        trust = replace(trust, reputation=False)
        search = replace(search, active_objectives=tuple(n for n in OBJECTIVES if n != "historical_unreliability"))
    elif name == "no_direct_risk":
        trust = replace(trust, direct_risk=False)
        search = replace(search, direct_risk=False, active_objectives=tuple(n for n in OBJECTIVES if n != "current_risk"))
    elif name in ("no_expertise", "uniform_random"):
        search = replace(search, initialization=name)
    elif name in ("no_magnitude", "no_direction", "no_loss"):
        trust = replace(trust, signals=tuple(n for n in trust.signals if n != name[3:]))
    elif name.startswith("without_"):
        search = replace(search, active_objectives=tuple(n for n in OBJECTIVES if n != name[8:]))
    elif name == "uncapped":
        search = replace(search, cap=1.)
    elif name == "clipping":
        if clipping_threshold is None:
            raise ValueError("Clipping requires reviewed clean threshold evidence")
        trust = replace(trust, clipping_threshold=clipping_threshold)
    return search, trust

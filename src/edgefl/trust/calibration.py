"""Fit clean-only normalization, then audit an independent trajectory without refitting."""

import math
import numpy as np
from edgefl.contracts.phase_e import SIGNALS
from edgefl.trust.risk import calibrated_risk


def fit(records, scope, fit_seeds=(101, 102)):
    if set(records) != set(fit_seeds) or any(not records[s] for s in fit_seeds):
        raise ValueError("Both registered clean fitting seeds are required")
    samples = [r for seed in fit_seeds for r in records[seed]]
    anchors = {}
    for signal in SIGNALS:
        values = [sample["raw"][signal] for sample in samples]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid calibration signal")
        lower, upper = (float(v) for v in np.quantile(values, [.5, .99], method="linear"))
        anchors[signal] = {"lower": lower, "upper": upper, "degenerate": upper - lower <= 1e-12 * max(1., abs(lower), abs(upper))}
    result = {"schema_version": "phase-e.calibration.v1", "scope": scope, "fit_seeds": list(fit_seeds),
              "anchors": anchors, "quantile_method": "linear", "fit_client_rounds": len(samples)}
    scores = [calibrated_risk(sample["raw"], result)[1] for sample in samples]
    result["flag_threshold"] = float(np.quantile(scores, .99, method="linear"))
    return result


def audit(calibration, samples, profiles, declared_specialist=False):
    if not samples:
        raise ValueError("Independent clean audit is empty")
    groups = {"all_honest": [], "concentrated": [], "declared_specialist_scenario": []}
    by_round, by_client = {}, {}
    for sample in samples:
        score = calibrated_risk(sample["raw"], calibration)[1]
        client, current = sample["client"], sample["round"]
        groups["all_honest"].append(score)
        if profiles[client]["concentrated"]:
            groups["concentrated"].append(score)
        if declared_specialist:
            groups["declared_specialist_scenario"].append(score)
        by_round.setdefault(str(current), []).append(score)
        by_client.setdefault(client, []).append(score)
    def summary(scores):
        n = len(scores)
        flags = sum(s > calibration["flag_threshold"] for s in scores)
        return {"client_rounds": n, "flags": flags, "flag_rate": flags / n if n else None,
                "mean_risk": sum(scores) / n if n else None, "missing_reason": None if n else "no_cohort_coverage"}
    return {"schema_version": "phase-e.calibration-audit.v1", "seed": 103,
            "cohorts": {k: summary(v) for k, v in groups.items()},
            "by_round": {k: summary(v) for k, v in by_round.items()},
            "by_client": {k: summary(v) for k, v in by_client.items()}, "refitted": False}

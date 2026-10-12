"""Explicit raw anomaly signals and frozen clean-excess normalization."""

import math
import statistics
import torch
from edgefl.contracts.phase_e import SIGNALS
from edgefl.learning.metrics import norm
from edgefl.learning.checkpoints import validate_state


def raw_signals(updates, losses, parent_loss, minimum_valid=3):
    ids = sorted(updates)
    if len(ids) < minimum_valid or set(losses) != set(ids):
        raise ValueError("Insufficient valid clients for peer risk")
    template = updates[ids[0]]
    for delta in updates.values():
        validate_state(delta, template)
    median = {}
    for key in sorted(template):
        ordered = torch.stack([updates[c][key].double() for c in ids]).sort(dim=0).values
        median[key] = (ordered[(len(ids) - 1) // 2] + ordered[len(ids) // 2]) / 2
    norms = {c: norm(updates[c]) for c in ids}
    reference, direction_norm = statistics.median(norms.values()), norm(median)
    result = {}
    for client in ids:
        n = norms[client]
        if direction_norm == 0:
            direction = 0.0
        elif n == 0:
            direction = 1.0
        else:
            dot = sum(float(updates[client][k].double().mul(median[k]).sum()) for k in sorted(template))
            direction = (1 - max(-1., min(1., dot / (n * direction_norm)))) / 2
        if not math.isfinite(losses[client]) or not math.isfinite(parent_loss):
            raise ValueError("Nonfinite risk loss")
        result[client] = {"magnitude": max(0., math.log((n + 1e-12) / (reference + 1e-12))),
                          "direction": direction, "loss": max(0., losses[client] - parent_loss),
                          "peer_median_norm": reference, "coordinate_median_norm": direction_norm,
                          "direction_reference_available": direction_norm > 0, "valid_peers_including_self": len(ids)}
    return result


def normalize(value, anchors):
    lo, hi = anchors["lower"], anchors["upper"]
    if not all(math.isfinite(x) for x in (value, lo, hi)) or hi < lo or value < 0:
        raise ValueError("Invalid risk normalization values")
    tolerance = 1e-12 * max(1., abs(lo), abs(hi))
    if hi - lo <= tolerance:
        return float(value > lo + tolerance)
    return max(0., min(1., (value - lo) / (hi - lo)))


def calibrated_risk(raw, calibration):
    if calibration.get("schema_version") != "phase-e.calibration.v1" or set(calibration["anchors"]) != set(SIGNALS):
        raise ValueError("Invalid risk calibration")
    components = {name: normalize(raw[name], calibration["anchors"][name]) for name in SIGNALS}
    return components, sum(components.values()) / len(SIGNALS)

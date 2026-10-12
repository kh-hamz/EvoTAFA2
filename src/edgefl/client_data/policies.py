"""Shared support and predictor policies, independent of pipeline orchestration."""

import hashlib

from edgefl.config import canonical_json
from edgefl.contracts.phase_c import SupportPolicy
from edgefl.data.schema import ADDRESSES, LABELS, NUMERIC, TEXT

FEATURE_POLICY_VERSION = "edgeiiot.predictors.v2"
MANDATORY_EXCLUSIONS = ADDRESSES | frozenset((
    "frame.time", "Attack_label", "Attack_type", "tcp.payload", "http.file_data"))


def support_policy(values):
    specification = values.get("support_policy")
    if specification is None:
        minimum = int(values.get("require_binary_support", False))
        return SupportPolicy(assigned=(minimum, minimum), training=(minimum, minimum))
    pairs = [tuple(specification[role][key] for key in ("benign", "attack"))
             for role in ("assigned", "training", "validation")]
    return SupportPolicy(specification["profile"], *pairs)


def metric_support(counts):
    benign = counts.get("Normal", 0)
    attack = sum(n for label, n in counts.items() if label != "Normal")
    return {"loss_accuracy": bool(benign + attack), "benign_fpr": bool(benign),
            "binary_roc_auc_pr_auc": bool(benign and attack),
            "binary_declared_class_macro_f1": bool(benign and attack)}


def class_counts(counts):
    return {label: counts.get(label, 0) for label in sorted(LABELS)}


def validate_feature_policy(preprocessing):
    excluded = set(preprocessing["identifier_payload_exclusions"])
    strict = set(preprocessing["strict_shortcut_exclusions"])
    if not MANDATORY_EXCLUSIONS <= excluded:
        raise ValueError("Mandatory identifier/payload exclusions missing: " +
                         ", ".join(sorted(MANDATORY_EXCLUSIONS - excluded)))
    unknown = (excluded | strict) - (NUMERIC | TEXT)
    if unknown:
        raise ValueError("Unknown preprocessing fields: " + ", ".join(sorted(unknown)))
    if excluded & strict:
        raise ValueError("Strict exclusions must list only additional fields")


def feature_policy_identity(preprocessing):
    return {"version": FEATURE_POLICY_VERSION,
            "sha256": hashlib.sha256(canonical_json({
                "version": FEATURE_POLICY_VERSION,
                "mandatory": sorted(MANDATORY_EXCLUSIONS),
                "preprocessing": preprocessing,
            }).encode()).hexdigest()}

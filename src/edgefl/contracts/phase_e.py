"""Phase E names and immutable execution records; foundation contracts are unchanged."""

from dataclasses import dataclass

VERSION = "phase-e.v1"
CONDITIONS = ("clean", "untargeted_label", "targeted_label", "sign_flip", "update_scaling", "additive_noise", "intermittent_sign")
METHODS = ("fedavg", "fedprox", "median", "trimmed_mean", "fltrust", "adaptive", "frozen")
SIGNALS = ("magnitude", "direction", "loss")
STAGES = {
    "prepare-phase-e": {"prepared", "phase_d_acceptance"},
    "plan-attacks": {"metadata"},
    "run-clean-pilot": {"metadata", "phase_d_acceptance"},
    "calibrate-risk": {"metadata", "fit_101", "fit_102", "audit_103"},
    "run-trust-federated": {"metadata", "initialization", "phase_d_acceptance", "attack_plan", "calibration", "calibration_review"},
    "assess-round": {"run"},
    "review-phase-e": {"subject"},
    "validate-phase-e": {"metadata", "calibration", "calibration_review"},
}


@dataclass(frozen=True)
class SubmissionDisposition:
    client_id: str
    status: str
    reason: str | None = None

    def __post_init__(self):
        if self.status not in ("valid", "invalid", "absent"):
            raise ValueError("Invalid submission disposition")
        if self.status == "invalid" and not self.reason:
            raise ValueError("Invalid submissions require a reason")


@dataclass(frozen=True)
class ArtifactPolicy:
    phase: str
    configuration_sha256: str
    implementation_sha256: str
